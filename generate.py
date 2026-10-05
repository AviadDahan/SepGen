"""SepGen generation: a scene caption + one caption per source -> video, audio-mix and one waveform
per source, in one sampling run.

    python generate.py examples/generation_prompts.json --out-dir outputs/generation
    python generate.py examples/generation_prompts.json --cells turn_taking_1
    python generate.py prompts.json --stock          # SepGen off: stock LTX-2.5, no stems

The prompts file is a json list of
    {"segment": name, "seed": int, "scene_prompt": str, "stem_prompts": [str, str]}
and the model, geometry and method settings come from --config (configs/generation.json, the
paper's settings: two-stage 1536x1024, 113 frames at 25 fps, res_2s 15 steps, Estimated
Separation below sigma 0.97, Cross-Stem Attention Guidance 2.0).

Each cell writes video.mp4 (with the generated audio-mix), stem0.wav, stem1.wav,
mix_generated.wav, one muxed mp4 per track, and sources_split_ears.mp4 (one source per ear).
The output directory is created as `<dir>_RUNNING` and renamed once every cell has succeeded.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "sepgen"))

from ltx_core.components.guiders import MultiModalGuiderParams  # noqa: E402
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps  # noqa: E402
from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number  # noqa: E402
from ltx_pipelines.utils.constants import LTX_2_3_HQ_PARAMS  # noqa: E402
from ltx_pipelines.utils.media_io import encode_video  # noqa: E402
from ltx_pipelines.utils.model_paths import ModelPaths  # noqa: E402

from stempipe.artifacts import (  # noqa: E402
    dbfs, ensure_faststart, save_stem_result, span_names,
)
from stempipe.stem_config import StemConfig  # noqa: E402
from stempipe.stem_two_stages_hq import StemTwoStagesHQPipeline  # noqa: E402
from checkpoints import resolve_checkpoint  # noqa: E402

logger = logging.getLogger(__name__)
def guider_params(section: dict | None, fallback: MultiModalGuiderParams) -> MultiModalGuiderParams:
    return replace(fallback, **section) if section else fallback


def select_cells(cells: list[dict], only: list[str] | None) -> list[dict]:
    if only:
        by_id = {c["segment"]: c for c in cells}
        missing = [s for s in only if s not in by_id]
        if missing:
            raise SystemExit(f"cells not in the prompts file: {missing}")
        cells = [by_id[s] for s in only]
    for c in cells:
        for key in ("segment", "scene_prompt", "stem_prompts"):
            if not c.get(key):
                raise SystemExit(f"cell {c.get('segment', '?')} is missing {key!r}")
    return cells


def generate(cfg: dict, cells: list[dict], out_dir: Path, *, stock: bool = False) -> Path:
    """Render every cell into `out_dir`; returns the final (renamed) directory."""
    run_cfg, models = cfg["run"], cfg["models"]
    mode = "stock" if stock else "stem"

    final = Path(out_dir).resolve()
    run = final.parent / (final.name + "_RUNNING")
    if final.exists() or run.exists():
        raise SystemExit(f"{final} (or its _RUNNING twin) exists; choose a new --out-dir")
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({**cfg, "mode": mode, "cells": cells}, indent=2))
    print(f"[generate] {mode}, {len(cells)} cell(s) -> {final}", flush=True)

    model_paths = ModelPaths.from_split(
        transformer_path=models["transformer"], text_encoder_path=models["text_encoder"],
        video_vae_path=models["video_vae"], audio_vae_path=models["audio_vae"],
    )
    distilled = [LoraPathStrengthAndSDOps(models["distilled_lora"], 1.0,
                                          LTXV_LORA_COMFY_RENAMING_MAP)]

    # The stem transformer (base + fused distilled LoRA + the SepGen LoRA + the attention gates)
    # is built once and reused across cells; stock mode never builds it.
    stem_template = None
    stem_transformer = stem_builder = None
    if not stock:
        stem_template = StemConfig.from_dict(cfg["stem"])
        probe = replace(stem_template, stem_prompts=tuple(cells[0]["stem_prompts"]))
        probe.validate()
        from stempipe.model_build import build_stem_transformer

        stem_transformer, stem_builder = build_stem_transformer(
            model_paths=model_paths, distilled_lora=distilled,
            distilled_lora_strength=run_cfg["distilled_strength_stage_1"],
            adapter_checkpoint=str(resolve_checkpoint(models["adapter_checkpoint"])),
            adapter_rank=models.get("adapter_rank", 128),
            adapter_alpha=models.get("adapter_alpha", 128),
            stem=probe,
        )

    pipeline = StemTwoStagesHQPipeline(
        model_paths=model_paths, distilled_lora=distilled,
        distilled_lora_strength_stage_1=run_cfg["distilled_strength_stage_1"],
        distilled_lora_strength_stage_2=run_cfg["distilled_strength_stage_2"],
        spatial_upsampler_path=models["spatial_upsampler"], loras=(),
        stem_transformer=stem_transformer, stem_builder=stem_builder,
    )

    v_params = guider_params(cfg.get("video_guider"), LTX_2_3_HQ_PARAMS.video_guider_params)
    a_params = guider_params(cfg.get("audio_guider"), LTX_2_3_HQ_PARAMS.audio_guider_params)
    levels: dict[str, dict[str, float]] = {}

    for i, cell in enumerate(cells, 1):
        t0 = time.time()
        cell_dir = run / cell["segment"]
        cell_dir.mkdir(parents=True, exist_ok=True)

        stem_cfg = None
        if stem_template is not None:
            cell_negs = cell.get("stem_negative_prompts")
            stem_cfg = replace(
                stem_template,
                stem_prompts=tuple(cell["stem_prompts"]),
                stem_negative_prompts=(tuple(cell_negs) if cell_negs
                                       else stem_template.stem_negative_prompts))
            stem_cfg.validate()
        logger.info("[%d/%d] %s (%s)", i, len(cells), cell["segment"], mode)

        seed = int(cell.get("seed", run_cfg["seed"]))
        video, audio, num_frames, tiling, stem_result = pipeline(
            prompt=cell["scene_prompt"], negative_prompt=run_cfg["negative_prompt"],
            seed=seed, height=run_cfg["height"], width=run_cfg["width"],
            frame_rate=run_cfg["frame_rate"], num_frames=run_cfg["num_frames"],
            num_inference_steps=run_cfg["num_inference_steps"],
            video_guider_params=v_params, audio_guider_params=a_params,
            images=[], tiling_config=AUTO_TILING, stem=stem_cfg,
        )
        video_path = cell_dir / "video.mp4"
        encode_video(video=video, fps=run_cfg["frame_rate"], audio=audio,
                     output_path=str(video_path),
                     video_chunks_number=get_video_chunks_number(num_frames, tiling))
        ensure_faststart(video_path)

        prompts = {"segment": cell["segment"], "scene_prompt": cell["scene_prompt"],
                   "stem_prompts": {f"stem{k}": p for k, p in enumerate(cell["stem_prompts"])},
                   "negative_prompt": run_cfg["negative_prompt"], "mode": mode, "seed": seed}
        if stem_result is not None:
            wavs = save_stem_result(stem_result, cell_dir, fps=run_cfg["frame_rate"],
                                    video_path=video_path, prompts=prompts)
            names = span_names(len(stem_result.spans) - 1)
            levels[cell["segment"]] = {n: round(dbfs(w), 1) for n, w in zip(names, wavs)}
        else:
            (cell_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))
            levels[cell["segment"]] = {}
        (run / "levels.json").write_text(json.dumps(levels, indent=2))
        logger.info("    %s in %.0fs -> %s", cell["segment"], time.time() - t0, cell_dir)

    os.replace(run, final)
    return final


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="SepGen generation")
    ap.add_argument("prompts", help="json list of cells (see examples/generation_prompts.json)")
    ap.add_argument("--config", type=Path, default=REPO / "configs/generation.json")
    ap.add_argument("--models-dir", type=Path, default=Path(os.environ.get(
        "SEPGEN_MODELS", REPO / "models/ltx2.5")), help="LTX-2.5 weights (download_weights.sh)")
    ap.add_argument("--checkpoint", default=None,
                    help="override the config's adapter: gen-3k | sep-12k | path")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: outputs/generation/<prompts name>_<time>")
    ap.add_argument("--stock", action="store_true",
                    help="render with SepGen off: plain LTX-2.5, no stems")
    ap.add_argument("--cells", nargs="*", default=None, help="render only these segments")
    args = ap.parse_args()

    os.environ["SEPGEN_MODELS"] = str(args.models_dir.resolve())
    cfg = json.loads(os.path.expandvars(args.config.read_text()))
    if args.checkpoint is not None:
        cfg["models"]["adapter_checkpoint"] = args.checkpoint
    cells = select_cells(json.loads(Path(args.prompts).read_text()), args.cells)
    out_dir = args.out_dir or (REPO / "outputs/generation"
                               / f"{Path(args.prompts).stem}_{datetime.now():%Y%m%d_%H%M%S}")
    final = generate(cfg, cells, out_dir, stock=args.stock)
    print(f"[generate] done -> {final}", flush=True)
    return 0


if __name__ == "__main__":
    with torch.inference_mode():
        raise SystemExit(main())
