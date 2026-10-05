#!/usr/bin/env python
"""New-view render of one clip with the LTX-2.3 CrossView-Warp IC-LoRA (demo stage 3).

Conditions LTX-2.3's ICLoraPipeline on two reference videos, (original clip, depth-warp guide),
with Cseti's CrossView-Warp v2 IC-LoRA and its trigger prompt "crossview", and renders the same
scene under the authored moving camera at 1536x1024. Writes `render.mp4` (with the pipeline's own
generated audio) and `render_mixaudio.mp4` (the clip's original audio-mix muxed on instead).

Weights: the base is the LTX-2.3 22B DEV checkpoint with the distilled LoRA (384, v1.1) fused at
1.0 into both stages, i.e. the distilled sampling recipe; the CrossView LoRA is stage-1-only, as
in upstream. Upstream ICLoraPipeline builds stage 2 LoRA-free on the same base (right for a
distilled checkpoint), so stage 2 is rebuilt here with the distilled LoRA fused. Stage 2 starts
from sigma 0.421875 instead of upstream's 0.909375: with the bare trigger prompt, the upstream
start re-invents content (extra objects and people in flat regions), while 0.421875 refines
detail only.

Memory on a 48 GB GPU (sm_86, where fp8 is unavailable): TWO PROCESSES. The first run finds no
prompt cache, encodes the constant trigger prompt with pipeline-wide CPU offload, saves the
context (prompt_ctx.pt) and exits -- the process boundary is what releases Gemma's memory. The
second run renders with the cached context, streaming the stage weights from CPU layer by layer,
and with the reference-guide VAE encode and the decode tiled.

Env: LTX-2.3 (setup_demo.sh), one GPU. Run the same command twice:
  python demo/render_crossview.py --video clip.mp4 --warp out/guide/warp.mp4 \
      --mix-wav mix.wav --seed 90115 --out-dir out/render
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch

from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae.tiling import (SpatialTilingConfig,
                                             TemporalTilingConfig)
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.utils.blocks import DiffusionStage
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import OffloadMode

REPO = Path(__file__).resolve().parents[1]
DEV_CHECKPOINT = "ltx-2.3-22b-dev.safetensors"
DISTILLED_LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
CROSSVIEW_LORA = "loras/LTX2.3-22B_IC-LoRA-CrossView-Warp_v2_6000.safetensors"
SPATIAL_UPSAMPLER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_DIR = "gemma3"


def nb_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True).stdout.strip()
    return int(out)


def mux_original_mix(render: Path, mix_wav: Path, out: Path) -> None:
    # NO -shortest: the mix can be a few ms shorter than the video and -shortest
    # would drop the last frame.
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(render), "-i", str(mix_wav),
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
         "-b:a", "192k", "-movflags", "+faststart", str(out)], check=True)


@torch.inference_mode()   # as upstream ic_lora.main(); without it every forward keeps
# autograd graphs and the reference-guide encode alone runs out of memory
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", type=Path, required=True, help="the original clip")
    ap.add_argument("--warp", type=Path, required=True,
                    help="build_warp_guide.py output (warp.mp4)")
    ap.add_argument("--mix-wav", type=Path, required=True,
                    help="audio muxed onto render_mixaudio.mp4")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--models-dir", type=Path, default=Path(os.environ.get(
        "LTX23_MODELS", REPO / "models/ltx2.3")),
        help="LTX-2.3 weights (download_demo_weights.sh)")
    ap.add_argument("--checkpoint-path", default=None,
                    help=f"base checkpoint (default: <models-dir>/{DEV_CHECKPOINT})")
    ap.add_argument("--distilled-lora-path", default=None,
                    help=f"default: <models-dir>/{DISTILLED_LORA}")
    ap.add_argument("--distilled-lora-strength-stage-1", type=float, default=1.0,
                    help="distilled LoRA strength fused into stage 1; 1.0 = "
                         "the distilled sampling recipe (default: 1.0)")
    ap.add_argument("--distilled-lora-strength-stage-2", type=float, default=1.0,
                    help="distilled LoRA strength fused into the rebuilt "
                         "stage 2 (default: 1.0)")
    ap.add_argument("--crossview-lora-path", default=None,
                    help=f"default: <models-dir>/{CROSSVIEW_LORA}")
    ap.add_argument("--crossview-lora-strength", type=float, default=1.0,
                    help="CrossView IC-LoRA strength, stage 1 only; raise to "
                         "1.2-1.3 if the camera move is weak (default: 1.0)")
    ap.add_argument("--spatial-upsampler-path", default=None,
                    help=f"default: <models-dir>/{SPATIAL_UPSAMPLER}")
    ap.add_argument("--gemma-root", default=None,
                    help=f"Gemma-3 directory (default: <models-dir>/{GEMMA_DIR})")
    ap.add_argument("--prompt", default="crossview",
                    help='IC-LoRA trigger prompt (default: "crossview")')
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--num-frames", type=int, default=121)
    ap.add_argument("--frame-rate", type=float, default=24.0)
    ap.add_argument("--cond-strength-original", type=float, default=1.0)
    ap.add_argument("--cond-strength-warp", type=float, default=1.0)
    ap.add_argument("--conditioning-attention-strength", type=float, default=1.0)
    ap.add_argument("--skip-stage-2", action="store_true",
                    help="stage 1 only -> half-res output (fast smoke)")
    ap.add_argument("--quantization", choices=("fp8-cast", "none"),
                    default="none",
                    help="transformer quantization (default none; fp8-cast needs an "
                         "Ada/Hopper GPU)")
    ap.add_argument("--offload-mode", choices=("none", "cpu"), default="cpu",
                    help="weight offloading FOR THE DIFFUSION STAGES ONLY: cpu (default) "
                         "streams the stage weights layer by layer, which is what fits the "
                         "IC-LoRA token load on 48 GB")
    ap.add_argument("--vae-tile-pixels", type=int, default=384,
                    help="video-VAE spatial tile (default 384)")
    ap.add_argument("--vae-tile-frames", type=int, default=48,
                    help="video-VAE temporal tile in frames (default 48)")
    ap.add_argument("--stage2-start-sigma", type=float, default=0.421875,
                    choices=(0.909375, 0.725, 0.421875, 0.0),
                    help="stage-2 re-noise start (values from the distilled "
                         "sigma grid). 0.421875 (default) refines detail without "
                         "inventing content; 0.909375 is upstream's; 0.0 = pure "
                         "upsample+decode")
    ap.add_argument("--prompt-cache", type=Path, default=None,
                    help="cached prompt context (default: <out-dir>/prompt_ctx.pt). "
                         "Missing -> encode + save + exit; present -> render")
    args = ap.parse_args()
    md = args.models_dir
    checkpoint_path = args.checkpoint_path or str(md / DEV_CHECKPOINT)
    distilled_lora_path = args.distilled_lora_path or str(md / DISTILLED_LORA)
    crossview_lora_path = args.crossview_lora_path or str(md / CROSSVIEW_LORA)
    spatial_upsampler_path = args.spatial_upsampler_path or str(md / SPATIAL_UPSAMPLER)
    gemma_root = args.gemma_root or str(md / GEMMA_DIR)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    comfy = LTXV_LORA_COMFY_RENAMING_MAP
    stage_1_loras = [
        LoraPathStrengthAndSDOps(distilled_lora_path,
                                 args.distilled_lora_strength_stage_1, comfy),
        LoraPathStrengthAndSDOps(crossview_lora_path,
                                 args.crossview_lora_strength, comfy),
    ]
    quant = (None if args.quantization == "none"
             else QuantizationKind(args.quantization).to_policy(checkpoint_path))
    offload = OffloadMode.CPU if args.offload_mode == "cpu" else OffloadMode.NONE
    prompt_cache = args.prompt_cache or (args.out_dir / "prompt_ctx.pt")

    # Three memory peaks that cannot coexist on 48 GB: the prompt encoder (Gemma + connector),
    # the reference-guide VAE encode, and the 44 GB bf16 transformer. Phase A (no cache file):
    # pipeline-wide offload, encode the constant trigger prompt once, save, EXIT. Phase B:
    # offload-free pipeline whose prompt encoder is a cache loader (Gemma never loads), stages
    # rebuilt with CPU weight streaming, encode/decode tiled.
    if not prompt_cache.is_file():
        print(f"[crossview] PHASE A: encoding prompt {args.prompt!r} "
              f"(pipeline-wide offload) -> {prompt_cache}", flush=True)
        pipeline = ICLoraPipeline(
            distilled_checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=gemma_root,
            loras=[],
            offload_mode=OffloadMode.CPU)
        (ctx,) = pipeline.prompt_encoder([args.prompt])
        prompt_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ctx, prompt_cache)
        print(f"[crossview] PHASE A done: prompt context saved "
              f"({prompt_cache.stat().st_size / 2**20:.1f} MiB). "
              f"Run the same command again to render.", flush=True)
        return

    (args.out_dir / "args.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()},
         "torch": torch.__version__,
         "stage_2": "rebuilt with distilled LoRA fused (see docstring)"},
        indent=2))
    print(f"[crossview] PHASE B: rendering with cached prompt context "
          f"{prompt_cache}", flush=True)
    pipeline = ICLoraPipeline(
        distilled_checkpoint_path=checkpoint_path,
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=stage_1_loras,
        quantization=quant)
    cached_prompt_ctx = torch.load(prompt_cache, map_location=pipeline.device,
                                   weights_only=False)
    pipeline.prompt_encoder = lambda prompts, **kw: [cached_prompt_ctx] * len(prompts)
    pipeline.stage_1 = DiffusionStage(
        checkpoint_path, pipeline.dtype, pipeline.device,
        loras=tuple(stage_1_loras), quantization=quant, offload_mode=offload)
    if not args.skip_stage_2:
        # dev base: upstream's LoRA-free stage 2 would refine with the pure dev model on
        # distilled sigmas; fuse the distilled LoRA instead (the CrossView LoRA is not fused
        # here -- upstream drops reference conditioning in stage 2).
        pipeline.stage_2 = DiffusionStage(
            checkpoint_path, pipeline.dtype, pipeline.device,
            loras=(LoraPathStrengthAndSDOps(
                distilled_lora_path,
                args.distilled_lora_strength_stage_2, comfy),),
            quantization=quant,
            offload_mode=offload)

    # Tile the reference-guide VAE encode. Upstream's helper supports a tiled encode but
    # ic_lora.py passes tiling_config=None at its call site; inject ours through the imported
    # symbol.
    encode_tiling = TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=args.vae_tile_pixels, tile_overlap_in_pixels=64),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=args.vae_tile_frames, tile_overlap_in_frames=24))
    import ltx_pipelines.ic_lora as _ic_mod
    _orig_append = _ic_mod.append_ic_lora_reference_video_conditionings

    def _append_with_tiling(*a, **kw):
        kw["tiling_config"] = encode_tiling
        return _orig_append(*a, **kw)

    _ic_mod.append_ic_lora_reference_video_conditionings = _append_with_tiling
    print(f"[crossview] pipeline up: base={Path(checkpoint_path).name}, "
          f"stage1 loras=[distilled@{args.distilled_lora_strength_stage_1}, "
          f"crossview@{args.crossview_lora_strength}], "
          f"stage2 distilled@{args.distilled_lora_strength_stage_2} "
          f"(skip={args.skip_stage_2}), ref_downscale="
          f"{pipeline.reference_downscale_factor}", flush=True)

    render = args.out_dir / "render.mp4"
    guides = [(str(args.video), args.cond_strength_original),
              (str(args.warp), args.cond_strength_warp)]
    print(f"[crossview] seed {args.seed}, guides {[Path(p).name for p, _ in guides]}",
          flush=True)
    tiling_config = TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=args.vae_tile_pixels,
            tile_overlap_in_pixels=64),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=args.vae_tile_frames,
            tile_overlap_in_frames=24))
    stage_2_sigmas = torch.tensor(
        [s for s in (0.909375, 0.725, 0.421875, 0.0)
         if s <= args.stage2_start_sigma + 1e-9])
    video, audio = pipeline(
        prompt=args.prompt, seed=args.seed,
        height=args.height, width=args.width,
        num_frames=args.num_frames, frame_rate=args.frame_rate,
        images=[], video_conditioning=guides,
        tiling_config=tiling_config,
        conditioning_attention_strength=args.conditioning_attention_strength,
        stage_2_sigmas=stage_2_sigmas,
        skip_stage_2=args.skip_stage_2)
    encode_video(video=video, fps=args.frame_rate, audio=audio,
                 output_path=str(render),
                 video_chunks_number=get_video_chunks_number(
                     args.num_frames, tiling_config))
    n = nb_frames(render)
    assert n == args.num_frames, f"render has {n} frames"
    mux_original_mix(render, args.mix_wav, args.out_dir / "render_mixaudio.mp4")
    n2 = nb_frames(args.out_dir / "render_mixaudio.mp4")
    assert n2 == args.num_frames, f"mixaudio mux has {n2} frames"
    (args.out_dir / "clip_meta.json").write_text(json.dumps({
        "seed": args.seed, "guides": guides, "skip_stage_2": args.skip_stage_2},
        indent=1))
    print(f"[crossview] done ({n} frames) -> {render}", flush=True)


if __name__ == "__main__":
    main()
