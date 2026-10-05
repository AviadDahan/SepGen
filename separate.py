"""SepGen separation: one video (its soundtrack is the audio-mix) + a caption per source -> one
waveform per source.

The observed video is presented on the audio's noise schedule and the observed audio-mix is held
at sigma 0 at every step; only the two stem spans are denoised. The model is LTX-2.5 22B dev
(int8-quantized, as in training) with the SepGen LoRA, block-diagonal caption routing, the
protected audio-mix span and Cross-Stem Attention Guidance (NAG with the sibling caption as the
negative branch).

Single clip:
    python separate.py --video clip.mp4 \
        --scene-prompt "A man and a woman talk in a kitchen." \
        --stem0-prompt "A man, speaking." --stem1-prompt "A woman, speaking." \
        --out-dir out/clip

Batch (one model load for every item): --batch-manifest items.json, a json list of
    {"name", "video", "scene_prompt", "stem0_prompt", "stem1_prompt", "mix_wav"?: null}
(see examples/separation_manifest.json).

Outputs per item: stem_0.wav, stem_1.wav, mix.wav (when extracted from the video),
first_frame.jpg, repro.mp4 (the rendered video with the audio-mix).

The paper's settings: 30 Euler steps, Cross-Stem Attention Guidance scale 2.0 (tau 2.5,
alpha 0.5), 768x512, the clip's own length (e.g. --num-frames 137 --frame-rate 24 for 24 fps
clips; frames must be 8k+1). Use --vae-tiling above ~185 frames on a 48 GB GPU.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "sepgen"))

AUDIO_TOKENS_PER_SECOND = 25.0
VAE_TEMPORAL_STRIDE = 8    # video frames per latent frame
VAE_SPATIAL_STRIDE = 32    # pixels per latent cell
HF_REPO = "AviadDahan/SepGen"
NAMED_CHECKPOINTS = {"sep-12k": "sep-12k/lora_weights.safetensors",
                     "gen-3k": "gen-3k/lora_weights.safetensors"}


def resolve_checkpoint(name_or_path: str) -> Path:
    """`sep-12k` / `gen-3k` -> downloaded from the Hugging Face repo; anything else is a path."""
    if name_or_path in NAMED_CHECKPOINTS:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(HF_REPO, NAMED_CHECKPOINTS[name_or_path]))
    path = Path(name_or_path)
    if not path.is_file():
        raise SystemExit(f"checkpoint not found: {path}")
    return path


def load_config(path: Path, models_dir: Path) -> dict:
    """The run config with ${SEPGEN_MODELS} expanded to the LTX-2.5 weights directory."""
    os.environ["SEPGEN_MODELS"] = str(models_dir.resolve())
    return yaml.safe_load(os.path.expandvars(path.read_text()))


def letterbox(video: torch.Tensor, target_wh: tuple[int, int]) -> torch.Tensor:
    """[F,C,H,W] in [0,1] -> [F,C,H',W'] aspect-preserving letterbox on black."""
    _, _, h, w = video.shape
    tw, th = target_wh
    scale = min(tw / w, th / h)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    video = torch.nn.functional.interpolate(video, size=(nh, nw), mode="bilinear",
                                            align_corners=False)
    out = torch.zeros(video.shape[0], video.shape[1], th, tw)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    out[:, :, y0:y0 + nh, x0:x0 + nw] = video
    return out


def stub_data_root(root: Path, raw: dict) -> str:
    """The trainer validates that its data root exists even when nothing is read from it."""
    for name in ("latents", "conditions", raw["training_strategy"]["audio_latents_dir"]):
        (root / name).mkdir(parents=True, exist_ok=True)
    return str(root)


def main():  # noqa: PLR0915
    ap = argparse.ArgumentParser(description="SepGen separation")
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--scene-prompt", default=None, help="caption of the whole scene")
    ap.add_argument("--stem0-prompt", default=None, help="caption of the first source")
    ap.add_argument("--stem1-prompt", default=None, help="caption of the second source")
    ap.add_argument("--mix-wav", type=Path, default=None,
                    help="audio-mix to separate (default: the video's own soundtrack)")
    ap.add_argument("--batch-manifest", type=Path, default=None,
                    help="json list of items; replaces the single-clip flags")
    ap.add_argument("--checkpoint", default="sep-12k",
                    help="sep-12k | gen-3k (downloaded from Hugging Face) | path to a LoRA file")
    ap.add_argument("--config", type=Path, default=REPO / "configs/train_sep12k.yaml",
                    help="model + guidance config (both checkpoints share it at inference)")
    ap.add_argument("--models-dir", type=Path, default=Path(os.environ.get(
        "SEPGEN_MODELS", REPO / "models/ltx2.5")), help="LTX-2.5 weights (download_weights.sh)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-frames", type=int, default=None,
                    help="frames to render, 8k+1 (default: config, 113)")
    ap.add_argument("--frame-rate", type=float, default=None, help="default: config, 25")
    ap.add_argument("--width", type=int, default=None, help="default: config, 768")
    ap.add_argument("--height", type=int, default=None, help="default: config, 512")
    ap.add_argument("--nag-scale", type=float, default=2.0,
                    help="Cross-Stem Attention Guidance scale (paper: 2.0; 1.0 disables)")
    ap.add_argument("--nag-tau", type=float, default=2.5)
    ap.add_argument("--nag-alpha", type=float, default=0.5)
    ap.add_argument("--vae-tiling", action="store_true",
                    help="encode the video in temporal tiles (needed above ~185 frames)")
    ap.add_argument("--vae-tile-frames", type=int, default=80)
    ap.add_argument("--vae-tile-overlap", type=int, default=24)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if args.batch_manifest is not None:
        items = json.loads(args.batch_manifest.read_text())
        for it in items:
            it["video"] = Path(it["video"])
    else:
        if not (args.video and args.scene_prompt and args.stem0_prompt and args.stem1_prompt):
            raise SystemExit("single-clip mode needs --video and the three prompts "
                             "(or use --batch-manifest)")
        items = [{"name": None, "video": args.video, "scene_prompt": args.scene_prompt,
                  "stem0_prompt": args.stem0_prompt, "stem1_prompt": args.stem1_prompt,
                  "mix_wav": str(args.mix_wav) if args.mix_wav else None}]

    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "args.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
    dev = torch.device("cuda")

    def item_out(it) -> Path:
        d = out_root if it["name"] is None else out_root / it["name"]
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------ 1. INGESTION
    from ltx_trainer.model_loader import load_video_vae_encoder
    from ltx_trainer.video_utils import read_video
    from ltx_core.types import Audio
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.loader.registry import DummyRegistry
    from ltx_core.model.audio_vae import (
        AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
        AudioEncoderConfigurator,
        encode_audio,
    )

    raw = load_config(args.config, args.models_dir)
    video_vae_path = raw["model"]["video_vae_path"]
    audio_vae_path = raw["model"]["audio_vae_path"]

    cfg_w, cfg_h, cfg_f = raw["validation"]["video_dims"]
    num_frames = args.num_frames if args.num_frames is not None else int(cfg_f)
    frame_rate = (args.frame_rate if args.frame_rate is not None
                  else float(raw["validation"]["frame_rate"]))
    width = args.width if args.width is not None else int(cfg_w)
    height = args.height if args.height is not None else int(cfg_h)
    if (num_frames - 1) % VAE_TEMPORAL_STRIDE:
        raise SystemExit(f"--num-frames must be 8k+1, got {num_frames}")
    raw["validation"]["video_dims"] = [width, height, num_frames]
    raw["validation"]["frame_rate"] = frame_rate
    grid = ((num_frames - 1) // VAE_TEMPORAL_STRIDE + 1,
            height // VAE_SPATIAL_STRIDE, width // VAE_SPATIAL_STRIDE)
    span_len = int(round(num_frames / frame_rate * AUDIO_TOKENS_PER_SECOND))
    print(f"[separate] {width}x{height}x{num_frames} @ {frame_rate} fps -> latent grid {grid}, "
          f"audio span {span_len} tokens", flush=True)

    venc = load_video_vae_encoder(video_vae_path, dev, torch.bfloat16).eval()
    aenc = SingleGPUModelBuilder(
        model_class_configurator=AudioEncoderConfigurator, model_path=audio_vae_path,
        model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
        registry=DummyRegistry()).build(device=torch.device("cpu"),
                                        dtype=torch.float32).eval()
    for it in items:
        out = item_out(it)
        mix_path = Path(it["mix_wav"]) if it.get("mix_wav") else out / "mix.wav"
        if not it.get("mix_wav"):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(it["video"]),
                            "-vn", "-acodec", "pcm_s16le", str(mix_path)], check=True)
        vid, fps = read_video(str(it["video"]))                   # [F, C, H, W] in [0,1]
        if abs(float(fps) - frame_rate) > 0.6:
            print(f"[separate] WARNING: {it['video'].name} fps {fps:.1f} != {frame_rate}",
                  flush=True)
        if vid.shape[0] < num_frames:
            raise SystemExit(f"{it['video']} has {vid.shape[0]} frames < {num_frames}")
        frames = letterbox(vid[:num_frames], (width, height))
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            vin = (frames.permute(1, 0, 2, 3)[None] * 2 - 1).to(dev, torch.float32)
            if args.vae_tiling:
                from ltx_core.tiling import DimensionSizeConfig, TileSizeConfig
                vlat = venc.tiled_encode(vin, TileSizeConfig(
                    frames=DimensionSizeConfig(tile_size=args.vae_tile_frames,
                                               overlap=args.vae_tile_overlap)))
            else:
                vlat = venc(vin)
            del vin
        if tuple(vlat.shape[2:]) != grid:
            raise SystemExit(f"{it['video']}: latent grid {tuple(vlat.shape[2:])} != {grid}")
        it["vlat"] = vlat.cpu()
        del frames, vid
        wav, sr = sf.read(mix_path, dtype="float32", always_2d=True)
        w = torch.from_numpy(wav.T)
        if w.shape[0] == 1:
            w = w.repeat(2, 1)
        with torch.no_grad():
            mix_lat = encode_audio(Audio(waveform=w[None], sampling_rate=sr), aenc)
        if mix_lat.shape[2] < span_len:
            raise SystemExit(f"{it['video']}: audio-mix too short ({mix_lat.shape[2]} tokens "
                             f"< span {span_len})")
        it["mix_lat"] = mix_lat[:, :, :span_len]
        ff = out / "first_frame.jpg"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(it["video"]),
                        "-vf", f"select=eq(n\\,0),scale={width}:{height}", "-vframes", "1",
                        str(ff)], check=True)
        it["ff"] = ff
        print(f"[separate] ingested {it['video'].name}", flush=True)
    del aenc
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ 2. MODEL BUILD
    # The trainer builds the int8 base + LoRA exactly as in training, and its validation runner
    # encodes every caption once: prompts = [scene_i, stem0_i, stem1_i] per item.
    strategy_raw = raw.pop("joint_stem", {})
    val_raw = raw.pop("joint_stem_validation", {})
    raw["model"]["load_checkpoint"] = str(resolve_checkpoint(args.checkpoint))
    raw["wandb"]["enabled"] = False
    raw["output_dir"] = str(out_root / "trainer_workdir")
    raw["data"]["preprocessed_data_root"] = stub_data_root(
        out_root / "trainer_workdir/no_training_data", raw)
    prompts = []
    for it in items:
        prompts += [it["scene_prompt"], it["stem0_prompt"], it["stem1_prompt"]]
    raw["validation"]["prompts"] = prompts

    from stemgen.joint_stems import JointStemConfig, JointStemStrategy
    from stemgen.joint_stem_sampler import (
        DevGuidance,
        JointStemGenerationRequest,
        JointStemSampler,
    )
    import ltx_trainer.trainer as trainer_mod
    from ltx_trainer.config import LtxTrainerConfig

    strategy_cfg = JointStemConfig(**strategy_raw)
    strategy = JointStemStrategy(strategy_cfg)
    trainer_mod.get_training_strategy = lambda _cfg: strategy
    JointStemSampler.joint_num_stems = 2
    JointStemSampler.joint_include_mix = True
    JointStemSampler.joint_span_allow = strategy.span_allow_matrix
    JointStemSampler.joint_dev_guidance = DevGuidance(**val_raw["dev_guidance"])
    trainer = trainer_mod.LtxvTrainer(LtxTrainerConfig(**raw))

    runner = trainer._validation_runner                           # noqa: SLF001
    cached = runner._cached_embeddings                            # noqa: SLF001
    if not cached or cached[0].audio_context_negative is None:
        raise RuntimeError("cached audio negative missing")
    neg_len = int(cached[0].audio_context_negative.shape[1])

    # Attention gates, in the order training installed them.
    from stemgen.text_block_gate import build_block_diagonal_text_bias, install_text_block_gate
    install_text_block_gate(trainer._transformer, negative_context_len=neg_len)  # noqa: SLF001
    import nag_attn2 as nag
    use_nag = args.nag_scale != 1.0
    if use_nag:
        nag.install_nag_attn2(trainer._transformer)               # noqa: SLF001
    if strategy.span_allow_matrix is not None:
        from stemgen.audio_span_gate import install_audio_span_gate
        install_audio_span_gate(trainer._transformer)             # noqa: SLF001
    if strategy_cfg.frozen_mix_lora:
        from stemgen.lora_span_gate import install_lora_span_gate, set_lora_span_gate
        set_lora_span_gate(num_spans=3, gated_span=2)
        install_lora_span_gate(trainer._transformer)              # noqa: SLF001
    if strategy_cfg.a2v_mix_only:
        from stemgen.a2v_mix_gate import install_a2v_mix_gate, set_a2v_mix_gate
        set_a2v_mix_gate(num_spans=3)
        install_a2v_mix_gate(trainer._transformer)                # noqa: SLF001

    from torchvision.transforms import functional as TF  # noqa: N812
    from ltx_trainer.utils import open_image_as_srgb
    from ltx_trainer.video_utils import save_video
    from stemgen.ltx25_validation_sampler import GenerationConfig

    runner._load_decoder_components()                             # noqa: SLF001
    sampler = JointStemSampler(
        transformer=trainer._transformer,                         # noqa: SLF001
        vae_decoder=runner._vae_decoder,                          # noqa: SLF001
        vae_encoder=venc, text_encoder=None,
        audio_decoder=runner._audio_decoder,                      # noqa: SLF001
        vocoder=runner._vocoder,                                  # noqa: SLF001
        sampling_context=None)
    sampler.tiling_provider = runner
    sr_out = runner._vocoder.output_sampling_rate                 # noqa: SLF001
    vcfg = trainer._config.validation                             # noqa: SLF001
    n_steps = int(vcfg.inference_steps)

    # ------------------------------------------------------------------ 3. PER CLIP
    for idx, it in enumerate(items):
        out = item_out(it)
        c_scene, c_s0, c_s1 = cached[3 * idx], cached[3 * idx + 1], cached[3 * idx + 2]
        name = it["name"] or it["video"].stem
        print(f"[separate] === {name} ===", flush=True)

        # Block-diagonal caption routing: [stem0 | stem1 | audio-mix] each read their own text.
        span_ctxs = (c_s0, c_s1, c_scene)
        JointStemSampler.joint_audio_context = torch.cat(
            [c.audio_context_positive for c in span_ctxs], dim=1)
        JointStemSampler.joint_block_masks = [
            torch.ones(1, c.audio_context_positive.shape[1]) for c in span_ctxs]
        if use_nag:
            # Cross-Stem Attention Guidance: the negative branch routes each stem to its
            # sibling's caption (stem0 <- stem1 text, stem1 <- stem0 text); the mix keeps its own.
            _s0, _s1 = c_s0.audio_context_positive, c_s1.audio_context_positive
            _sc = c_scene.audio_context_positive
            _masks = JointStemSampler.joint_block_masks
            nag.set_nag(args.nag_scale, args.nag_tau, args.nag_alpha,
                        null_ctx=torch.cat([_s1, _s0, _sc], dim=1),
                        null_bias=build_block_diagonal_text_bias(
                            span_lens=[span_len] * 3,
                            block_masks=[_masks[1].to(dev), _masks[0].to(dev),
                                         _masks[2].to(dev)],
                            dtype=_s0.dtype, device=dev))

        # Observed video on the audio's noise schedule; observed audio-mix clean at every step.
        JointStemSampler.joint_observed_video = strategy._video_patchifier.patchify(  # noqa: SLF001
            it["vlat"].to(torch.bfloat16))
        JointStemSampler.joint_observed_video_schedule = True
        mix_tokens = strategy._audio_patchifier.patchify(           # noqa: SLF001
            it["mix_lat"].to(torch.float32))
        JointStemSampler.joint_x0_guidance = None
        request = JointStemGenerationRequest(
            ref_time_offset=strategy_cfg.ref_time_offset, mix_lead_alpha=1.0,
            teacher_force_mix_latent=mix_tokens, teacher_force_seed=args.seed,
            mix_sigma_track=torch.zeros(n_steps + 1),
            span_pe_layout=strategy_cfg.span_pe_layout)
        sampler.audio_conditionings = [request]
        gen_config = GenerationConfig(
            prompt=it["scene_prompt"], negative_prompt=vcfg.negative_prompt,
            height=height, width=width, num_frames=num_frames, frame_rate=vcfg.frame_rate,
            num_inference_steps=n_steps, guidance_scale=vcfg.video_cfg_scale,
            seed=args.seed,
            condition_image=TF.to_tensor(open_image_as_srgb(str(it["ff"]))),
            generate_audio=True, cached_embeddings=c_scene,
            stg_scale=vcfg.video_stg_scale, stg_blocks=vcfg.stg_blocks)
        video_out, primary = sampler.generate(config=gen_config, device=dev)

        slots = sampler.last_slot_audio
        for k in range(2):
            wav_k = slots[f"generated_stem{k}"].detach().cpu()
            arr = wav_k.T.numpy() if wav_k.ndim > 1 else wav_k.numpy()
            sf.write(out / f"stem_{k}.wav", arr, sr_out)
            rms = float(np.sqrt(np.mean(arr ** 2)))
            print(f"[separate]   stem_{k}.wav ({20 * np.log10(max(rms, 1e-9)):+.1f} dBFS)",
                  flush=True)
        save_video(video_tensor=video_out, output_path=out / "repro.mp4",
                   fps=vcfg.frame_rate, audio=primary, audio_sample_rate=sr_out)
        it.pop("vlat", None), it.pop("mix_lat", None)
        print(f"[separate]   wrote {out}/", flush=True)

    print("[separate] done", flush=True)


if __name__ == "__main__":
    main()
