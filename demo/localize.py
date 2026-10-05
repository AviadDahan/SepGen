"""SepGen separation + localization of one clip (demo stage 1).

The same separation render as separate.py (observed video on the audio's noise schedule, observed
audio-mix clean at every step, the two stem spans denoised; sep-12k, 30 Euler steps, Cross-Stem
Attention Guidance 2.0), with attention capture hooks recording two localization channels during
that same render (sepgen/localize/capture.py):

  t2v   per-stem output-space saliency of the video's text cross-attention, replayed with each
        stem's own caption;
  v2a   the stem spans' attention onto the video tokens.

Readout (sepgen/localize/readout.py): both channels are averaged over the sigma ladder and the
block band 26-33 + 36 into per-stem maps; per stem the calmer channel is picked by a GT-free
suspicion score against frozen statistics (sepgen/localize/sampler_stats.json); its maps give a
support-gated geometric-median pointer per latent frame, smoothed by a constant-velocity Kalman
filter + RTS. A stem whose suspicion exceeds the frozen threshold is flagged low-confidence.

Outputs (--out-dir): stem_0.wav, stem_1.wav, mix.wav (when extracted from the video),
first_frame.jpg, repro.mp4, track.json (per stem: channel, suspicion, flag, smoothed pointer per
latent frame in cells and pixels), maps.npz (the chosen-channel maps of both channels),
overlay.png.

  python demo/localize.py --video clip.mp4 --mix-wav mix.wav \
      --scene-prompt "..." --stem0-prompt "..." --stem1-prompt "..." \
      --num-frames 121 --frame-rate 24 --out-dir out/localize
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))               # sepgen.localize
sys.path.insert(0, str(REPO / "sepgen"))    # stemgen, nag_attn2, checkpoints

from sepgen.localize import capture as csa  # noqa: E402  (sets LTX_MASKED_ATTENTION first)
from sepgen.localize import readout  # noqa: E402
from sepgen.localize.readout import (  # noqa: E402
    BAND,
    EPS,
    Q_GRID,
    TOPQ,
    content_columns,
    kalman_forward,
    measurements,
    rts_backward,
    sharpen,
    stem_signals,
    to_prob,
)
from checkpoints import resolve_checkpoint  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Cap = csa.Cap
N_STEMS = 2
AUDIO_TOKENS_PER_SECOND = 25.0
VAE_TEMPORAL_STRIDE = 8    # video frames per latent frame
VAE_SPATIAL_STRIDE = 32    # pixels per latent cell
STATS_JSON = REPO / "sepgen/localize/sampler_stats.json"
SIGNAL_DIRECTION = {"jitter": True, "meas_sigma": True, "tconsist": False}


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


def build_maps(t2v_out_content: np.ndarray, v2a_maps: np.ndarray) -> dict:
    """Ladder arrays [S,48,...] -> the two channels' deployable maps2 [2,F,H,W]."""
    pm = t2v_out_content.mean(axis=0)
    t2v = sharpen(to_prob(pm[list(BAND)].mean(axis=0), "out_content"), TOPQ)
    v = v2a_maps.mean(axis=0)[list(BAND)].mean(axis=0)             # [3, F, H, W]
    stems = v[:2] + EPS
    v2a = sharpen(stems / stems.sum(axis=0, keepdims=True), TOPQ)
    return {"t2v": t2v, "v2a": v2a}


def suspicion(maps2: np.ndarray, k: int, stats: dict) -> tuple[float, dict]:
    sig = stem_signals(maps2[k], maps2[1 - k])
    zs = []
    for nm, hi_bad in SIGNAL_DIRECTION.items():
        zval = (sig[nm] - stats[nm][0]) / stats[nm][1]
        zs.append(zval if hi_bad else -zval)
    return float(np.mean(zs)), sig


def main():  # noqa: PLR0915
    ap = argparse.ArgumentParser(description="SepGen separation + localization of one clip")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--scene-prompt", required=True, help="caption of the whole scene")
    ap.add_argument("--stem0-prompt", required=True, help="caption of the first source")
    ap.add_argument("--stem1-prompt", required=True, help="caption of the second source")
    ap.add_argument("--mix-wav", type=Path, default=None,
                    help="audio-mix to separate (default: the video's own soundtrack)")
    ap.add_argument("--checkpoint", default="sep-12k",
                    help="sep-12k (downloaded from Hugging Face) | path to a LoRA file")
    ap.add_argument("--config", type=Path, default=REPO / "configs/train_sep12k.yaml",
                    help="model + guidance config")
    ap.add_argument("--models-dir", type=Path, default=Path(os.environ.get(
        "SEPGEN_MODELS", REPO / "models/ltx2.5")), help="LTX-2.5 weights (download_weights.sh)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-frames", type=int, default=None,
                    help="frames to render, 8k+1 (default: config, 113; the demo uses 121)")
    ap.add_argument("--frame-rate", type=float, default=None,
                    help="default: config, 25 (the demo uses 24)")
    ap.add_argument("--width", type=int, default=None, help="default: config, 768")
    ap.add_argument("--height", type=int, default=None, help="default: config, 512")
    ap.add_argument("--nag-scale", type=float, default=2.0,
                    help="Cross-Stem Attention Guidance scale (paper: 2.0; 1.0 disables)")
    ap.add_argument("--nag-tau", type=float, default=2.5)
    ap.add_argument("--nag-alpha", type=float, default=0.5)
    ap.add_argument("--stats-json", type=Path, default=STATS_JSON,
                    help="frozen GT-free channel-selection statistics + flag threshold")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
    dev = torch.device("cuda")
    stats_blob = json.loads(args.stats_json.read_text())
    stats = stats_blob["stats"]
    flag_z = float(stats_blob["flag_z"])
    print(f"[localize] stats {args.stats_json.name}, flag_z {flag_z:+.2f}", flush=True)

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
    gemma_path = raw["model"]["text_encoder_path"]
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
    n_video_tokens = int(np.prod(grid))
    span_len = int(round(num_frames / frame_rate * AUDIO_TOKENS_PER_SECOND))
    readout.GRID = grid    # the readout's frame loop and cell grid
    print(f"[localize] {width}x{height}x{num_frames} @ {frame_rate} fps -> latent grid {grid} "
          f"({n_video_tokens} video tokens), audio span {span_len}", flush=True)

    venc = load_video_vae_encoder(video_vae_path, dev, torch.bfloat16).eval()
    aenc = SingleGPUModelBuilder(
        model_class_configurator=AudioEncoderConfigurator, model_path=audio_vae_path,
        model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
        registry=DummyRegistry()).build(device=torch.device("cpu"),
                                        dtype=torch.float32).eval()
    mix_path = args.mix_wav if args.mix_wav else out / "mix.wav"
    if not args.mix_wav:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(args.video),
                        "-vn", "-acodec", "pcm_s16le", str(mix_path)], check=True)
    vid, fps = read_video(str(args.video))                        # [F, C, H, W] in [0,1]
    if abs(float(fps) - frame_rate) > 0.6:
        print(f"[localize] WARNING: {args.video.name} fps {fps:.1f} != {frame_rate}", flush=True)
    if vid.shape[0] < num_frames:
        raise SystemExit(f"{args.video} has {vid.shape[0]} frames < {num_frames}")
    frames = letterbox(vid[:num_frames], (width, height))
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        vin = (frames.permute(1, 0, 2, 3)[None] * 2 - 1).to(dev, torch.float32)
        vlat = venc(vin)
        del vin
    if tuple(vlat.shape[2:]) != grid:
        raise SystemExit(f"{args.video}: latent grid {tuple(vlat.shape[2:])} != {grid}")
    vlat = vlat.cpu()
    del frames, vid
    wav, sr = sf.read(mix_path, dtype="float32", always_2d=True)
    w = torch.from_numpy(wav.T)
    if w.shape[0] == 1:
        w = w.repeat(2, 1)
    with torch.no_grad():
        mix_lat = encode_audio(Audio(waveform=w[None], sampling_rate=sr), aenc)
    if mix_lat.shape[2] < span_len:
        raise SystemExit(f"{args.video}: audio-mix too short ({mix_lat.shape[2]} tokens "
                         f"< span {span_len})")
    mix_lat = mix_lat[:, :, :span_len]
    ff = out / "first_frame.jpg"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(args.video),
                    "-vf", f"select=eq(n\\,0),scale={width}:{height}", "-vframes", "1",
                    str(ff)], check=True)
    print(f"[localize] ingested {args.video.name}", flush=True)
    del aenc
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ 2. MODEL BUILD
    # The trainer builds the int8 base + LoRA exactly as in training, and its validation runner
    # encodes every caption once: prompts = [scene, stem0, stem1].
    strategy_raw = raw.pop("joint_stem", {})
    val_raw = raw.pop("joint_stem_validation", {})
    raw["model"]["load_checkpoint"] = str(resolve_checkpoint(args.checkpoint))
    raw["wandb"]["enabled"] = False
    raw["output_dir"] = str(out / "trainer_workdir")
    raw["data"]["preprocessed_data_root"] = stub_data_root(
        out / "trainer_workdir/no_training_data", raw)
    raw["validation"]["prompts"] = [args.scene_prompt, args.stem0_prompt, args.stem1_prompt]

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
    JointStemSampler.joint_num_stems = N_STEMS
    JointStemSampler.joint_include_mix = True
    JointStemSampler.joint_span_allow = strategy.span_allow_matrix
    JointStemSampler.joint_dev_guidance = DevGuidance(**val_raw["dev_guidance"])
    trainer = trainer_mod.LtxvTrainer(LtxTrainerConfig(**raw))

    runner = trainer._validation_runner                           # noqa: SLF001
    cached = runner._cached_embeddings                            # noqa: SLF001
    if not cached or cached[0].audio_context_negative is None:
        raise RuntimeError("cached audio negative missing")
    neg_len = int(cached[0].audio_context_negative.shape[1])

    # Attention gates, in the order training installed them; capture hooks last.
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
    _, n_blocks = csa.install_capture(trainer._transformer)       # noqa: SLF001
    csa.install_context_modulation_probe()
    trainer._transformer.register_forward_pre_hook(               # noqa: SLF001
        csa.transformer_pre_hook, with_kwargs=True)
    Cap.neg_len = neg_len
    Cap.grid = grid
    Cap.video_tokens = n_video_tokens
    Cap.out_space = True

    from torchvision.transforms import functional as TF  # noqa: N812
    from ltx_trainer.utils import open_image_as_srgb
    from ltx_trainer.video_utils import save_video
    from stemgen.ltx25_validation_sampler import GenerationConfig
    # The tokenizer is built from the same file as the text encoder, so the token counts below
    # match the encoder's own tokenization (stem_ntok and content_columns index the context).
    from ltx_core.text_encoders.gemma.gemma_assets import (
        GemmaAssets,
        build_gemma_hf_tokenizer,
    )
    gemma_tok = build_gemma_hf_tokenizer(GemmaAssets.load(gemma_path))

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

    # ------------------------------------------------------------------ 3. RENDER + CAPTURE
    c_scene, c_s0, c_s1 = cached[0], cached[1], cached[2]
    stem_prompts = [args.stem0_prompt, args.stem1_prompt]
    stem_ntok = [len(gemma_tok(p, add_special_tokens=True)["input_ids"]) for p in stem_prompts]
    stem_content = [content_columns(p, n, gemma_tok) for p, n in zip(stem_prompts, stem_ntok)]

    # Block-diagonal caption routing: [stem0 | stem1 | audio-mix] each read their own text.
    span_ctxs = (c_s0, c_s1, c_scene)
    JointStemSampler.joint_audio_context = torch.cat(
        [c.audio_context_positive for c in span_ctxs], dim=1)
    JointStemSampler.joint_block_masks = [
        torch.ones(1, c.audio_context_positive.shape[1]) for c in span_ctxs]
    if use_nag:
        # Cross-Stem Attention Guidance: the negative branch routes each stem to its sibling's
        # caption (stem0 <- stem1 text, stem1 <- stem0 text); the mix keeps its own.
        _s0, _s1 = c_s0.audio_context_positive, c_s1.audio_context_positive
        _sc = c_scene.audio_context_positive
        _masks = JointStemSampler.joint_block_masks
        nag.set_nag(args.nag_scale, args.nag_tau, args.nag_alpha,
                    null_ctx=torch.cat([_s1, _s0, _sc], dim=1),
                    null_bias=build_block_diagonal_text_bias(
                        span_lens=[span_len] * 3,
                        block_masks=[_masks[1].to(dev), _masks[0].to(dev), _masks[2].to(dev)],
                        dtype=_s0.dtype, device=dev))

    # Observed video on the audio's noise schedule; observed audio-mix clean at every step.
    JointStemSampler.joint_observed_video = strategy._video_patchifier.patchify(  # noqa: SLF001
        vlat.to(torch.bfloat16))
    JointStemSampler.joint_observed_video_schedule = True
    mix_tokens = strategy._audio_patchifier.patchify(mix_lat.to(torch.float32))  # noqa: SLF001
    JointStemSampler.joint_x0_guidance = None
    request = JointStemGenerationRequest(
        ref_time_offset=strategy_cfg.ref_time_offset, mix_lead_alpha=1.0,
        teacher_force_mix_latent=mix_tokens, teacher_force_seed=args.seed,
        mix_sigma_track=torch.zeros(n_steps + 1),
        span_pe_layout=strategy_cfg.span_pe_layout)
    sampler.audio_conditionings = [request]
    gen_config = GenerationConfig(
        prompt=args.scene_prompt, negative_prompt=vcfg.negative_prompt,
        height=height, width=width, num_frames=num_frames, frame_rate=vcfg.frame_rate,
        num_inference_steps=n_steps, guidance_scale=vcfg.video_cfg_scale,
        seed=args.seed,
        condition_image=TF.to_tensor(open_image_as_srgb(str(ff))),
        generate_audio=True, cached_embeddings=c_scene,
        stg_scale=vcfg.video_stg_scale, stg_blocks=vcfg.stg_blocks)

    Cap.reset()
    Cap.stem_ctx = [c_s0.video_context_positive, c_s1.video_context_positive]
    Cap.scene_ctx = c_scene.video_context_positive
    Cap.stem_valid = [torch.arange(2, n) for n in stem_ntok]
    Cap.stem_content_valid = stem_content
    Cap.ctx_dtype = c_scene.video_context_positive.dtype
    Cap.enabled = True
    video_out, primary = sampler.generate(config=gen_config, device=dev)
    Cap.enabled = False
    if len(Cap.sigmas) != n_steps:
        raise RuntimeError(f"captured {len(Cap.sigmas)} pos passes, expected {n_steps}")

    slots = sampler.last_slot_audio
    for k in range(N_STEMS):
        wav_k = slots[f"generated_stem{k}"].detach().cpu()
        arr = wav_k.T.numpy() if wav_k.ndim > 1 else wav_k.numpy()
        sf.write(out / f"stem_{k}.wav", arr, sr_out)
        rms = float(np.sqrt(np.mean(arr ** 2)))
        print(f"[localize]   stem_{k}.wav ({20 * np.log10(max(rms, 1e-9)):+.1f} dBFS)",
              flush=True)
    save_video(video_tensor=video_out, output_path=out / "repro.mp4",
               fps=vcfg.frame_rate, audio=primary, audio_sample_rate=sr_out)

    # ------------------------------------------------------------------ 4. READOUT
    t2v_arr = csa.stack_blocks(Cap.steps_t2v_out_content, n_blocks,
                               (N_STEMS, n_video_tokens)).astype(np.float32)
    t2v_arr = t2v_arr.reshape(len(Cap.sigmas), n_blocks, N_STEMS, *grid)
    v2a_arr = csa.stack_blocks(Cap.steps_v2a_maps, n_blocks,
                               (3, n_video_tokens)).astype(np.float32)
    v2a_arr = v2a_arr.reshape(len(Cap.sigmas), n_blocks, 3, *grid)
    chans = build_maps(t2v_arr, v2a_arr)

    track, picked = {}, {}
    xx_scale = width / grid[2]
    yy_scale = height / grid[1]
    for k in range(N_STEMS):
        susp, raw_sig = {}, {}
        for ch, m in chans.items():
            susp[ch], raw_sig[ch] = suspicion(m, k, stats)
        pick = min(susp, key=susp.get)
        picked[k] = pick
        z, R = measurements(chans[pick][k])
        lls = [kalman_forward(z, R, 1.0, q)[4] for q in Q_GRID]
        q_ml = float(Q_GRID[int(np.argmax(lls))])
        ms, Ps, mp, Pp, _ = kalman_forward(z, R, 1.0, q_ml)
        sm, _ = rts_backward(ms, Ps, mp, Pp, 1.0, q_ml)
        track[f"stem{k}"] = {
            "channel": pick, "suspicion": susp, "raw_signals": raw_sig,
            "flagged": bool(susp[pick] > flag_z), "q_ml": q_ml,
            "cells_yx": [[round(float(y), 3), round(float(x), 3)] for y, x in sm[:, :2]],
            "pixels_xy": [[round(float(x * xx_scale + xx_scale / 2), 1),
                           round(float(y * yy_scale + yy_scale / 2), 1)]
                          for y, x in sm[:, :2]],
            "raw_cells_yx": [[round(float(y), 3), round(float(x), 3)] for y, x in z]}
        print(f"[localize]   stem{k}: channel {pick} (suspicion t2v {susp['t2v']:+.2f} "
              f"v2a {susp['v2a']:+.2f})"
              f"{'  << LOW CONFIDENCE' if track[f'stem{k}']['flagged'] else ''}", flush=True)

    (out / "track.json").write_text(json.dumps(
        {"video": str(args.video),
         "prompts": {"scene": args.scene_prompt, "stem0": args.stem0_prompt,
                     "stem1": args.stem1_prompt},
         "grid_FHW": list(grid), "frame_wh": [width, height],
         "gt_metrics": None, **track}, indent=1))
    np.savez_compressed(out / "maps.npz", t2v=chans["t2v"].astype(np.float16),
                        v2a=chans["v2a"].astype(np.float16),
                        sigmas=np.array(Cap.sigmas, dtype=np.float32))

    # Overlay: chosen-channel heat + smoothed track on five frames of the clip.
    show = tuple(int(round(f)) for f in np.linspace(0, grid[0] - 1, 5))
    overlay_frames = letterbox(read_video(str(args.video))[0][:num_frames], (width, height))
    fig, axes = plt.subplots(N_STEMS, len(show), figsize=(3.2 * len(show), 4.6))
    for col, lf in enumerate(show):
        frame = overlay_frames[min(VAE_TEMPORAL_STRIDE * lf,
                                   num_frames - 1)].permute(1, 2, 0).numpy()
        for k in range(N_STEMS):
            ax = axes[k, col]
            ax.imshow(frame, extent=[0, grid[2], grid[1], 0])
            m = chans[picked[k]][k, lf]
            heat = m / (m.max() + 1e-12)
            heat = np.where(heat >= 0.3, heat, 0.0)
            ax.imshow(heat, cmap="magma", alpha=0.3 * (heat > 0),
                      extent=[0, grid[2], grid[1], 0], interpolation="nearest")
            pts = np.array(track[f"stem{k}"]["cells_yx"])
            ax.plot(pts[:, 1] + 0.5, pts[:, 0] + 0.5, "-", color="white", lw=2.6, alpha=0.9)
            ax.plot(pts[:, 1] + 0.5, pts[:, 0] + 0.5, "-", color="#ff6d00", lw=1.4)
            ax.plot(pts[lf, 1] + 0.5, pts[lf, 0] + 0.5, "X", ms=14, color="#ff6d00",
                    markeredgecolor="white", markeredgewidth=1.6)
            ax.set_xlim(0, grid[2]), ax.set_ylim(grid[1], 0)
            ax.set_xticks([]), ax.set_yticks([])
            if col == 0:
                flag = " (LOW CONF)" if track[f"stem{k}"]["flagged"] else ""
                ax.set_ylabel(f"stem{k} [{picked[k]}]{flag}", fontsize=9)
            if k == 0:
                ax.set_title(f"latent f{lf}", fontsize=9)
    fig.suptitle(f"{args.video.stem} — localizer tracks", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "overlay.png", dpi=110)
    plt.close(fig)
    print(f"[localize] wrote {out}/", flush=True)


if __name__ == "__main__":
    main()
