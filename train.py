#!/usr/bin/env python
"""Train SepGen: a LoRA on the audio stream of LTX-2.5 22B dev that adds two stem spans next to
the audio-mix span.

    python train.py configs/train_sep12k.yaml                       # separation checkpoint
    python train.py configs/train_gen3k.yaml --stop-at-step 3000    # generation checkpoint

The joint-stem strategy (stemgen/joint_stems.py) is injected into the stock LTX trainer through
its training-strategy extension point, so LoRA setup, int8 quantization, gradient
checkpointing, checkpointing and resume are upstream's. On top of it this script installs the
attention gates the method relies on, identically to sampling:
  - block-diagonal caption routing on the audio text cross-attention (each span reads its caption);
  - the span mask on audio self-attention (the audio-mix attends only to itself; stems attend to
    the audio-mix and to each other);
  - a zero LoRA delta on audio-mix tokens (the protected audio-mix);
  - video reads audio from the audio-mix span only;
  - the positional source-box bias on video-to-audio attention for positional captions.

The data root holds precomputed latents and text features (docs/DATA.md). Validation renders are
not run during training (validation.interval: null); evaluate checkpoints with separate.py.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("LTX_MASKED_ATTENTION", "sdpa")

import yaml  # noqa: E402

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "sepgen"))


def main():
    ap = argparse.ArgumentParser(description="SepGen training")
    ap.add_argument("config")
    ap.add_argument("--models-dir", type=Path, default=Path(os.environ.get(
        "SEPGEN_MODELS", REPO / "models/ltx2.5")), help="LTX-2.5 weights (download_weights.sh)")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="override data.preprocessed_data_root")
    ap.add_argument("--init-checkpoint", default=None,
                    help="warm-start LoRA (sep-12k from Hugging Face, or a path); overrides "
                         "model.load_checkpoint")
    ap.add_argument("--output-dir", type=Path, default=None, help="override output_dir")
    ap.add_argument("--stop-at-step", type=int, default=None,
                    help="stop after saving this step, keeping the LR schedule of "
                         "optimization.steps (gen-3k = step 3000 of a 12,000-step schedule)")
    args = ap.parse_args()

    config_path = Path(args.config)
    os.environ["SEPGEN_MODELS"] = str(args.models_dir.resolve())
    raw = yaml.safe_load(os.path.expandvars(config_path.read_text()))
    if args.data_root is not None:
        raw["data"]["preprocessed_data_root"] = str(args.data_root)
    if args.output_dir is not None:
        raw["output_dir"] = str(args.output_dir)
    if args.init_checkpoint is not None:
        if args.init_checkpoint in ("sep-12k", "gen-3k"):
            from huggingface_hub import hf_hub_download
            raw["model"]["load_checkpoint"] = hf_hub_download(
                "AviadDahan/SepGen", f"{args.init_checkpoint}/lora_weights.safetensors")
        else:
            raw["model"]["load_checkpoint"] = args.init_checkpoint
    strategy_raw = raw.pop("joint_stem", {})
    raw.pop("joint_stem_validation", None)

    from stemgen.joint_stems import JointStemConfig, JointStemStrategy

    strategy_cfg = JointStemConfig(**strategy_raw)
    strategy = JointStemStrategy(strategy_cfg)
    num_stems = strategy_cfg.num_stems
    print(f"[sepgen] num_stems={num_stems} include_mix={strategy_cfg.include_mix} "
          f"mix_sigma_mode={strategy_cfg.mix_sigma_mode} sep_pattern_p={strategy_cfg.sep_pattern_p} "
          f"topology={strategy_cfg.span_attention_topology}", flush=True)

    # The trainer's validation runner caches these captions' embeddings at construction; only the
    # negative-prompt length is used here (it lets the caption-routing gate tell the CFG passes
    # apart). No validation render runs.
    raw.setdefault("validation", {})
    raw["validation"]["prompts"] = ["Two sound sources in one scene."]
    raw["validation"]["interval"] = None

    import ltx_trainer.trainer as trainer_mod
    from ltx_trainer.config import LtxTrainerConfig

    trainer_mod.get_training_strategy = lambda _cfg: strategy
    cfg = LtxTrainerConfig(**raw)
    # LTX-2 v1.2.0 builds the dataloader from the strategy CONFIG's data sources; swap in the
    # joint-stem config after parsing so the dataloader reads every source this strategy needs.
    cfg.__dict__["training_strategy"] = strategy_cfg
    sources = strategy_cfg.get_data_sources()
    print(f"[sepgen] dataloader sources ({len(sources)}): {sorted(sources)}", flush=True)

    trainer = trainer_mod.LtxvTrainer(cfg)
    if trainer._training_strategy is not strategy:               # noqa: SLF001
        raise RuntimeError("strategy injection failed -- trainer is not using the joint strategy")

    # ---- per-stem and scene captions through the embeddings connector ---------------------------
    # The stems and the audio-mix each read their own caption. Those features are stored in the
    # same pre-connector space as `conditions`, so they take the same connector pass before the
    # original step overwrites `conditions` in place.
    from ltx_core.text_encoders.gemma.embeddings_processor import convert_to_additive_mask

    _orig_training_step = trainer._training_step                      # noqa: SLF001

    def _training_step_with_extra_conditions(batch):
        conditions = batch["conditions"]
        video_features = conditions.get("video_prompt_embeds", conditions.get("prompt_embeds"))
        mask = conditions["prompt_attention_mask"]
        for source_name, source in batch.items():
            if not source_name.startswith("conditions_") or not isinstance(source, dict):
                continue
            extra_audio = source.get("audio_prompt_embeds")
            if extra_audio is None:
                continue
            extra_mask = source.get("audio_prompt_attention_mask", mask)
            extra_additive = convert_to_additive_mask(extra_mask, video_features.dtype)
            _v, extra_audio_embeds, extra_attention_mask = \
                trainer._embeddings_processor.create_embeddings(              # noqa: SLF001
                    video_features, extra_audio, extra_additive)
            source["audio_prompt_embeds"] = extra_audio_embeds
            source["audio_prompt_attention_mask"] = extra_attention_mask
        return _orig_training_step(batch)

    trainer._training_step = _training_step_with_extra_conditions     # noqa: SLF001

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "launch_config.yaml").write_text(config_path.read_text())

    # ---- attention gates (identical at sampling time; see separate.py / stempipe) ----------------
    from stemgen.text_block_gate import install_text_block_gate

    cached = trainer._validation_runner._cached_embeddings        # noqa: SLF001
    if not cached or cached[0].audio_context_negative is None:
        raise RuntimeError("validation.negative_prompt is required (its length keys the CFG passes)")
    neg_len = int(cached[0].audio_context_negative.shape[1])
    install_text_block_gate(trainer._transformer, negative_context_len=neg_len)  # noqa: SLF001

    if strategy.span_allow_matrix is not None:
        from stemgen.audio_span_gate import install_audio_span_gate
        install_audio_span_gate(trainer._transformer)             # noqa: SLF001
    if strategy_cfg.frozen_mix_lora:
        from stemgen.lora_span_gate import install_lora_span_gate, set_lora_span_gate
        set_lora_span_gate(num_spans=num_stems + 1, gated_span=num_stems)   # mix is last
        install_lora_span_gate(trainer._transformer)              # noqa: SLF001
    if strategy_cfg.a2v_mix_only:
        from stemgen.a2v_mix_gate import install_a2v_mix_gate, set_a2v_mix_gate
        set_a2v_mix_gate(num_spans=num_stems + 1)
        install_a2v_mix_gate(trainer._transformer)                # noqa: SLF001
    if strategy_cfg.positional_spatial_mask:
        from stemgen.spatial_mask_gate import install_v2a_mask_gate
        install_v2a_mask_gate(trainer._transformer)               # noqa: SLF001
    print("[sepgen] gates installed: caption routing, span mask, protected audio-mix LoRA, "
          "video<-mix only, positional source box", flush=True)

    if args.stop_at_step is not None:
        interval = cfg.checkpoints.interval
        if not interval or args.stop_at_step % interval:
            raise SystemExit(f"--stop-at-step {args.stop_at_step} must be a multiple of "
                             f"checkpoints.interval ({interval}) so that step is saved")
        _orig_save = trainer._save_checkpoint                     # noqa: SLF001

        def _save_then_stop():
            path = _orig_save()
            if trainer._global_step >= args.stop_at_step:          # noqa: SLF001
                print(f"[sepgen] reached step {trainer._global_step}; stopping -> {path}",  # noqa: SLF001
                      flush=True)
                raise SystemExit(0)
            return path

        trainer._save_checkpoint = _save_then_stop                 # noqa: SLF001

    path, stats = trainer.train()
    print(f"[sepgen] done -> {path}\n{stats}", flush=True)


if __name__ == "__main__":
    main()
