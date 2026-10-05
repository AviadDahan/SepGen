"""Assemble the stage-1 transformer for the stem method.

base (bf16) + distilled LoRA **fused** at build time
             + our separation adapter **attached, not fused** (the span gate needs the
               delta to remain separable so it can be switched off on the mix span)
             + the five attention gates
             -> X0Model

Deliberately no quantization policy: an int8 base was root-caused as the source of soft,
stippled video (the audio path hides it because the adapter was trained on top of the
quantized base; the video path carries no adapter and eats the loss).
"""
from __future__ import annotations

import logging

import torch
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import ModelRegistry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from ltx_core.model.transformer.model import X0Model
from ltx_core.model.transformer.model_configurator import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
)
from safetensors.torch import load_file

from .capture import install_capture
from .gates import install_gates
from .stem_config import StemConfig

logger = logging.getLogger(__name__)

_PREFIX = "diffusion_model."
_A_SUFFIX = ".lora_A.weight"
_B_SUFFIX = ".lora_B.weight"


def read_adapter(path: str) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], int]:
    """-> {module_path: (A, B)}, rank.  Module paths are relative to the transformer root."""
    raw = load_file(path)
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    ranks = set()
    for key, tensor in raw.items():
        stripped = key[len(_PREFIX):] if key.startswith(_PREFIX) else key
        if stripped.endswith(_A_SUFFIX):
            module = stripped[: -len(_A_SUFFIX)]
            b = raw.get(f"{_PREFIX}{module}{_B_SUFFIX}", raw.get(f"{module}{_B_SUFFIX}"))
            if b is None:
                raise ValueError(f"adapter key {key} has no matching lora_B")
            pairs[module] = (tensor, b)
            ranks.add(tensor.shape[0])
        elif not stripped.endswith(_B_SUFFIX):
            raise ValueError(f"unexpected adapter key {key!r}")
    if len(ranks) != 1:
        raise ValueError(f"adapter mixes ranks {sorted(ranks)}; expected exactly one")
    return pairs, ranks.pop()


def _target_suffixes(module_paths: list[str]) -> list[str]:
    """The distinct projection names the adapter touches, e.g. 'audio_attn1.to_k'.

    Derived from the checkpoint itself so no training config is consulted.
    """
    seen = []
    for path in module_paths:
        parts = path.split(".")
        # strip the leading 'transformer_blocks.<i>.'
        if len(parts) >= 3 and parts[0] == "transformer_blocks":
            suffix = ".".join(parts[2:])
        else:
            suffix = parts[-1]
        if suffix not in seen:
            seen.append(suffix)
    return seen


def build_stem_transformer(*, model_paths, distilled_lora, distilled_lora_strength: float,
                           adapter_checkpoint: str, adapter_rank: int, adapter_alpha: int,
                           stem: StemConfig, dtype: torch.dtype = torch.bfloat16,
                           device: torch.device | None = None,
                           span_len: int | None = None):
    """-> (X0Model with gates installed, the builder it came from)."""
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import LoraLayer

    device = device or torch.device("cuda")
    pairs, ckpt_rank = read_adapter(adapter_checkpoint)
    if ckpt_rank != adapter_rank:
        logger.warning("adapter rank in checkpoint is %d; overriding --adapter-rank %d",
                       ckpt_rank, adapter_rank)
        adapter_rank = ckpt_rank
    targets = _target_suffixes(list(pairs))
    logger.info("adapter: %d modules, rank %d, targets %s", len(pairs), adapter_rank, targets)

    loras = ()
    if distilled_lora and distilled_lora_strength:
        loras = (LoraPathStrengthAndSDOps(path=distilled_lora[0].path,
                                          strength=distilled_lora_strength,
                                          sd_ops=distilled_lora[0].sd_ops),)
    builder = SingleGPUModelBuilder(
        model_class_configurator=LTXModelConfigurator,
        model_path=model_paths.transformer(),
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        loras=loras,
        registry=ModelRegistry(cache_models=False, cache_weights=False),
    )
    logger.info("building base transformer (bf16, distilled LoRA fused @ %s)",
                distilled_lora_strength)
    base = builder.build(device=device, dtype=dtype)

    # A registry that hands back an already-adapted shell would silently nest adapters.
    if any(isinstance(m, LoraLayer) for m in base.modules()):
        raise RuntimeError("base model already carries LoRA layers before we attached ours")

    peft_model = get_peft_model(
        base,
        LoraConfig(r=adapter_rank, lora_alpha=adapter_alpha, lora_dropout=0.0,
                   target_modules=targets, init_lora_weights=True),
    )
    _load_adapter_weights(peft_model, pairs, dtype=dtype, device=device)

    model = X0Model(peft_model).eval()
    inner = model.velocity_model if hasattr(model, "velocity_model") else model
    handles = install_gates(
        inner,
        num_spans=stem.num_spans,
        span_len=span_len or 0,
        a2v_mode=stem.a2v_mode,
    )
    model.stem_gates = handles
    if stem.capture is not None and stem.capture.enabled:
        # A separate seam from the gates (`attention_function`, not `forward`), installed
        # once here; the per-cell handles is swapped onto `handles.capture` by the pipeline.
        install_capture(inner, handles)
    logger.info("gates installed: %s", handles.counts)
    # Park on host RAM until stage 1 asks for it. Upstream never has a transformer resident
    # while the text encoder runs; ours would, and 42 GB + Gemma does not fit on one card.
    # PreparedDiffusionStage._transformer_ctx moves it back for the denoising loop.
    model.to("cpu")
    torch.cuda.empty_cache()
    return model, builder


def _load_adapter_weights(peft_model, pairs, *, dtype, device) -> None:
    """Assign A/B by module path, and refuse anything less than a complete, exact load."""
    from peft.tuners.lora import LoraLayer

    filled, unmatched = set(), []
    by_suffix = {}
    for name, module in peft_model.named_modules():
        if isinstance(module, LoraLayer):
            # peft names look like base_model.model.transformer_blocks.0.audio_attn1.to_k
            key = name.split("base_model.model.", 1)[-1]
            by_suffix[key] = module

    for path, (a, b) in pairs.items():
        module = by_suffix.get(path)
        if module is None:
            unmatched.append(path)
            continue
        with torch.no_grad():
            module.lora_A["default"].weight.copy_(a.to(dtype=dtype, device=device))
            module.lora_B["default"].weight.copy_(b.to(dtype=dtype, device=device))
        filled.add(path)

    if unmatched:
        raise RuntimeError(
            f"{len(unmatched)} adapter modules had no counterpart in the model, e.g. "
            f"{unmatched[:3]} -- the checkpoint and the checkpoint architecture disagree")
    missing = set(by_suffix) - filled
    if missing:
        raise RuntimeError(
            f"{len(missing)} adapted modules got no weights, e.g. {sorted(missing)[:3]} -- "
            "they would contribute a randomly-initialised delta")
    logger.info("adapter loaded into %d modules (complete, exact)", len(filled))
