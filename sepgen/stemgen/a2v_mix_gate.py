"""Arm N complement: VIDEO listens to the MIX span only (key mask on ``audio_to_video_attn``).

WHY. The DiT block's ``audio_to_video_attn`` (transformer.py:352-360) lets VIDEO tokens read the
whole audio sequence ``[stem0 | stem1 | mix]``. Under arm N (frozen-prior mix) that is the one
remaining second-order leak into the mix: stems (LoRA-influenced activations) -> video features
-> mix's ``video_to_audio_attn`` read. Restricting video's audio-listening to the MIX span closes
the loop -- the video now syncs to the scene's actual audio (the mix, computed by pure base
weights), never to the internal decomposition channels -- and the mix's ENTIRE input set becomes
base-computed: its own span (arm-K mask), base-encoded text (K/V not adapted), and video that
heard only base-computed audio. User decision 2026-07-28: "video should also listen to the mix."

MECHANISM. Same wrapper pattern as ``audio_span_gate`` / ``text_block_gate``: wrap every
``*audio_to_video_attn`` forward; when the call site passes ``mask=None`` (every joint-path call
site today) and the gate is armed, inject an additive pre-softmax key bias ``[1, 1, 1, T_audio]``
that is 0.0 on the trailing mix span and ``finfo.min`` on the stem spans. The bias depends only
on the KEY index, so it broadcasts over batch, heads and video-query tokens, and it applies
identically on the positive, CFG-negative and STG-perturbed passes (property of the sequence
layout, not the prompt -- no length routing needed: the audio key axis is identical on all
passes). ``LTX_MASKED_ATTENTION=sdpa`` required, already set by every joint entrypoint.

Failure modes handled loud: audio key count not divisible by the span count (layout assumption
broken), no module found, double-install (idempotent skip).
"""
from __future__ import annotations

import torch
from torch import Tensor

_STATE: dict = {"num_spans": None, "enabled": False}


def set_a2v_mix_gate(num_spans: int) -> None:
    """Arm the gate: the audio key axis divides into ``num_spans`` equal spans; only the LAST
    (the mix) stays visible to video queries."""
    if num_spans < 2:
        raise ValueError(f"num_spans must be >= 2 (stems + mix), got {num_spans}")
    _STATE.update(num_spans=num_spans, enabled=True)


def clear_a2v_mix_gate() -> None:
    _STATE.update(enabled=False)


def _key_bias(t_keys: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    n = _STATE["num_spans"]
    if t_keys % n != 0:
        raise RuntimeError(
            f"a2v_mix_gate: audio key axis {t_keys} not divisible by {n} spans -- refusing to "
            "mask the wrong tokens")
    span = t_keys // n
    bias = torch.full((1, 1, 1, t_keys), torch.finfo(dtype).min, dtype=dtype, device=device)
    bias[..., (n - 1) * span:] = 0.0
    return bias


def install_a2v_mix_gate(model: torch.nn.Module) -> list[str]:
    """Wrap every ``*audio_to_video_attn`` forward to inject the mix-only key bias."""
    wrapped: list[str] = []
    for name, module in model.named_modules():
        if not name.endswith("audio_to_video_attn"):
            continue
        if getattr(module, "_a2v_mix_gated", False):
            continue
        orig_forward = module.forward

        def make(orig):
            def forward(x, context=None, mask=None, **kwargs):
                if mask is None and _STATE["enabled"] and context is not None:
                    mask = _key_bias(context.shape[1], x.dtype, x.device)
                return orig(x, context=context, mask=mask, **kwargs)
            return forward

        module.forward = make(orig_forward)
        module._a2v_mix_gated = True                              # noqa: SLF001
        module._a2v_mix_orig_forward = orig_forward               # noqa: SLF001
        wrapped.append(name)
    if not wrapped:
        raise RuntimeError(
            "install_a2v_mix_gate found no 'audio_to_video_attn' modules to gate")
    return wrapped
