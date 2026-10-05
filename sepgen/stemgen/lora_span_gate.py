"""Arm N: per-token LoRA gating -- the MIX span's computation stays the FROZEN base model.

MECHANISM. Ear evidence (2026-07-28): even with the mix span supervised (arm R1), its render
quality at step 500 is "way worse than checkpoint 0" -- LoRA drift on the shared audio pathway
degrades base fidelity faster than a velocity loss can repair it, and the only guarantee of
checkpoint-0 quality is checkpoint-0 COMPUTATION. This module zeroes the LoRA delta on the mix
span's token positions in every audio-side adapted module whose token axis is the audio sequence
``[stem0 | stem1 | mix]``, so the mix span's forward pass is bit-identical to the frozen base at
every training step, forever -- one network, one unified generation, per-token routing.

WHAT IS GATEABLE AND WHAT IS NOT. A LoRA delta can be zeroed per token only where the module's
input token axis IS the audio sequence: ``audio_attn1`` q/k/v/out (self-attention over the audio
tokens), ``audio_ff`` (per-token MLP), and the QUERY/OUT projections of ``audio_attn2`` /
``video_to_audio_attn``. The K/V projections of those two cross-attentions run over TEXT / VIDEO
tokens shared by all spans and CANNOT be span-gated -- an arm-N config must simply drop them from
``lora.target_modules`` (the install fails loud if it finds an adapted, ungated K/V there).

CONTRACT WITH THE REST OF THE STACK. The gate is a property of the SEQUENCE LAYOUT, exactly like
the arm-K span mask: it applies identically on the training forward and on every sampling pass
(positive, CFG-negative, STG-perturbed). Combine with ``span_attention_topology: mix_protected``
so the mix span also SEES only base-model inputs (itself + video + its own text); without the
mask the frozen weights would still read LoRA-influenced stem activations. Gradients: the delta
is exactly zero on mix tokens, so stem-loss gradients through the mix span's own projections
vanish -- the "stems reshape the mix's representation" channel (arm-K post-mortem, mechanism 2)
is closed at the weight level.

Assumes the repo's LoRA config: ONE adapter ("default"), dropout 0.0 (Identity) -- asserted at
install, because the wrapper re-derives peft's Linear forward (base + B(A(x)) * scaling) and any
deviation must fail loud, not drift silently.
"""
from __future__ import annotations

import re

import torch
from torch import Tensor

# Module-name suffixes whose input token axis is the audio sequence [stem0|stem1|mix].
GATEABLE_SUFFIXES = (
    "audio_attn1.to_q", "audio_attn1.to_k", "audio_attn1.to_v", "audio_attn1.to_out.0",
    "audio_attn2.to_q", "audio_attn2.to_out.0",
    "video_to_audio_attn.to_q", "video_to_audio_attn.to_out.0",
    "audio_ff.net.0.proj", "audio_ff.net.2",
)
# Adapted-but-ungateable projections an arm-N config must NOT contain.
FORBIDDEN_SUFFIXES = (
    "audio_attn2.to_k", "audio_attn2.to_v",
    "video_to_audio_attn.to_k", "video_to_audio_attn.to_v",
)

_STATE: dict = {"num_spans": None, "gated_span": None, "enabled": False}


def set_lora_span_gate(num_spans: int, gated_span) -> None:
    """Arm the gate: token axis divides into ``num_spans`` equal spans; span ``gated_span``
    (sequence order, e.g. 2 = the trailing mix) gets ZERO LoRA delta.

    ``gated_span="all"`` zeroes the delta on EVERY token — the wrapped modules then run
    the pure frozen base computation (phase-12 autoguidance null branch, 2026-08-07)."""
    if gated_span != "all" and gated_span >= num_spans:
        raise ValueError(f"gated_span {gated_span} out of range for {num_spans} spans")
    _STATE.update(num_spans=num_spans, gated_span=gated_span, enabled=True)


def clear_lora_span_gate() -> None:
    _STATE.update(enabled=False)


class _GatedLoraForward:
    """Replaces one peft lora.Linear's forward: base + delta, delta zeroed on the gated span."""

    def __init__(self, module):
        self.module = module

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        m = self.module
        base = m.base_layer(x, *args, **kwargs)
        if _STATE["enabled"] and _STATE["gated_span"] == "all":
            return base  # pure frozen base computation on every token
        adapter = m.active_adapters[0]
        delta = m.lora_B[adapter](m.lora_A[adapter](x)) * m.scaling[adapter]
        if _STATE["enabled"] and x.dim() == 3:
            t = x.shape[1]
            n = _STATE["num_spans"]
            if t % n == 0:
                span_len = t // n
                k = _STATE["gated_span"]
                gate = torch.ones(1, t, 1, dtype=delta.dtype, device=delta.device)
                gate[:, k * span_len:(k + 1) * span_len] = 0.0
                delta = delta * gate
            else:
                raise RuntimeError(
                    f"lora_span_gate: token axis {t} not divisible by {n} spans on a gated "
                    "module -- the gate would silently misalign; layout assumption broken")
        return base + delta


def install_lora_span_gate(model) -> list[str]:
    """Wrap every gateable adapted module; fail loud on adapted-but-ungateable ones."""
    from peft.tuners.lora.layer import LoraLayer

    wrapped: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        if any(name.endswith(s) for s in FORBIDDEN_SUFFIXES):
            raise RuntimeError(
                f"arm N: {name} is LoRA-adapted but its token axis (text/video keys) cannot be "
                "span-gated -- remove it from lora.target_modules for a frozen-prior-mix run")
        if not any(name.endswith(s) for s in GATEABLE_SUFFIXES):
            continue                                       # video-side or unrelated adapters
        if len(module.active_adapters) != 1:
            raise RuntimeError(f"{name}: expected exactly one active adapter, "
                               f"got {module.active_adapters}")
        adapter = module.active_adapters[0]
        drop = module.lora_dropout[adapter]
        if not isinstance(drop, torch.nn.Identity):
            raise RuntimeError(
                f"{name}: lora_dropout is {type(drop).__name__}, not Identity -- the gated "
                "forward re-derives peft's math for dropout=0.0 only")
        module.forward = _GatedLoraForward(module)
        wrapped.append(name)
    if not wrapped:
        raise RuntimeError("lora_span_gate: no gateable adapted modules found")
    # One block index sanity print (48 blocks x 10 projections expected for the full target set).
    blocks = {int(m.group(1)) for n in wrapped
              for m in [re.search(r"\.(\d+)\.(?:audio_|video_to_audio_)", n)] if m}
    print(f"[lora-span-gate] {len(wrapped)} adapted modules gated across "
          f"{len(blocks)} blocks; span {_STATE.get('gated_span')} of "
          f"{_STATE.get('num_spans')} runs the FROZEN base computation", flush=True)
    return wrapped
