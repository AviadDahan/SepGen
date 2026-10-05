"""SepGen: per-channel TEXT cross-attention via a block-diagonal mask.

The joint K-stem sequence is ``[stem0 | stem1 (| mix)]`` in ONE audio stream. Each span must be
driven by its OWN caption: stem0 by the first speaker's caption, stem1 by the second's, the mix
(arms B/C) by the scene caption. We concatenate the per-channel text blocks along the key axis --
``context = concat[stem0_text, stem1_text (, scene_text)]`` -- and force each audio span to attend
ONLY to its own text block with an additive block-diagonal bias on the audio text cross-attention
(``audio_attn2``), plus each block's own padding.

Mechanism (verified against ltx_core):
  * ``audio_attn2`` (Q = audio latents, K/V = ``audio.context`` Gemma text) receives the audio
    ``Modality.context_mask`` as its ``mask=`` kwarg via ``_apply_text_cross_attention`` /
    ``apply_cross_attention_adaln`` (transformer.py). ``Attention.forward`` uses ``mask`` as an
    ADDITIVE pre-softmax bias on the SDPA path; ``TransformerArgsPreprocessor._prepare_attention_mask``
    passes a FLOAT mask through untouched, so a ``[B, 1, q_audio, kv_text]`` float bias flows
    natively to ``audio_attn2``. Requires ``LTX_MASKED_ATTENTION=sdpa`` (the xformers cutlass
    kernel needs ``attn_bias.stride(-2) % 8 == 0``, which arbitrary text kv lengths violate).

TRAINING uses the native path: ``prepare_training_inputs`` sets the audio ``Modality.context_mask``
to the block bias directly (single forward, no CFG, so no negative-context length clash).

VALIDATION reuses the frozen sampler's ``_run_denoising`` (CFG + STG), which hardcodes
``context_mask=None`` and swaps ``context`` for the negative prompt on the CFG pass. There we cannot
set the mask on the Modality, so we WRAP ``audio_attn2`` (exactly like ``spatial_mask_gate`` wraps
``video_to_audio_attn``) and inject the stashed block bias -- but ONLY when the incoming context
length matches the positive concat length. The negative pass presents the single-block negative
prompt (a different length), where per-channel structure is meaningless, so the gate leaves it
unmasked (every span attends the whole negative prompt). ``install_text_block_gate`` asserts the
positive and negative lengths differ so the length test can never alias.
"""
from __future__ import annotations

import torch
from torch import Tensor


def build_block_diagonal_text_bias(
    span_lens: list[int],
    block_masks: list[Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Additive block-diagonal bias ``[B, 1, sum(span_lens), sum(block widths)]``.

    Query tokens of span ``i`` (the ``i``-th audio channel) are allowed to attend ONLY to text
    block ``i`` and, within it, only to that block's VALID (non-pad) keys; everything else gets a
    large negative bias so softmax drives it to zero.

    Args:
        span_lens: audio-token count of each channel span, in order ``[stem0, stem1 (, mix)]``.
        block_masks: per-block binary (1=valid, 0=pad) key masks, each ``[B, S_i]``, same order.
        dtype/device: of the returned bias (match the audio context tensor).
    """
    if len(span_lens) != len(block_masks):
        raise ValueError(
            f"span_lens ({len(span_lens)}) and block_masks ({len(block_masks)}) must align "
            "one-to-one -- one text block per audio channel span")
    batch = block_masks[0].shape[0]
    q_total = sum(span_lens)
    k_total = sum(int(m.shape[1]) for m in block_masks)
    neg = torch.finfo(dtype).min
    bias = torch.full((batch, 1, q_total, k_total), neg, dtype=dtype, device=device)
    q0 = 0
    k0 = 0
    for span_len, mask in zip(span_lens, block_masks):
        width = int(mask.shape[1])
        if mask.shape[0] != batch:
            raise ValueError(f"block mask batch {mask.shape[0]} != {batch}")
        valid = mask.to(device=device).bool()                       # [B, S_i]
        allowed = torch.where(
            valid[:, None, None, :], torch.zeros((), dtype=dtype, device=device),
            torch.full((), neg, dtype=dtype, device=device),
        ).expand(batch, 1, span_len, width)
        bias[:, :, q0:q0 + span_len, k0:k0 + width] = allowed
        q0 += span_len
        k0 += width
    return bias


class _BiasHolder:
    """Module-global stash for the current validation forward's block bias (or None)."""

    bias: Tensor | None = None


_HOLDER = _BiasHolder()


def set_text_block_bias(bias: Tensor | None) -> None:
    _HOLDER.bias = bias


def clear_text_block_bias() -> None:
    _HOLDER.bias = None


def current_text_block_bias() -> Tensor | None:
    return _HOLDER.bias


def install_text_block_gate(model: torch.nn.Module, negative_context_len: int) -> list[str]:
    """Wrap every ``*audio_attn2`` forward to inject the stashed block bias on the POSITIVE pass.

    The gate injects only when the incoming ``context`` length equals the stashed bias key length,
    so the CFG negative pass (single-block negative prompt of ``negative_context_len`` tokens) is
    left unmasked. Idempotent; returns wrapped module names. Fails loud if no ``audio_attn2`` exists.
    """
    wrapped: list[str] = []
    for name, module in model.named_modules():
        if not name.endswith("audio_attn2"):
            continue
        if getattr(module, "_text_block_gated", False):
            continue
        orig_forward = module.forward

        def make(orig):
            def forward(x, context=None, mask=None, **kwargs):
                if mask is None:
                    bias = _HOLDER.bias
                    if bias is not None and context is not None:
                        if context.shape[0] != bias.shape[0]:
                            raise ValueError(
                                f"text-block bias batch {bias.shape[0]} != context batch "
                                f"{context.shape[0]}")
                        if context.shape[1] == bias.shape[-1]:
                            mask = bias.to(dtype=context.dtype, device=context.device)
                        elif context.shape[1] != negative_context_len:
                            raise ValueError(
                                f"audio_attn2 context length {context.shape[1]} matches neither the "
                                f"positive concat length {bias.shape[-1]} nor the negative length "
                                f"{negative_context_len}; the block gate cannot place the mask")
                return orig(x, context=context, mask=mask, **kwargs)
            return forward

        module.forward = make(orig_forward)
        module._text_block_gated = True                              # noqa: SLF001
        module._text_block_orig_forward = orig_forward               # noqa: SLF001
        wrapped.append(name)
    if not wrapped:
        raise RuntimeError(
            "install_text_block_gate found no 'audio_attn2' modules -- the transformer has no "
            "audio text cross-attention to gate")
    return wrapped


def uninstall_text_block_gate(model: torch.nn.Module) -> None:
    for _name, module in model.named_modules():
        if getattr(module, "_text_block_gated", False):
            module.forward = module._text_block_orig_forward         # noqa: SLF001
            del module._text_block_gated
            del module._text_block_orig_forward
    clear_text_block_bias()
