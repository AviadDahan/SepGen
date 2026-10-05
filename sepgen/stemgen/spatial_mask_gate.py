"""SepGen: spatial-mask video-key gate for `video_to_audio_attn`.

The plan's preferred mechanism: a soft performer mask over VIDEO tokens becomes an additive
log-bias on the video keys that AUDIO queries see through `video_to_audio_attn`:

    v2a_logits[:, :, :, key_j] += log(clamp(m_k[key_j], eps, 1))     # key_j = a video token

so a token outside performer-k's support (m->0) is driven toward log(eps) RELATIVE to supported
tokens (m=1, unbiased). IMPORTANT — the bias is added pre-softmax, and softmax is shift-invariant:
softmax(z + c*1) = softmax(z). So only the RELATIVE gap between in-mask and out-of-mask keys
matters; a mask that is CONSTANT across all video keys (all-0 "full_frame" OR all-log(eps) "empty")
produces IDENTICAL attention -- "empty" is NOT a no-video control, it equals "full_frame". The gate
REDISTRIBUTES attention mass toward the masked performer (rows still sum to 1); it cannot reduce the
total video->audio contribution. A true video-mute control needs OUTPUT gating (multiply the v2a
output by 0), not a uniform additive bias. This gates ONLY `video_to_audio_attn` (audio queries, video
context); it never touches `audio_to_video_attn` (which updates video from audio and is outside the
LoRA targets, video_loss_weight=0). It is NOT LTX's `ConditioningItemAttentionStrengthWrapper` /
video self-attention mask — that path is for appended video IC-LoRA reference tokens, which
`fixed_reference` does not use.

Mechanism: the LTX `Attention.forward` already accepts a `mask` kwarg interpreted as an ADDITIVE
bias on the SDPA/xformers masked path, but the v2a call site (frozen `transformer.py`) passes none.
We therefore wrap the bound `forward` of every `video_to_audio_attn` module so it injects
`mask=<current bias>` when a bias is stashed for the active forward. The bias is per (batch, video
key), broadcast over heads and audio-query positions: shape [B, 1, 1, S_video].

Set the bias just before a forward (training: in prepare_training_inputs; validation: in the
sampler before `_run_denoising`) and it is consumed by every wrapped v2a module in that forward.
Single active forward at a time (one GPU, sequential steps), so a module-global holder is safe.

Use `spatial_mask_to_key_bias` to turn a soft mask [B, F, H, W] into the bias, flattening in the
(f h w) order that `VideoLatentPatchifier.patchify` produces (so bias index == video key index).
"""
from __future__ import annotations

import torch
from torch import Tensor


class _BiasHolder:
    """Module-global stash for the current forward's video-key bias (or None = ungated)."""

    bias: Tensor | None = None


_HOLDER = _BiasHolder()


def set_video_key_bias(bias: Tensor | None) -> None:
    _HOLDER.bias = bias


def clear_video_key_bias() -> None:
    _HOLDER.bias = None


def current_video_key_bias() -> Tensor | None:
    return _HOLDER.bias


def spatial_mask_to_key_bias(mask_fhw: Tensor, eps: float = 1e-4) -> Tensor:
    """[B, F, H, W] soft mask in [0,1] -> additive log-bias [B, 1, 1, F*H*W] over video keys.

    Flatten order is (f h w) row-major, matching VideoLatentPatchifier.patchify
    ("b c (f p1)(h p2)(w p3) -> b (f h w) ..."), so bias column j gates video key j one-for-one.
    """
    if mask_fhw.dim() != 4:
        raise ValueError(f"expected [B,F,H,W], got {tuple(mask_fhw.shape)}")
    b = mask_fhw.shape[0]
    flat = mask_fhw.reshape(b, -1)                       # (f h w) row-major
    bias = torch.log(flat.clamp(min=eps, max=1.0))
    return bias.view(b, 1, 1, -1)


def spatial_masks_to_per_query_bias(
    masks_fhw: list[Tensor | None],
    span_lens: list[int],
    eps: float = 1e-4,
) -> Tensor:
    """PER-SPAN masks -> per-audio-query additive log-bias [B, 1, T_audio_total, S_video].

    Joint separation runs ONE audio pass over ``[stem0 | stem1 | mix]`` where each stem span must
    attend a DIFFERENT performer's video support. A single [B,1,1,S] key bias cannot express that
    (it gates every audio query identically), so this builds a per-query-row bias: audio rows of
    span i receive ``log(clamp(masks_fhw[i], eps, 1))`` over the video keys; a ``None`` entry
    (the mix span) receives 0 (ungated -- the mix legitimately attends the whole frame).

    Args:
        masks_fhw: one entry per audio span, index-aligned with ``span_lens``. Each entry is a
            soft mask [B, F, H, W] in [0,1] (flattened (f h w) row-major to match
            VideoLatentPatchifier.patchify) or None for an ungated span.
        span_lens: audio token count of each span, in sequence order.
        eps: floor inside log(clamp(m, eps, 1)).
    """
    if len(masks_fhw) != len(span_lens):
        raise ValueError(
            f"masks_fhw has {len(masks_fhw)} entries but span_lens has {len(span_lens)}; one "
            "mask (or None) is required per audio span")
    gated = [m for m in masks_fhw if m is not None]
    if not gated:
        raise ValueError("all spans are None -- a fully ungated per-query bias is a no-op; "
                         "call clear_video_key_bias() instead")
    b = gated[0].shape[0]
    s_video = gated[0].reshape(b, -1).shape[1]
    rows: list[Tensor] = []
    for i, (mask, span_len) in enumerate(zip(masks_fhw, span_lens)):
        if mask is None:
            rows.append(torch.zeros(
                b, span_len, s_video, device=gated[0].device, dtype=gated[0].dtype))
            continue
        if mask.dim() != 4:
            raise ValueError(f"span {i}: expected [B,F,H,W], got {tuple(mask.shape)}")
        flat = mask.reshape(mask.shape[0], -1)
        if flat.shape != (b, s_video):
            raise ValueError(
                f"span {i}: mask flattens to {tuple(flat.shape)}, expected ({b}, {s_video}) -- "
                "all span masks must share one batch and one [F,H,W] video-token lattice")
        key_bias = torch.log(flat.clamp(min=eps, max=1.0))           # [B, S]
        rows.append(key_bias.unsqueeze(1).expand(b, span_len, s_video))
    return torch.cat(rows, dim=1).unsqueeze(1)                       # [B, 1, T_total, S]


def install_v2a_mask_gate(model: torch.nn.Module) -> list[str]:
    """Wrap every `*video_to_audio_attn` module's forward to inject the stashed bias as `mask=`.

    Idempotent: modules already wrapped (marked with `_v2a_mask_gated`) are skipped. Returns the
    list of wrapped module names. Only wraps the video->audio direction, never audio->video.
    """
    wrapped: list[str] = []
    for name, module in model.named_modules():
        if not name.endswith("video_to_audio_attn"):
            continue
        if getattr(module, "_v2a_mask_gated", False):
            continue
        orig_forward = module.forward

        def make(orig):
            def forward(x, context=None, mask=None, **kwargs):
                if mask is None:
                    bias = _HOLDER.bias
                    if bias is not None:
                        ctx = context if context is not None else x
                        if bias.shape[0] != ctx.shape[0]:
                            raise ValueError(
                                f"v2a mask bias batch {bias.shape[0]} != context batch "
                                f"{ctx.shape[0]}")
                        if bias.shape[-1] != ctx.shape[1]:
                            raise ValueError(
                                f"v2a mask bias key-len {bias.shape[-1]} != #video tokens "
                                f"{ctx.shape[1]}")
                        # Two accepted shapes: [B,1,1,S] (one key bias broadcast over every
                        # audio query -- the single-target extractor) or [B,1,T_q,S] (per-query
                        # rows -- joint separation, where each stem span is gated by its own
                        # mask). SDPA broadcasts both; anything else is a layout bug.
                        if bias.shape[-2] not in (1, x.shape[1]):
                            raise ValueError(
                                f"v2a mask bias query-len {bias.shape[-2]} is neither 1 nor "
                                f"the audio query length {x.shape[1]} -- per-query bias rows "
                                "must cover the ENTIRE audio sequence (all spans)")
                        mask = bias.to(dtype=ctx.dtype, device=ctx.device)
                return orig(x, context=context, mask=mask, **kwargs)
            return forward

        module.forward = make(orig_forward)
        module._v2a_mask_gated = True                    # noqa: SLF001
        module._v2a_orig_forward = orig_forward          # noqa: SLF001
        wrapped.append(name)
    if not wrapped:
        raise RuntimeError(
            "install_v2a_mask_gate found no 'video_to_audio_attn' modules -- the model has no "
            "audio-video cross attention, so the mask gate has nowhere to attach")
    return wrapped


def uninstall_v2a_mask_gate(model: torch.nn.Module) -> None:
    for _name, module in model.named_modules():
        if getattr(module, "_v2a_mask_gated", False):
            module.forward = module._v2a_orig_forward    # noqa: SLF001
            del module._v2a_mask_gated
            del module._v2a_orig_forward
    clear_video_key_bias()
