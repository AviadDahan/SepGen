"""SepGen Cross-Stem Attention Guidance: Normalized Attention Guidance (NAG) on the audio text cross-attention.

NAG (arXiv:2505.21179; reference implementation ChenDarYen/Normalized-Attention-Guidance)
performs negative guidance in ATTENTION-FEATURE space instead of score space —
the regime where our six score-space null-branch variants all failed to move
the leaked words. Per audio_attn2 forward:

    Z+ = attn(x, positive context, positive block bias)
    Z- = attn(x, null context,     null block bias)
    G  = Z+ * scale - Z- * (scale - 1)                    # extrapolation
    r  = ||G||_1 / ||Z+||_1  (per token, feature dim)     # L1 ratio
    G  = G * min(r, tau) / r                              # norm clipping
    out= G * alpha + Z+ * (1 - alpha)                     # blend
    (reference defaults: tau 2.5, alpha 0.5; scale from usage 3-9)

The null here is the SWAPPED-PROMPT context (each stem span reads the sibling's
text) — the anti-leakage contrast, now applied inside attention features.

Composability: installed ON TOP of the text_block_gate wrapper. The Z+ call
passes the original (mask=None) args, so the gate injects the stashed positive
bias as usual; the Z- call passes the null bias EXPLICITLY (mask not None), so
the gate passes it through untouched. The CFG-negative forward's context width
differs from the null width, so NAG skips it automatically. audio_attn2 is not
hooked by the localization capture (it captures v2a/attn2/attn1 video-side
only), so the extra attention call is capture-invisible.

State is module-global (set per item, cleared after), matching the other gates.

"""

import torch

_STATE = {"enabled": False, "scale": 0.0, "tau": 2.5, "alpha": 0.5,
          "null_ctx": None, "null_bias": None}


def set_nag(scale: float, tau: float, alpha: float,
            null_ctx: torch.Tensor, null_bias: torch.Tensor) -> None:
    if scale <= 1.0:
        raise ValueError(f"nag scale must be > 1 to have any effect, got {scale}")
    _STATE.update(enabled=True, scale=float(scale), tau=float(tau), alpha=float(alpha),
                  null_ctx=null_ctx, null_bias=null_bias)


def clear_nag() -> None:
    _STATE.update(enabled=False, null_ctx=None, null_bias=None)


def install_nag_attn2(model: torch.nn.Module) -> list[str]:
    """Wrap every audio_attn2 forward (idempotent). Must run AFTER
    install_text_block_gate so Z+ keeps the block-diagonal routing."""
    wrapped: list[str] = []
    for name, module in model.named_modules():
        if not name.endswith("audio_attn2"):
            continue
        if getattr(module, "_nag_wrapped", False):
            continue
        gated_forward = module.forward  # the text-block-gate wrapper

        def make(inner):
            def forward(x, context=None, mask=None, **kwargs):
                out_pos = inner(x, context=context, mask=mask, **kwargs)
                st = _STATE
                if (not st["enabled"] or context is None or st["null_ctx"] is None
                        or context.shape[1] != st["null_ctx"].shape[1]):
                    return out_pos
                null_ctx = st["null_ctx"].to(device=context.device, dtype=context.dtype)
                null_bias = st["null_bias"].to(device=context.device, dtype=context.dtype)
                out_neg = inner(x, context=null_ctx, mask=null_bias, **kwargs)
                g = out_pos * st["scale"] - out_neg * (st["scale"] - 1.0)
                n_pos = out_pos.float().norm(p=1, dim=-1, keepdim=True).clamp_min(1e-12)
                n_g = g.float().norm(p=1, dim=-1, keepdim=True).clamp_min(1e-12)
                ratio = n_g / n_pos
                clip = torch.minimum(ratio, torch.full_like(ratio, st["tau"])) / ratio
                g = g * clip.to(g.dtype)
                return (g * st["alpha"] + out_pos * (1.0 - st["alpha"])).to(out_pos.dtype)
            return forward

        module.forward = make(gated_forward)
        module._nag_wrapped = True                                   # noqa: SLF001
        wrapped.append(name)
    if not wrapped:
        raise RuntimeError("install_nag_attn2 found no audio_attn2 modules")
    print(f"[nag] wrapped {len(wrapped)} audio_attn2 modules "
          f"(scale/tau/alpha armed per item)", flush=True)
    return wrapped
