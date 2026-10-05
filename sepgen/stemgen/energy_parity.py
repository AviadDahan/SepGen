"""Frozen-anchor ENERGY-PARITY guidance for joint-stem sampling (2026-07-28).

MOTIVATION (urmp_stem_collapse_phase0.md, per-step mel trace): the stem power drain is created
in the high-sigma commitment phase -- the trained model's x0-hat recruits less mel energy than
base as content commits, and the flat low-sigma tail never recovers it. Two measurement facts
shape this fix:
  * The FROZEN MIX span (arm N) is a live, trustworthy energy anchor: its per-step x0-hat is
    byte-identical to the base model's by construction. For incoherent sources, audio POWER is
    approximately additive, so sum_k P(stem_k) ~ P(mix) is the physically-grounded target.
  * Loudness lives in a DIRECTION of the per-channel-normalized audio latent space, NOT its
    norm (v1 trace: drained checkpoints have equal-or-higher latent RMS). A scalar latent
    rescale cannot change loudness; the correction must move the latent along the loudness
    direction via GRADIENTS through the (small, frozen, differentiable) audio VAE decoder.

METHOD. At each sampling step, after guidance composes the x0-hat: decode the mix span's x0-hat
to mel (no_grad; anchor), decode the stem spans' x0-hats to mel with grad, and take
parameter-free 1-D Gauss-Newton steps on the stems' tokens to close

    r = log( sum_k mean(exp(mel_k)) ) - log( mean(exp(mel_mix)) )        ->  |r| < tol

(z <- z - r * g / ||g||^2, the exact Newton step for r(z) linearized along its own gradient --
no learning rate, no tuned scale; tol and an iteration cap are fixed constants). Both stems
share one correction objective, so their RATIO is preserved -- this fixes the TOTAL energy
drain only, deliberately not the winner-take-all split (a separate problem, tracked as such).

This is inference-time only, checkpoint-agnostic, and needs no ground truth -- the anchor is
the model's own frozen-prior mix prediction. Related published mechanisms: Epsilon Scaling
(ICLR 2024) corrects per-step prediction MAGNITUDE against exposure bias; reconstruction /
universal guidance apply measurement gradients through frozen decoders during sampling.
"""
from __future__ import annotations

import torch
from torch import Tensor

# Fixed constants, not tuning knobs: tol = 0.05 nats ~ 0.4 dB residual parity error;
# the iteration cap only bounds worst-case cost (Newton typically converges in 2-3).
PARITY_TOL = 0.05
PARITY_MAX_ITERS = 6


def mel_power(mel: Tensor) -> Tensor:
    """Mean LINEAR mel power of a decoded log-mel spectrogram (natural log per ops.py)."""
    return mel.float().exp().mean()


@torch.enable_grad()
def parity_correct_stems(
    denoised_audio: Tensor,
    num_stems: int,
    span_len: int,
    decode_mel,
) -> tuple[Tensor, dict]:
    """Return denoised_audio with stem spans nudged toward energy parity with the mix span.

    Args:
        denoised_audio: [B, (num_stems+1)*span_len, D] post-guidance x0-hat (mix LAST).
        decode_mel: callable tokens [B, span_len, D] -> log-mel tensor (differentiable).
    Returns (corrected tensor, stats dict for logging).
    """
    mix_sl = slice(num_stems * span_len, (num_stems + 1) * span_len)
    with torch.no_grad():
        p_mix = mel_power(decode_mel(denoised_audio[:, mix_sl]))

    stems = (denoised_audio[:, : num_stems * span_len]
             .detach().float().clone().requires_grad_(True))

    r0 = None
    for _ in range(PARITY_MAX_ITERS):
        p_stems = sum(
            mel_power(decode_mel(stems[:, k * span_len:(k + 1) * span_len]))
            for k in range(num_stems))
        r = torch.log(p_stems) - torch.log(p_mix)
        if r0 is None:
            r0 = float(r)
        if abs(float(r)) < PARITY_TOL:
            break
        (g,) = torch.autograd.grad(r, stems)
        g_norm_sq = g.float().pow(2).sum()
        if not torch.isfinite(g_norm_sq) or g_norm_sq <= 0:
            break
        with torch.no_grad():
            stems -= (r / g_norm_sq) * g
        stems.requires_grad_(True)

    corrected = torch.cat(
        [stems.detach().to(denoised_audio.dtype), denoised_audio[:, mix_sl]], dim=1)
    return corrected, {"r0": r0, "r_final": float(r)}
