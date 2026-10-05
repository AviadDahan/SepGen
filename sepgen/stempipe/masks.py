"""Every mask the method needs, as pure functions.

Layout convention, assumed everywhere: the audio sequence is
`[span_0 | span_1 | ... | span_{K-1} | mix]` -- K sources followed by the mixture, THE MIX
LAST -- each span exactly `span_len` tokens.

Two kinds of bias are built here:
  * SELF-attention biases over the audio sequence, shape [1, 1, T, T] (T = spans*span_len);
  * CROSS-attention biases from audio tokens to the concatenated text blocks, shape
    [1, 1, T, sum(block widths)].
Both are additive log-space masks: 0.0 = attend, finfo.min = forbidden.
"""
from __future__ import annotations

import torch


def _neg(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min


# --------------------------------------------------------------------------- self-attention


def span_allow_matrix(num_stems: int, sibling: bool, mix_protected: bool = True) -> torch.Tensor:
    """[S, S] bool: allow[q, k] = span q may attend span k.

    `mix_protected`: the mix NEVER reads the stems, so the mixture's trajectory cannot be
    pulled by the decomposition it is supposed to be decomposed from. Stems always read
    themselves and the mix; `sibling` additionally lets them read each other.
    """
    spans = num_stems + 1
    allow = torch.eye(spans, dtype=torch.bool)
    mix = spans - 1
    allow[:mix, mix] = True                       # every stem reads the mix
    if sibling:
        allow[:mix, :mix] = True                  # stems read each other
    if not mix_protected:
        allow[mix, :mix] = True
    return allow


def span_attention_bias(span_len: int, allow: torch.Tensor, *, dtype: torch.dtype,
                        device: torch.device) -> torch.Tensor:
    """[1, 1, T, T] additive bias realising `allow` at token granularity."""
    spans = allow.shape[0]
    total = span_len * spans
    bias = torch.full((1, 1, total, total), _neg(dtype), dtype=dtype, device=device)
    for q in range(spans):
        for k in range(spans):
            if allow[q, k]:
                bias[..., q * span_len:(q + 1) * span_len, k * span_len:(k + 1) * span_len] = 0.0
    assert_no_dead_row(bias)
    return bias


def assert_no_dead_row(bias: torch.Tensor) -> None:
    """A fully-masked query row makes softmax return NaN. Never ship one."""
    finite = (bias > _neg(bias.dtype) / 2).any(dim=-1)
    if not bool(finite.all()):
        dead = (~finite).nonzero()[:5].tolist()
        raise ValueError(f"attention bias has fully-masked query rows, e.g. {dead}")


# --------------------------------------------------------------------------- text cross-attention


def _block_bounds(block_widths: list[int]) -> list[tuple[int, int]]:
    bounds, off = [], 0
    for w in block_widths:
        bounds.append((off, off + w))
        off += w
    return bounds


def _apply_block(bias: torch.Tensor, q0: int, q1: int, k0: int, k1: int,
                 valid: torch.Tensor | None) -> None:
    """Open span rows [q0,q1) onto text block [k0,k1), respecting that block's padding."""
    bias[..., q0:q1, k0:k1] = 0.0
    if valid is not None:
        pad = ~valid.to(torch.bool)
        if bool(pad.any()):
            bias[..., q0:q1, k0:k1][..., pad] = _neg(bias.dtype)


def block_diagonal_text_bias(span_len: int, block_widths: list[int],
                             block_masks: list[torch.Tensor | None] | None, *,
                             dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Each span attends ONLY its own caption block.

    `block_widths` are the widths of `[caption_0, ..., caption_{K-1}, scene]` in the
    concatenated context; span k gets block k, and the trailing mix span gets the scene
    block. This is what makes one forward pass carry K+1 different conditionings.
    """
    spans = len(block_widths)
    total_q, total_k = span_len * spans, sum(block_widths)
    bias = torch.full((1, 1, total_q, total_k), _neg(dtype), dtype=dtype, device=device)
    for k, (k0, k1) in enumerate(_block_bounds(block_widths)):
        mask = block_masks[k] if block_masks is not None else None
        _apply_block(bias, k * span_len, (k + 1) * span_len, k0, k1,
                     None if mask is None else mask.reshape(-1))
    assert_no_dead_row(bias)
    return bias


def sibling_complement_text_bias(span_len: int, block_widths: list[int],
                                 block_masks: list[torch.Tensor | None] | None, *,
                                 dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """The null used by NAG and by the sibling negative.

    Source span k attends every OTHER source's caption block (never its own); the mix span
    keeps the scene block, so the mix is a no-op under this null by construction -- which is
    why guidance built on it can never move the mixture.
    """
    spans = len(block_widths)
    num_stems = spans - 1
    if num_stems < 2:
        raise ValueError("the sibling complement needs at least 2 sources")
    total_q, total_k = span_len * spans, sum(block_widths)
    bias = torch.full((1, 1, total_q, total_k), _neg(dtype), dtype=dtype, device=device)
    bounds = _block_bounds(block_widths)
    for q in range(num_stems):
        for k in range(num_stems):
            if k == q:
                continue
            mask = block_masks[k] if block_masks is not None else None
            _apply_block(bias, q * span_len, (q + 1) * span_len, *bounds[k],
                         None if mask is None else mask.reshape(-1))
    scene_mask = block_masks[-1] if block_masks is not None else None
    _apply_block(bias, num_stems * span_len, spans * span_len, *bounds[-1],
                 None if scene_mask is None else scene_mask.reshape(-1))
    assert_no_dead_row(bias)
    return bias


# --------------------------------------------------------------------------- beta boost


def beta_masks(span_len: int, num_spans: int, band: int, *, device: torch.device
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """(uniform, diagonal) float masks over the stem-query x mix-key block.

    `uniform` covers every stem query against every mix key. `diagonal` covers only pairs
    within `band` latent frames of each other, i.e. "read the mix at your own time", which
    is what keeps the amplification from smearing a source across the whole clip.
    """
    mix_lo = (num_spans - 1) * span_len
    total = span_len * num_spans
    uniform = torch.zeros((1, 1, total, total), dtype=torch.float32, device=device)
    uniform[..., :mix_lo, mix_lo:] = 1.0

    diagonal = torch.zeros_like(uniform)
    t = torch.arange(span_len, device=device)
    near = (t[:, None] - t[None, :]).abs() <= band          # [span_len, span_len]
    for k in range(num_spans - 1):
        diagonal[..., k * span_len:(k + 1) * span_len, mix_lo:][..., near] = 1.0
    return uniform, diagonal


def apply_beta(base: torch.Tensor, uniform: torch.Tensor, diagonal: torch.Tensor,
               b0: float, bf: float, frac: float) -> torch.Tensor:
    """base + b0*(1-frac)*uniform + bf*diagonal, never lifting a forbidden entry.

    The decaying term fades as the schedule advances (early steps are where a source decides
    whether it exists at all); the diagonal term is constant.
    """
    boost = (b0 * (1.0 - frac)) * uniform + bf * diagonal
    allowed = base > _neg(base.dtype) / 2
    return torch.where(allowed, base + boost.to(base.dtype), base)


# --------------------------------------------------------------------------- staggering


def span_sigma_grid(sigmas: torch.Tensor, num_spans: int, span_len: int, *,
                    mix_lead_alpha: float = 1.0,
                    mix_sigma_track: tuple[float, ...] | None = None,
                    device: torch.device | None = None) -> torch.Tensor:
    """[K+1, 1, T, 1] per-token sigmas — the machinery behind staggered generation.

    Stems always follow `sigmas`. The mix span follows `mix_sigma_track` if given, else
    `mix_lead_alpha * sigmas` (alpha < 1 => the mixture runs ahead of its sources).
    """
    device = device or sigmas.device
    steps = sigmas.shape[0]
    total = span_len * num_spans
    grid = sigmas.to(device).reshape(steps, 1, 1, 1).expand(steps, 1, total, 1).clone()
    mix_lo = (num_spans - 1) * span_len
    if mix_sigma_track is not None:
        track = torch.tensor(mix_sigma_track, dtype=grid.dtype, device=device)
        validate_mix_track(track, sigmas.to(device))
        grid[:, :, mix_lo:, :] = track.reshape(steps, 1, 1, 1)
    elif mix_lead_alpha != 1.0:
        grid[:, :, mix_lo:, :] = grid[:, :, mix_lo:, :] * mix_lead_alpha
    return grid


def validate_mix_track(track: torch.Tensor, sigmas: torch.Tensor) -> None:
    if track.shape != sigmas.shape:
        raise ValueError(f"mix track has {tuple(track.shape)} entries, schedule has {tuple(sigmas.shape)}")
    if bool((track < 0).any()):
        raise ValueError("mix track has negative sigmas")
    if bool((track[1:] > track[:-1] + 1e-6).any()):
        raise ValueError("mix track is not monotone non-increasing")
    if bool((track > sigmas + 1e-6).any()):
        raise ValueError("mix track is behind the stems (mix sigma above the stem sigma)")
    if float(track[-1]) != 0.0:
        raise ValueError(f"mix track must end at exactly 0.0, ends at {float(track[-1])}")


def per_token_euler_step(latent: torch.Tensor, denoised: torch.Tensor,
                         sigma: torch.Tensor, sigma_next: torch.Tensor) -> torch.Tensor:
    """Euler update with per-token sigmas; tokens already at sigma 0 are held fixed.

    At a uniform grid this is exactly the stock scalar Euler step (unit-tested).
    """
    x, x0 = latent.float(), denoised.float()
    s, s_next = sigma.float(), sigma_next.float()
    step = torch.where(s > 0, (s_next - s) / s.clamp(min=1e-12), torch.zeros_like(s))
    return (x + (x - x0) * step).to(latent.dtype)
