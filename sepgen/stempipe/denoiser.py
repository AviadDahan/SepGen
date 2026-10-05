"""The method's guidance: per-span composition and the trained interface.

One `Denoiser` in upstream's sense — `(transformer, video_state, audio_state, sigmas,
step_index) -> (DenoisedLatentResult, DenoisedLatentResult)`. Everything our method does to
the *prediction* happens here; everything it does to *attention* happens in the gates.

Two facts drive the design:

* The res_2s loop calls a denoiser TWICE per step (the midpoint call arrives with a
  length-1 sigma tensor and `step_index=0`) plus once more at the terminal sigma. Anything
  scheduled must therefore key on SIGMA, never on the step index. The Euler loop calls once
  per step and can use either; it uses the step index so its beta curve is defined in the
  same currency it was tuned in.
* Upstream's CFG-rescale normalises over the whole modality tensor. On a three-span sequence
  that couples the spans: one source's variance shift would rescale the mixture. Every
  composition here is therefore per span.
"""
from __future__ import annotations

import numpy as np
import torch
from dataclasses import replace
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_pipelines.utils.helpers import modality_from_latent_state
from ltx_pipelines.utils.types import DenoisedLatentResult

from .masks import (
    apply_beta,
    beta_masks,
    block_diagonal_text_bias,
    sibling_complement_text_bias,
    span_allow_matrix,
    span_attention_bias,
)
from .stem_config import StemConfig

# Beta's decay is defined against a 30-step reference grid so the res_2s path (15 steps with
# midpoints) follows the same curve it was tuned on rather than a step count.
_BETA_REFERENCE_STEPS = 30

# sentinel: 'this pass routes captions the ordinary way'
_UNSET = object()


class JointStemDenoiser:
    def __init__(self, *, stem: StemConfig, gates, span_len: int,
                 v_context_p, v_context_n, stem_encodings, scene_encoding, negative_encoding,
                 video_guider_params: MultiModalGuiderParams,
                 audio_guider_params: MultiModalGuiderParams,
                 sigmas: torch.Tensor, device: torch.device, dtype: torch.dtype,
                 mix_negative_encoding=None, stem_negative_encodings=None):
        self.stem = stem
        self.gates = gates
        self.span_len = span_len
        self.device, self.dtype = device, dtype
        self.num_spans = stem.num_spans
        self.mix_lo = (self.num_spans - 1) * span_len

        gates.span_len = span_len

        # --- contexts --------------------------------------------------------
        blocks = [e.audio_encoding for e in stem_encodings] + [scene_encoding.audio_encoding]
        self.a_context_p = torch.cat(blocks, dim=1)
        self.a_context_n = negative_encoding.audio_encoding
        self.v_context_p, self.v_context_n = v_context_p, v_context_n
        widths = [b.shape[1] for b in blocks]
        masks = [getattr(e, "attention_mask", None) for e in stem_encodings]
        masks.append(getattr(scene_encoding, "attention_mask", None))

        self.widths = widths

        # --- per-channel negatives -------------------------------------------
        # With any per-span negative given, the negative side is built exactly like the
        # positive side: one block per span plus a block-diagonal bias, so each source is
        # pushed away from ITS OWN negative rather than a shared one. Without them the
        # negative stays a single block that every span reads (upstream behaviour).
        stem_negs = list(stem_negative_encodings or [])
        self.per_span_negatives = bool(stem_negs or mix_negative_encoding is not None)
        neg_blocks = [(stem_negs[k].audio_encoding if k < len(stem_negs)
                       else self.a_context_n) for k in range(stem.num_stems)]
        neg_blocks.append(mix_negative_encoding.audio_encoding
                          if mix_negative_encoding is not None else self.a_context_n)
        neg_masks = [(getattr(stem_negs[k], "attention_mask", None) if k < len(stem_negs)
                      else getattr(negative_encoding, "attention_mask", None))
                     for k in range(stem.num_stems)]
        neg_masks.append(getattr(mix_negative_encoding, "attention_mask", None)
                         if mix_negative_encoding is not None
                         else getattr(negative_encoding, "attention_mask", None))
        # Built ALWAYS, even when every block is the same shared negative: a span attending
        # its own copy of identical text under a block-diagonal mask is the same attention as
        # attending one shared block (the other blocks are exactly zero-weighted), and having
        # the negative match the positive's width is what lets the passes share one batched
        # forward.
        self.a_context_n_spans = torch.cat(neg_blocks, dim=1)
        self.neg_bias = block_diagonal_text_bias(
            span_len, [b.shape[1] for b in neg_blocks], neg_masks, dtype=dtype, device=device)

        # --- biases ----------------------------------------------------------
        self.text_bias = block_diagonal_text_bias(
            span_len, widths, masks, dtype=dtype, device=device) if stem.text_block_diagonal else None
        self.null_bias = sibling_complement_text_bias(
            span_len, widths, masks, dtype=dtype, device=device)
        self.span_base = span_attention_bias(
            span_len, span_allow_matrix(stem.num_stems, stem.span_sibling),
            dtype=dtype, device=device)
        self.beta_uniform, self.beta_diag = (
            beta_masks(span_len, self.num_spans, stem.beta.w, device=device)
            if stem.beta else (None, None))

        # --- guiders ---------------------------------------------------------
        self.video_guider = MultiModalGuider(params=video_guider_params,
                                             negative_context=v_context_n)
        self.mix_guider = MultiModalGuider(params=audio_guider_params,
                                           negative_context=self.a_context_n)
        self.stem_guider = MultiModalGuider(
            params=replace(audio_guider_params,
                           rescale_scale=stem.stem_rescale,
                           modality_scale=stem.stem_modality_scale),
            negative_context=self.a_context_n)

        # --- schedules -------------------------------------------------------
        self.sigmas = sigmas
        self._ref_grid = np.linspace(float(sigmas[0]), 0.0, _BETA_REFERENCE_STEPS + 1)
        self._n_steps = len(sigmas) - 1
        self._forwards = 0

    # ------------------------------------------------------------------ arming

    def arm(self, joint_state) -> None:
        self.gates.span_base = self.span_base
        self.gates.restore_span_base()
        if self.stem.nag is not None:
            self.gates.nag_configured = True
            self.gates.nag_scale = self.stem.nag.scale
            self.gates.nag_tau = self.stem.nag.tau
            self.gates.nag_alpha = self.stem.nag.alpha
            self.gates.nag_null_bias = self.null_bias

    def disarm(self) -> None:
        """Restore the armed base so beta's constant band cannot compound into the next cell."""
        self.gates.restore_span_base()
        self.gates.text_bias = None
        self.gates.nag_active = False
        self.gates.nag_configured = False


    # ------------------------------------------------------------------ helpers

    def _beta_frac(self, sigma: float, step_index: int) -> float:
        if self.stem.sampler == "euler":
            return min(1.0, step_index / max(1, self._n_steps))
        # sigma -> fraction on the reference grid (descending sigmas -> ascending fraction)
        return float(np.interp(sigma, self._ref_grid[::-1],
                               np.linspace(1.0, 0.0, _BETA_REFERENCE_STEPS + 1)))

    def _ti_fires(self, sigma: float, step_index: int) -> bool:
        ti = self.stem.trained_interface
        if not ti.enabled:
            return False
        if self.stem.sampler == "euler":
            return step_index >= ti.from_step
        return sigma <= ti.sigma_gate

    def _forward(self, transformer, video_state, audio_state, sigma, v_ctx, a_ctx, *,
                 audio_override=None, modality: bool = False,
                 text_bias=_UNSET, nag: bool = False, capture: str | None = None):
        """Run one guidance pass, DECLARING what it is.

        `text_bias` says how captions route on this pass (None = no routing, e.g. a single
        negative block every span reads); `nag` says whether NAG participates; `capture`
        names the stream an armed attention capture should file this pass under. The gates
        act on the declaration only -- they never guess from tensor shapes.
        """
        bias = self.text_bias if text_bias is _UNSET else text_bias
        with self.gates.pass_context(text_bias=bias, nag=nag, capture=capture):
            return self._raw_forward(transformer, video_state, audio_state, sigma, v_ctx,
                                     a_ctx, audio_override=audio_override, modality=modality)

    def _raw_forward(self, transformer, video_state, audio_state, sigma, v_ctx, a_ctx, *,
                     audio_override=None, modality: bool = False):
        b = audio_state.latent.shape[0]
        sig = sigma.reshape(1).expand(b) if sigma.dim() == 0 else sigma[:1].expand(b)
        v_mod = modality_from_latent_state(video_state, v_ctx, sig)
        a_mod = modality_from_latent_state(audio_state, a_ctx, sig)
        if audio_override is not None:
            a_mod = replace(a_mod, **audio_override)
        perturbations = None
        if modality:
            perturbations = BatchedPerturbationConfig(
                [PerturbationConfig([
                    Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                    Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
                ])],
                num_blocks=transformer.num_blocks,
                device=audio_state.latent.device,
                dtype=audio_state.latent.dtype,
            )
        self._forwards += 1
        return transformer(video=v_mod, audio=a_mod, perturbations=perturbations)

    def _compose_audio(self, pos, neg, mod, *, stem_neg=None):
        """Per span: stems on the stem guider, the mix on the preset's."""
        parts = []
        for k in range(self.num_spans):
            lo, hi = k * self.span_len, (k + 1) * self.span_len
            is_mix = k == self.num_spans - 1
            guider = self.mix_guider if is_mix else self.stem_guider
            uncond = neg[:, lo:hi]
            if stem_neg is not None and not is_mix:
                uncond = stem_neg[:, lo:hi]
            parts.append(guider.calculate(
                pos[:, lo:hi], uncond, pos[:, lo:hi],
                mod[:, lo:hi] if mod is not None else pos[:, lo:hi]))
        out = torch.cat(parts, dim=1)
        if out.shape != pos.shape:
            raise RuntimeError(f"per-span composition changed the shape: {out.shape} vs {pos.shape}")
        return out

    # ------------------------------------------------------------------ call

    def __call__(self, transformer, video_state, audio_state, sigmas, step_index):
        sigma = sigmas[step_index]
        s = float(sigma)

        # beta rides on the span-attention bias; the base is restored by disarm()
        if self.stem.beta is not None:
            self.gates.span_bias = apply_beta(
                self.span_base, self.beta_uniform, self.beta_diag,
                self.stem.beta.b0, self.stem.beta.bf, self._beta_frac(s, step_index))
        else:
            self.gates.span_bias = self.span_base

        need_neg = (self.video_guider.do_unconditional_generation()
                    or self.mix_guider.do_unconditional_generation())
        need_mod = (self.video_guider.do_isolated_modality_generation()
                    or self.mix_guider.do_isolated_modality_generation()
                    or self.stem_guider.do_isolated_modality_generation())

        # Only the conditional pass is named for capture: the negative pass is the model
        # reading what we do NOT want, and the modality pass runs with the audio-video cross
        # attention perturbed away entirely, so neither carries a source's placement.
        passes = [dict(v_ctx=self.v_context_p, a_ctx=self.a_context_p,
                       text_bias=self.text_bias, nag=True, capture="main")]
        if need_neg:
            passes.append(dict(v_ctx=self.v_context_n, a_ctx=self.a_context_n_spans,
                               text_bias=self.neg_bias, nag=False))
        if need_mod:
            passes.append(dict(v_ctx=self.v_context_p, a_ctx=self.a_context_p,
                               text_bias=self.text_bias, nag=True, modality=True))

        out = [self._forward(transformer, video_state, audio_state, sigma, p["v_ctx"],
                             p["a_ctx"], text_bias=p["text_bias"], nag=p.get("nag", False),
                             modality=p.get("modality", False), capture=p.get("capture"))
               for p in passes]
        pos_v, pos_a = out[0]
        neg_v, neg_a = out[1] if need_neg else (pos_v, pos_a)
        mod_v, mod_a = out[-1] if need_mod else (pos_v, pos_a)

        dv = self.video_guider.calculate(pos_v, neg_v, pos_v, mod_v)
        da = self._compose_audio(pos_a, neg_a, mod_a)

        if self._ti_fires(s, step_index):
            da = self._trained_interface(transformer, video_state, audio_state, sigma, da)

        return DenoisedLatentResult(denoised=dv), DenoisedLatentResult(denoised=da)

    # ------------------------------------------------------------------ trained interface

    def _trained_interface(self, transformer, video_state, audio_state, sigma, da):
        """Re-solve the SOURCES against a finished mixture.

        The adapter only ever saw sources next to a clean mix at timestep 0. Free-running
        joint generation never presents that, so from the gate onwards we build it: the
        mix rows are replaced by the guided estimate of the finished mixture and marked
        clean (timestep 0), while the source rows keep their own noise and their own
        timestep. One positive forward in that world, one negative (the sibling's caption,
        so `pos - neg` reads "my content, not my sibling's"), composed per span. The mix's
        own trajectory is never touched -- it is spliced back untouched -- so the mixture
        and the video are exactly what they would have been without this block.
        """
        mix_x0 = da[:, self.mix_lo:].to(audio_state.latent.dtype)
        lat = torch.cat([audio_state.latent[:, :self.mix_lo], mix_x0], dim=1)
        base_ts = audio_state.denoise_mask * sigma
        ts = base_ts.clone()
        ts[:, self.mix_lo:] = 0.0
        override = {"latent": lat, "timesteps": ts}

        # the negative direction: with the sibling negative, each source reads its SIBLING'S
        # caption, so `pos - neg` means "my content, not the other source's"
        sib = self.stem.trained_interface.sibling_negative
        ti_passes = [
            dict(v_ctx=self.v_context_p, a_ctx=self.a_context_p, text_bias=self.text_bias,
                 nag=True, capture="ti"),
            dict(v_ctx=self.v_context_p if sib else self.v_context_n,
                 a_ctx=self.a_context_p if sib else self.a_context_n_spans,
                 text_bias=self.null_bias if sib else self.neg_bias, nag=False),
        ]
        outs = [self._forward(transformer, video_state, audio_state, sigma, p["v_ctx"],
                              p["a_ctx"], text_bias=p["text_bias"], nag=p.get("nag", False),
                              audio_override=override, capture=p.get("capture"))
                for p in ti_passes]
        ti_pos, ti_neg = outs[0][1], outs[1][1]

        parts = []
        for k in range(self.num_spans - 1):
            lo, hi = k * self.span_len, (k + 1) * self.span_len
            parts.append(self.stem_guider.calculate(
                ti_pos[:, lo:hi], ti_neg[:, lo:hi], ti_pos[:, lo:hi], ti_pos[:, lo:hi]))
        stems = torch.cat(parts, dim=1)
        return torch.cat([stems.to(da.dtype), da[:, self.mix_lo:]], dim=1)
