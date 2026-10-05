"""SepGen: co-generate K speaker stems (+ optional mix) in ONE audio pass.

WHY THIS SHAPE (vs. the mix-conditioned single-stem extractor in ``iclora_mix.py``):

  * The pairwise mix-loss run failed by EAR in three ways: (1) the generated mix carried
    background music / ignored the prompt, (2) the two stems were incoherent with each other
    because each was a SEPARATE ``[stem | mix]`` pass, (3) stem and mix spans were exchangeable at
    generation (identical timesteps + RoPE + one shared caption).
  * This strategy fixes (2)/(3) structurally by co-generating BOTH stems in one audio sequence
    ``[stem0 | stem1 (| mix)]`` -- K parts of one performance, coupled through audio self-attention
    (``audio_attn1``) -- and fixes the conditioning bug behind (1) with PER-CHANNEL TEXT: each span
    attends only to its own caption via a block-diagonal mask on the audio text cross-attention
    (see ``text_block_gate.py``); the mix span (arms B/C) attends the SCENE caption.
  * TIMESTEPS: standard shared-timestep diffusion -- ONE sigma per step for video and every audio
    span. No per-channel timesteps, no clean-atom mixtures. Generation is the trained task: at
    inference every span denoises from noise jointly (see ``joint_stem_sampler.py``).

THE THREE ARMS differ ONLY in the mix span:
  * A -- ``include_mix=False``: audio = ``[stem0 | stem1]``. The mix at inference is the SUM of the
    two decoded stem waveforms (post-hoc, exact). No mix span, no mix loss.
  * B -- ``include_mix=True, mix_loss_weight>0``: audio = ``[stem0 | stem1 | mix]``; the mix carries
    a velocity loss and attends the scene caption -- a generated mix target.
  * C -- ``include_mix=True, mix_loss_weight=0``: same sequence but NO loss on the mix span
    (control). An unsupervised mix span has no learned velocity, so its generated mix is expected
    to be garbage -- C failing proves the mix-loss is load-bearing, not the villain.

Loss is a masked velocity MSE over the loss-bearing spans (every stem always; the mix only in B).

ARM D FAMILY (phase 10_3 second pass, `analysis/scene_dynamics_constrained.md` SS3.1/SS8):
on top of the arm-C configuration, an optional sigma-weighted ENERGY-PARTITION loss ties the
SUM of the predicted stems' per-40 ms mel-power energies to the GT mix's energy envelope:

    L_part = lambda_part * (1-sigma)^2 * mean_t [log(sum_k E_hat_k(t) + c) - log(E_mix(t) + c)]^2

with x0_hat_k = noisy_k - sigma * v_pred_k (exact in this parameterization), E_hat_k read off
the latent x0-hat by a FROZEN closed-form ridge probe (fit offline by build_mix_energy.py;
log E_hat(t) = w . norm([x0(t-1); x0(t); x0(t+1)]) + b, edge-padded), E_mix precomputed from
each segment's mix.wav (training-only data; nothing new at inference), and
c = 1e-3 * median(E_mix) per segment. Numerics: computed as logsumexp([logE_0, logE_1, log c])
in float32 outside autocast -- exactly equal to the SS3.1 form, immune to exp overflow on
off-manifold x0-hats. Arm D2 adds the analogous MIX-SPAN energy tie (supervises ONLY the mix
span's 25 Hz energy envelope, never its full content -- SS3.4's shortcut mitigation).

ARM E FAMILY (2026-07-25 corrective follow-up): the sum-only partition proved UNSTABLE --
a limit cycle where one span absorbs the whole mix (the sum-term is indifferent to WHICH span
carries the energy, and the velocity anchor does not block absorption). The fix is a
PER-SOURCE energy term using each stem's OWN GT energy profile (training-only label; nothing
new at inference):

    L_ps = lambda_ps * (1-sigma)^2 * mean_{k,t} [log(E_hat_k(t) + c) - log(E_k_GT(t) + c)]^2

E_hat_k read by the SAME frozen ridge probe on x0_hat_k; E_k_GT precomputed from each
segment's stem_k.wav by build_stem_energy.py (formula identical to the mix energy); c is the
SHARED floor 1e-3 * median(E_mix) -- so GT-silent slices target log(c), directly penalizing a
stem that carries the sibling's voice in its own silent half (the anti-absorption force).
Arm E = per-source only (partition off); arm E2 = per-source + sum-partition combined.

ARM F (mix-leads staggered denoising, 2026-07-25): fixes SCHEDULE COMMITMENT rather than the
loss. At inference the coarse who-speaks-when structure commits during HIGH-sigma denoising,
exactly where training on GT-noised latents can only teach conditional-mean averaging (the
(1-sigma)^2-suppressed energy losses are unlearnable there; only ~6% of shifted-logit-normal
draws land in their teachable band). Arm F staggers the schedule instead: the MIX span trains
at sigma_mix = mix_lead_alpha * sigma (always < sigma for alpha < 1) while the stems train at
sigma, so stems at high sigma condition on an already-committed mix through the existing
cross-span self-attention (audio_attn1). Mechanically: per-token timesteps are native
(Modality.timesteps is (B, T), per-token AdaLN), and the velocity target v = eps - x0 is
sigma-FREE, so each span stays a valid flow-matching objective at its own sigma -- only the
noising level and the per-token timesteps change. mix_lead_alpha = 1.0 reproduces the shared
schedule byte-identically (same RNG order, same values). Validation REQUIRES the matching
staggered loop (joint_stem_sampler._run_denoising_mix_leads): the frozen sampler applies one
scalar sigma to all audio tokens per step.

ARM H (span identity in the RoPE channel budget, 2026-07-25): all three spans today sit at
IDENTICAL rotary positions (ref_time_offset = 0.0), so no attention head can form a "my span vs.
the other span" term -- the measured attention flow shows stem->sibling below chance everywhere
plus an unexplained slot asymmetry under fully symmetric geometry. Arm H reserves audio HEAD 0's
lowest `span_axis_pairs` frequency pairs (13 = the measured free set: they rotate < 45 deg over a
whole span) and overwrites their (cos, sin) with a constant per-span phase omega * coord, giving
a constant cross-span rotation in the audio self-attention logit and NOTHING else: 31 of 32 heads
stay bit-identical, and the above-Nyquist pairs that make "same time index" a sharp event across
spans are untouched. Coordinates (-1, +1, 0) for [stem0, stem1, mix] with omega = pi/2 put the
mix at +-90 deg carrying the speaker's SIGN and the sibling at the sign-free antipode. This
strategy class only CARRIES the three config fields; the mechanism lives in
stemgen/span_rope_gate.py (installed by train_jointstem.py), and span_axis_pairs = 0 (default)
means the gate is never installed -- arms A-G byte-identical.

ARM K (asymmetric cross-span mask on the audio SELF-attention, 2026-07-25): today NOTHING masks
audio_attn1, so every span attends every span -- and the measured consequence is that the MIX span
puts 0.617 of its attention mass on the two STEM spans (analysis/attnflow_probe.md, arm C step-1500),
where the frozen base model was pretrained with a single audio stream whose self-attention context
is 100 % itself. The mix span -- the one the user has verified the base model renders well -- is
therefore computed OFF-DISTRIBUTION by construction, contaminated by tokens that are pure noise at
high sigma and collapsed garbage in exactly our failure modes. The identical bug was found and fixed
once already in phase 10_1 (nightrun/2026-07-20/p10_1ms_234415/ledger.md, "the mix slot was reading
the stems"): after forcing the mix slot to attend only itself, the step-0 mix matched the base
model's own single-stream generation (spectral-flatness position +0.11 vs +0.13, energy -22.0 dB vs
-22.0 dB). Phase 10_3 never carried that mask over. `span_attention_topology = "mix_protected"`
restores it: the mix span attends ONLY itself, stems attend own + mix, and the stem<->sibling channel
is a SEPARATE switch (`span_attention_sibling`) so the two hypotheses can be tested independently.
Default "full" returns a None mask -- arms A-H byte-identical. Mechanism + risk analysis:
stemgen/audio_span_gate.py and analysis/span_attention_mask_design.md. HONEST SCOPE: the
generated-mix-leader probe (analysis/probe_genmix_leader_result.md) already showed that supplying a
clean leader does NOT repair stem content, so this mask is expected to buy prior preservation and a
mix span whose quality no longer depends on stem state -- not a stem-content fix.

ARM CHAN (mix-channel conditioning, 2026-07-29): arms K/N/A2V all shape the ATTENTION pathway the
mix reaches the stems through, but attention is still an OPTIONAL channel gradient descent can let
atrophy -- the stems' loss is satisfiable from caption + first frame alone. This arm makes the mix
structurally UNAVOIDABLE by injecting it directly into the stem spans' INPUT tokens
(ControlNet/InstructPix2Pix-style conditioning): ``stem_input[t] += W(mix_token[t])``, W a
zero-initialized ``nn.Linear`` (stemgen/mix_channel_adapter.py, ``MixChannelAdapter``), so at
construction the adapter is an EXACT no-op -- checkpoint-0 training and generation are
byte-identical to the unmodified strategy. Unlike ``prepare_training_inputs`` (strategy-side,
training-only), the injection is wired onto the ONE shared ``audio_patchify_proj`` submodule the
transformer itself owns (``x = self.patchify_proj(modality.latent)``, called exactly once per
forward on the raw patchified+noised audio latent) -- installed by ``train_jointstem.py`` after the
trainer is built, the same pattern arms H/K/N/A2V already use for exactly this reason. Because
``LtxvTrainer.train`` constructs its validation sampler as
``ValidationSampler(transformer=self._transformer, ...)`` (the LITERAL SAME object, not a copy),
this one wrap fires identically on the training forward AND every validation/sampling forward
(positive, CFG-negative, STG-perturbed alike, since all three share ``modality.latent`` and differ
only in ``context``/``perturbations``) -- true train+inference symmetry with zero changes to
``joint_stem_sampler.py``. ``mix_channel_conditioning = False`` (default) means the gate is never
armed -- arms A-M byte-identical. Requires ``include_mix``. The adapter's own params are NOT swept
up by ``LtxvTrainer._collect_trainable_params`` (a plain ``nn.Module``, not a PEFT/LoRA adapter), so
``train_jointstem.py`` must construct it, attach it to ``strategy._mix_channel_adapter`` (a stash
point only -- NOT read inside ``prepare_training_inputs``), and append its params to
``trainer._trainable_params`` itself, before ``trainer.train()`` builds the optimizer. See
stemgen/mix_channel_adapter.py for the full mechanism writeup.

ARM I (self-bootstrapping the mix span, 2026-07-25): the audited failure is textbook EXPOSURE BIAS.
Training shows the stems a CLEAN GROUND-TRUTH mix through the cross-span self-attention, and at
inference no such mix exists (analysis/mixleads_audit.md BUG 1; Self Forcing arXiv 2506.08009 names
and fixes exactly this). With probability `selfboot_mix_p` per sample per step, the mix span's clean
latent is replaced by the model's OWN cached generated mix (JEN-1 Composer Sec. 4.3: an EMA
teacher's generated conditioning track replaces the GT track with p = 0.5, in audio multi-track
latent diffusion -- the closest published analogue). The mix span is a pure INPUT in every live arm
(mix_loss_weight = 0), so the swap adds NO supervision, and it happens BEFORE noising, so each
span keeps its exact (1-s)*x0 + s*eps convention and sigma-free velocity target v = eps - x0.
Nothing new at inference: the cache is training-only, built offline by
scripts/build_selfboot_mix_cache.py. selfboot_mix_p = 0.0 (default) means the whole block is
skipped -- no extra data source, no extra RNG draw, arms A-K byte-identical. Design + falsifiers:
analysis/selfboot_design.md.

ARM CONS (decode-domain mixture-consistency loss, 2026-07-29): none of the arms above ever require
the generated stems' CONTENT to sum back to the mix -- the failure this arm attacks is exactly
that drift (urmp_stem_collapse_phase0.md). A LATENT-space version of this idea is already
RETRACTED by direct measurement in that analysis (its "encoder-linearity probe": `enc(stem_0 +
stem_1) != enc(stem_0) + enc(stem_1)`, and the naive-sum predictor is WORSE than predicting zero),
so tying x0-hat latents to each other would train against the VAE's own geometry. Physical audio
POWER *is* approximately additive for incoherent sources, but only after the nonlinear VAE decode
into (log-)mel space -- so this loss inverts each span's x0-hat (same construction as
`_compute_partition_losses`: `x0_hat = noisy_span - sigma * v_pred`), decodes stem0/stem1/mix
through the TRUE (frozen) audio VAE decoder, and ties them in LOG-mel space with the same
logsumexp-plus-floor recipe already used for the probe-based energy losses: `loss_cons = L1(
logsumexp([mel_stem0, mel_stem1, log_c]), mel_mix )` (log(sum(exp)) is the log-of-linear-sum,
i.e. correctly additive in POWER, never naively adding the log values themselves). Layered on
top of arm N (frozen-prior mix): the mix span's x0-hat is base-model-computed and trustworthy by
construction, so this is the first arm where the MIX side of the comparison needs no supervision
of its own -- the gradient this loss contributes has somewhere trustworthy to pull the stems
toward. mix_consistency_weight = 0.0 (default) means the whole mechanism -- including touching
`self._audio_decoder`, which does not even exist as a real decoder until train_jointstem.py wires
it in -- is skipped; every prior arm byte-identical. Design: analysis/urmp_stem_collapse_phase0.md
(SS "Encoder-linearity probe"); cost feasibility pre-measured by
scripts/measure_audio_decoder_cost.py.

ARM DECORR (stem-pair decorrelation loss, 2026-07-29): CONS guarantees the stems' summed CONTENT
adheres to the mix (adherence), but adherence alone does not force the two stems to be DISTINCT
from each other -- a real, measured, SEPARATE failure mode. Direct measurement on the arm-A (SUM,
no mix span) checkpoint at step 3000 (`.scripts/analyze_sum_final.py`) shows winner-take-all
energy collapse on URMP mixes (Jupiter: stem levels -46.4 dB vs -34.8 dB approx-equal-mix; Spring:
-47.1 dB vs -32.9 dB) AND, on the same-instrument duet (Sonata, violin+violin), the two stems'
mel-envelopes correlate at +0.99 -- near-identical duplicates, not a separated pair. Adherence-only
losses (CONS) cannot see this: two duplicate stems that both equal half the mix still sum back to
the mix correctly. This arm adds an explicit anti-duplication term: decode stem0 and stem1's
x0-hats through the true frozen audio VAE decoder (SAME construction as CONS's `_compute_
mix_consistency_loss` -- x0_hat = noisy_span - sigma*v_pred, unpatchify, `self._audio_decoder`),
flatten each batch element's decoded log-mel to a vector, and compute the differentiable Pearson
correlation between the two stems' flattened mels (mean-center, dot product, normalize by norms --
the exact same operation as `analyze_sum_final.py`'s numpy `corr()`, reimplemented in torch with
gradients flowing back through the decoder into both stems' x0-hats). Only the POSITIVE part of
the correlation is penalized: `loss_decorr = stem_decorrelation_weight * relu(corr)^2`. This is
deliberate -- the underlying data is a MUSIC DUET where both instruments legitimately play
simultaneously (same activity timing), so penalizing correlation symmetrically would also punish
correct simultaneous-but-distinct content; only high POSITIVE correlation (near-duplicate content)
is the measured failure, so negative correlation is left unpenalized and unrewarded. Independent
of CONS: does not require `include_mix` (there is no mix span in this comparison, only stem0 vs
stem1) and keeps its OWN same-step stash (`self._decorr_state`, not `self._cons_state`) and its
own decode call, so DECORR can run standalone (e.g. layered on arm A/SUM, which has no mix span at
all) or alongside CONS. stem_decorrelation_weight = 0.0 (default) means the whole mechanism is
skipped -- every prior arm byte-identical. Requires num_stems == 2 (the pairwise formula does not
generalize to K > 2 stems without a design decision on which pairs to penalize, so this raises
NotImplementedError rather than silently ignoring stems beyond the first two).

KNOWN LIMITATIONS (2026-07-29 launch, not yet resolved by measurement):
  (1) The correlation mean-centers each stem's mel GLOBALLY (one scalar mean over the whole
      flattened [C, T', F'] tensor), not per mel-bin. Log-mel spectra share a strong FREQUENCY-
      AXIS TILT across almost any two audio signals from the same scene/microphone chain (low
      mel bins louder than high ones) -- a structural similarity that has nothing to do with
      CONTENT duplication. If that shared tilt dominates the flattened vector's variance, the
      baseline stem_corr could sit well above 0 even for genuinely distinct, non-duplicated
      stems, leaving this term less headroom than the identical-stems synthetic test (corr ~= 1)
      suggests. Watch the printed `stem_corr` component at step 0 against the loss_velocity
      magnitude to see whether this bites in practice; a per-mel-bin (temporal-only) correlation
      is the fallback design if the global-mean version proves too coarse.
  (2) relu(corr)^2 is minimized (-> 0) whenever EITHER stem has near-zero variance after mean-
      centering -- which is exactly what a near-silent/collapsed stem produces. This term
      therefore does NOT, by itself, oppose the winner-take-all energy collapse this arm's
      analysis motivated it with (analyze_sum_final.py's -46.4 dB vs -34.8 dB); a stem driven to
      silence trivially satisfies "not duplicated". It only attacks the SEPARATE symptom
      (near-identical duplicate content when both stems ARE active), and is deliberately layered
      on top of the existing frozen-mix-lora / mix-protected-attention setup this yaml carries
      unchanged, not shipped as a standalone anti-collapse fix.
"""
from __future__ import annotations

import math
from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_core.types import AudioLatentShape
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)

from stemgen.audio_span_gate import (
    SpanAttentionTopology,
    build_span_allow_mask,
    build_span_allow_matrix,
)
from stemgen.spatial_mask_gate import set_video_key_bias, spatial_masks_to_per_query_bias
from stemgen.text_block_gate import build_block_diagonal_text_bias


class JointStemConfig(TrainingStrategyConfigBase):
    """Configuration for joint K-stem dialogue generation."""

    name: Literal["text_to_video", "video_to_video"] = "text_to_video"

    num_stems: int = Field(default=2, ge=2, description="Number of speaker stems co-generated per pass")

    include_mix: bool = Field(
        default=False,
        description=(
            "Whether the audio sequence carries a MIX span after the stems ([stem0|stem1|mix]). "
            "False = Arm A (no mix span; the inference mix is the sum of decoded stems). True = "
            "Arms B/C (mix span present, attending the scene caption)."
        ),
    )

    mix_loss_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight of the velocity loss on the MIX span. >0 = Arm B (the mix is a supervised "
            "generation target). 0 = Arm A (no mix span) or Arm C (mix span present but "
            "UNsupervised -- the control). Requires include_mix when >0."
        ),
    )

    ref_time_offset: float = Field(
        default=0.0,
        description=(
            "Seconds to shift the mix span's RoPE positions relative to the stems. 0.0 = fully "
            "time-aligned (all spans share the same 1-D audio time axis), matching the joint "
            "generation sampler."
        ),
    )

    span_pe_layout: Literal["aligned", "concat_mix_first"] = Field(
        default="aligned",
        description=(
            "ARM CATPE. How the three audio spans share the 1-D RoPE time axis. 'aligned' (the "
            "default, every prior arm): all spans claim the SAME 0-D second range -- three "
            "parallel channels at identical timestamps ('mutual PEs'). 'concat_mix_first': each "
            "span gets its own slice of ONE timeline, mix first --\n"
            "    [ mix 0-D | stem0 D-2D | stem1 2D-3D ]      D = span duration (~4.56 s)\n"
            "implemented as VIRTUAL per-span offsets on the positions (mix +0, stem k +(k+1)*D) "
            "with the SEQUENCE order [stem0|stem1|mix] untouched. RoPE attention depends only on "
            "position values, never token index, so this is geometrically identical to physically "
            "reordering the tokens -- and it leaves every mix-LAST assumption (a2v_mix_gate's "
            "trailing span, loss masks, lora_span_gate, sampler decode slices) intact. "
            "CONSEQUENCE, by design: cross-attention PEs derive from these same positions, and "
            "the video's time support is only 0-D -- so both stems become RoPE-far from every "
            "video token and A/V time alignment survives for the MIX span only. 3*D = 13.68 s "
            "fits the checkpoint's 20 s audio max_pos (4 spans would; 5 would silently "
            "extrapolate)."
        ),
    )

    span_axis_pairs: int = Field(
        default=0,
        ge=0,
        le=32,
        description=(
            "Arm H (span identity in the RoPE channel budget): how many of audio HEAD 0's LOWEST "
            "frequency pairs are reserved to carry a constant per-span phase. 0 = OFF (arms A-G "
            "byte-identical -- the gate is never installed). 13 = the measured free set (those "
            "pairs rotate < 45 deg over a whole 113-token span, so their time content is "
            "negligible at our scale); head 0 owns 32 pairs, hence the ceiling. See "
            "stemgen/span_rope_gate.py and analysis/span_identity_design.md."
        ),
    )

    span_axis_omega: float = Field(
        default=math.pi / 2,
        gt=0.0,
        description=(
            "Radians of RoPE rotation per unit span coordinate on the reserved pairs. pi/2 is "
            "forced from both sides: above it the sibling rotation 2*omega wraps past pi and the "
            "sibling starts looking CLOSER again; below pi/4 the span rotation is confusable with "
            "the <= 45 deg of genuine within-span time rotation the reserved pairs still carry."
        ),
    )

    span_axis_coords: list[float] = Field(
        default_factory=lambda: [-1.0, 1.0, 0.0],
        description=(
            "Span coordinate per span in sequence order [stem0, stem1 (, mix)]. The default "
            "(-1, +1, 0) is EXCHANGE-SYMMETRIC: swapping the speakers is exactly c -> -c, the "
            "stem<->mix rotation is +-90 deg and carries the speaker's SIGN, and the sibling sits "
            "at the sign-free antipode (180 deg). Length must equal num_stems + include_mix. "
            "(0, 0, 0) is the channel-sacrifice CONTROL, not a parity setting."
        ),
    )

    mix_lead_alpha: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description=(
            "Arm F (mix-leads staggered denoising): the mix span trains at sigma_mix = "
            "mix_lead_alpha * sigma while the stems train at sigma, delivered to the DiT as "
            "per-token timesteps. 1.0 = shared schedule (all prior arms byte-identical). "
            "< 1.0 requires include_mix and the matching staggered validation loop "
            "(joint_stem_sampler._run_denoising_mix_leads). NOTE: when mix_sigma_mode = "
            "'independent' (arm G), mix_lead_alpha is IGNORED for training (sigma_mix is drawn "
            "i.i.d., not alpha*sigma) and only parametrises the mix-leads VALIDATION schedule."
        ),
    )

    mix_sigma_mode: Literal["leads", "independent", "independent_leq"] = Field(
        default="leads",
        description=(
            "How the MIX span's training sigma is set (Diffusion-Forcing-style per-span "
            "granularity). 'leads' (default, arms A-F byte-identical): sigma_mix = "
            "mix_lead_alpha * sigma, a DETERMINISTIC fixed-alpha map of the stem sigma -- "
            "out-of-distribution at inference for any leader mix the fixed map never produced. "
            "'independent' (arm G): sigma_mix is drawn INDEPENDENTLY from the SAME timestep "
            "sampler as the primary (stem) draw, decoupled from the stem sigma, so EVERY "
            "(sigma_stem, sigma_mix) pair is in-distribution -- a committed leader mix (low "
            "sigma_mix) under high-sigma stems is then trained-for. "
            "'independent_leq': the STEMS are drawn first "
            "(the primary draw), then sigma_mix is a second i.i.d. draw from the same sampler "
            "CLAMPED to the stem sigma (torch.minimum) -- so the mix is ALWAYS at or below the "
            "stems' noise level, i.e. the mix leads by construction, and sigma_mix > sigma_stem "
            "pairs the generation mechanism will never visit are never trained. The clamp puts "
            "an atom at sigma_mix = sigma_stem (probability ~ P(second draw >= stem draw)); the "
            "marginal below it follows the sampler. Requires include_mix."
        ),
    )

    sep_pattern_p: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Arm M (MGE-LDM-style track-aware timestep patterns, arXiv:2505.23305): per SAMPLE, "
            "with this probability the training step uses the SEPARATION pattern -- the mix span "
            "is presented CLEAN (sigma_mix = 0, tokens = un-noised GT mix, per-token timestep 0) "
            "as an OBSERVED track, and the loss is computed on the STEM spans only (the mix is "
            "context, never a target, so it cannot be corrupted by this pattern's gradients). "
            "This trains the exact conditional the leader schedules need at inference -- 'stems "
            "given a fully committed mix' -- which the continuous sampler otherwise NEVER "
            "produces (ShiftedLogitNormal reflects draws below 1e-3, so sigma_mix = 0 is "
            "out-of-distribution in every prior arm; see joint_stem_sampler."
            "build_mix_sigma_track_early_finish's caveat). With probability 1 - p the JOINT "
            "pattern runs unchanged (mix_sigma_mode applies; combine with mix_loss_weight > 0 to "
            "also anchor the mix's own generation). 0.0 = off, every prior arm byte-identical "
            "including the RNG stream. Requires include_mix."
        ),
    )

    input_perturbation_gamma: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description=(
            "Arm IP (DDPM-IP input perturbation, Ning et al. ICML 2023, arXiv:2301.11706): "
            "during training, every AUDIO span's noised input is built with a PERTURBED noise "
            "eps' = eps + gamma * xi (xi a fresh standard Gaussian) while the velocity target "
            "stays v = eps - x0 (UNperturbed) -- explicitly simulating inference-time prediction "
            "error in the network input so the model learns to denoise from slightly "
            "off-manifold states instead of only exact GT-noised ones. This attacks the "
            "root-caused stem power drain (2026-07-28: exposure bias in the 30-step recursion, "
            "confirmed by elimination -- teacher-forced calibration improves while generation "
            "drains; guidance/schedule/mix/data refuted). The paper's single robust default is "
            "gamma = 0.1 across all datasets. sigma-scaled ((1-s)x0 + s*eps'), so sigma = 0 "
            "spans (arm M's clean observed mix) are untouched automatically. 0.0 = off, prior "
            "arms byte-identical including the RNG stream (the extra randn is never drawn)."
        ),
    )

    frozen_mix_lora: bool = Field(
        default=False,
        description=(
            "Arm N (frozen-prior mix): zero the LoRA delta on the MIX span's tokens in every "
            "audio-side adapted module (stemgen/lora_span_gate.py), so the mix span's forward "
            "pass is bit-identical to the FROZEN base model at every step -- checkpoint-0 mix "
            "quality by construction, nothing to drift, no mix loss needed. Requires "
            "include_mix, span_attention_topology='mix_protected' (frozen weights must also "
            "READ only base-model inputs: itself + video + own text), mix_loss_weight == 0 "
            "(the mix pathway carries no LoRA params, so a mix loss would be dead compute), and "
            "a lora.target_modules WITHOUT audio_attn2/video_to_audio_attn to_k/to_v (those "
            "run over shared text/video tokens and cannot be span-gated -- install fails loud). "
            "The training script must call lora_span_gate.set_lora_span_gate + "
            "install_lora_span_gate after building the trainer."
        ),
    )

    a2v_mix_only: bool = Field(
        default=False,
        description=(
            "Restrict audio_to_video_attn (VIDEO reading audio) to the MIX span's keys only "
            "(stemgen/a2v_mix_gate.py). Closes arm N's remaining second-order leak (stems -> "
            "video -> mix) and is semantically the intended coupling: video syncs to the scene "
            "audio, not to internal decomposition channels. Requires include_mix. Applies to "
            "training and every sampling pass (installed by the training script)."
        ),
    )

    span_attention_topology: SpanAttentionTopology = Field(
        default="full",
        description=(
            "Arm K (asymmetric cross-span mask on the audio SELF-attention, audio_attn1). "
            "'full' (default) = legacy unrestricted attention: NO mask is built at all "
            "(Modality.attention_mask stays None, the unmasked kernel is taken), so arms A-H are "
            "byte-identical. 'mix_protected' = the mix span attends ONLY itself -- token-for-token "
            "the single-audio-stream self-attention shape the frozen base model was pretrained on "
            "-- while every stem attends its own span + the mix. Requires include_mix. See "
            "stemgen/audio_span_gate.py and analysis/span_attention_mask_design.md."
        ),
    )

    span_attention_sibling: bool = Field(
        default=False,
        description=(
            "Arm K, SEPARATE switch for the stem->SIBLING-stem channel (independent of the "
            "mix->stem channel above, so the two hypotheses are testable one at a time). True = "
            "stems may also attend the other stem span(s) -- the MINIMAL change from legacy, "
            "removing only the mix's view of the stems. False = each stem sees own + mix only, "
            "which additionally makes a stem's context invariant to K. Evidence for False: "
            "stem->sibling attention is BELOW the 1/3 chance line in every arm (0.235 / 0.215 / "
            "0.207) and is LOWEST in the arm with real turn-taking (analysis/attnflow_probe.md). "
            "Ignored when span_attention_topology='full'."
        ),
    )

    first_frame_conditioning_p: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Probability of first-frame conditioning (TI2AV)"
    )

    video_sigma_zero: bool = Field(
        default=False,
        description=(
            "Pin the VIDEO stream at sigma = 0, i.e. present it CLEAN at every step instead of "
            "noising it on the audio's schedule (user decision 2026-08-12). The video is never a "
            "prediction target here (`video_loss_weight` 0), so it inherited the generative "
            "schedule for no reason; this is the same mechanism `sep_pattern_p` already applies to "
            "the mix span. The sigma is still DRAWN and then zeroed, so the downstream randn order "
            "-- and every other arm's RNG stream -- is untouched. Note this makes "
            "`first_frame_conditioning_p` provably inert: with sigma = 0 both branches of the "
            "first-frame torch.where are the same tensor, so set it to 0.0 rather than leaving a "
            "config value that does nothing. Inference must then present the video clean too "
            "(`localize.py --video-conditioning observed_clean`); measured cost of that at "
            "inference on a NOISE-trained checkpoint: nothing gained on two dialogue clips and a "
            "collapse to chance on the spatial-prompt cells."
        ),
    )

    positional_conditions_suffix: str = Field(
        default="",
        description=(
            "Suffix of a SECOND set of text-condition dirs living beside the first ones in the "
            "same preprocessed root (e.g. '_pos' -> conditions_pos, conditions_scene_audio_pos, "
            "conditions_stem_audio{k}_pos). Empty = single register, and no extra data source is "
            "registered at all. Used with `positional_prompt_p` to train one checkpoint on both "
            "prompt registers without duplicating the roster -- a second copy of every segment "
            "would double the epoch accounting and make the roster lie about how many distinct "
            "clips exist."
        ),
    )

    register_mix: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "N-WAY PROMPT REGISTER MIX, declared in the config so the split is a stated property "
            "of the run rather than a constant buried in code. Maps a condition-dir SUFFIX to a "
            "weight; the empty string is the primary (unsuffixed) register. An even three-way mix "
            "of semantic / short / positional is `{'': 1, '_short': 1, '_pos': 1}`, and the "
            "weights are normalised, so {1,1,1} and {2,2,2} are the same run. Every suffix must "
            "have its four condition dirs (conditions{sfx}, conditions_scene_audio{sfx}, "
            "conditions_stem_audio{k}{sfx}) present in the SAME preprocessed root. Empty = single "
            "register, no draw, and every prior arm is byte-identical. Supersedes the two-way "
            "`positional_conditions_suffix` / `positional_prompt_p`, which still work for the arms "
            "that used them."
        ),
    )

    register_avail_dir: str | None = Field(
        default=None,
        description=(
            "PER-SEGMENT REGISTER AVAILABILITY: a data source holding, for each segment, which "
            "registers may be drawn for it. Without this, mixing N registers forces the roster "
            "down to their INTERSECTION -- a segment any one register cannot express is lost to "
            "all of them. That cost is not abstract: the short register cannot name two same-"
            "gender speakers apart (both stems become 'A man.'), and the positional register "
            "cannot say 'on the left' about two sources sharing a half, so a strict intersection "
            "silently deletes 586 of v7.gen's 4,020 segments -- including every same-gender "
            "turn-taking pair, which are the HARDEST and most informative rows in the set (user, "
            "2026-08-13: 'same gender examples are important'). With it, every segment stays and "
            "each register is drawn only where it can be truthfully stated, the remaining weights "
            "renormalised per sample. Each file holds `avail`, a float vector over the registers "
            "in the order `[''] + sorted(alts)`; the preflight checks that order against the "
            "config, and the router checks its width. None = every register available everywhere."
        ),
    )

    positional_prompt_p: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Per-sample probability of taking the ALTERNATE (suffixed) text register instead of "
            "the primary one. ALL FOUR text tensors switch together -- video, scene/mix and both "
            "stems -- because a sample whose video says one thing and whose stems say another "
            "re-creates the video/mix text split v7.gen was built to remove. p = 0 takes no RNG "
            "draw at all, so every prior arm stays byte-identical (same discipline as "
            "`stem_caption_dropout_p` and `sep_pattern_p`)."
        ),
    )

    video_loss_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight on the video loss. Zero by default: the video stream is along for the ride "
            "(audio-conditioning source), never a training target here."
        ),
    )

    partition_loss_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "lambda_part of the energy-partition loss (arm D family). 0 = off (arms A/B/C are "
            "byte-identical). >0 adds the sigma-weighted log-domain partition term tying "
            "sum_k E_hat_k(t) to the GT mix energy envelope."
        ),
    )

    partition_readout: Literal["probe", "decoder"] = Field(
        default="probe",
        description=(
            "How stem energies are read off x0-hat: 'probe' = frozen closed-form ridge probe "
            "(build_mix_energy.py; ~zero cost). 'decoder' = true VAE decode (pre-registered "
            "fidelity ablation; NOT implemented in this pass -- fails fast if selected)."
        ),
    )

    partition_probe_path: str = Field(
        default="",
        description=(
            "Path to energy_probe.pt (w, b, mu, sd) written by build_mix_energy.py. Required "
            "when the partition or mix-span-energy loss is active with probe readout."
        ),
    )

    mix_energy_dir: str = Field(
        default="mix_energy_25hz",
        description=(
            "Data-source dir (under preprocessed_data_root) holding per-segment GT mix energy "
            "envelopes {energy: [T]}, built by build_mix_energy.py. Loaded only when the "
            "partition or mix-span-energy loss is active."
        ),
    )

    mix_span_energy_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "lambda_mixE of the mix-span ENERGY tie (arm D2; SS3.4 shortcut mitigation): "
            "sigma-weighted log-domain loss tying the PREDICTED mix span's energy envelope to "
            "the GT mix energy (1 scalar per 40 ms -- never full-content supervision). "
            "Requires include_mix."
        ),
    )

    per_source_energy_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "lambda_ps of the PER-SOURCE energy loss (arm E family): sigma-weighted log-domain "
            "loss tying EACH stem's probe-read energy envelope to that stem's OWN GT energy "
            "(build_stem_energy.py). The anti-absorption fix for the sum-only partition's "
            "limit cycle: a stem carrying the sibling's voice in its GT-silent half pays "
            "directly. 0 = off (prior arms byte-identical)."
        ),
    )

    stem_energy_dir: str = Field(
        default="stem_energy_25hz",
        description=(
            "Data-source dir (under preprocessed_data_root) holding per-segment GT PER-STEM "
            "energy envelopes {energy: [K, T]}, built by build_stem_energy.py. Loaded only "
            "when the per-source energy loss is active."
        ),
    )

    per_source_energy_weight_urmp: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "G-fix-domain: per-sample OVERRIDE of per_source_energy_weight for URMP-origin "
            "segments only (dialogue keeps per_source_energy_weight unchanged). Root cause: the "
            "frozen ridge probe this loss reads is fit ONLY on dialogue (energy_probe.pt, "
            "R^2=0.99 on dialogue) and is confirmed (.scripts/probe_cross_domain_bias.py, "
            "2026-07-28) to misread real URMP music by a mean bias of -3.07 log-units (R^2 "
            "-1.24 on music) -- a systematic, directional error, not noise. None (default) "
            "keeps every prior arm byte-identical: the loss applies uniformly at "
            "per_source_energy_weight, exactly as before this field existed. Requires "
            "domain_flag_dir to resolve which segments are URMP."
        ),
    )

    domain_flag_dir: str = Field(
        default="domain_flag",
        description=(
            "Data-source dir holding one scalar {is_urmp: float 0./1.} per segment, read ONLY "
            "when per_source_energy_weight_urmp is set. Segment domain is fixed at dataset-build "
            "time (name-prefix convention: dialogue = celebvhq_*, URMP = digit-prefixed piece "
            "names) -- never inferred at train time."
        ),
    )

    stem_latents_dir_prefix: str = Field(default="audio_latents_stem")
    stem_audio_conditions_dir_prefix: str = Field(
        default="conditions_stem_audio",
        description="Per-stem AUDIO captions WITH the appended audio-discipline sentence (a NEW "
                    "condition dir; never overwrites the historical conditions_stem{k}).")
    mix_latents_dir: str = Field(default="audio_latents_mix")
    scene_audio_conditions_dir: str = Field(
        default="conditions_scene_audio",
        description="Scene AUDIO caption (mix span text) WITH the appended sentence; only loaded "
                    "when include_mix is set.")

    selfboot_mix_p: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Arm I (SELF-BOOTSTRAPPING; JEN-1 Composer Sec. 4.3 / Self Forcing): probability, per "
            "sample per step, that the mix span's CLEAN latent is the cached MODEL-GENERATED mix "
            "instead of the ground-truth mix. Fixes exposure bias -- the stems currently learn to "
            "read content out of a clean GT mix that does not exist at inference "
            "(analysis/mixleads_audit.md BUG 1). Adds NO loss on the mix span and NO new inference "
            "input; the cache is training-only. 0.0 = off (every prior arm byte-identical, "
            "including the RNG stream). Requires include_mix, mix_loss_weight == 0 and "
            "mix_span_energy_weight == 0."
        ),
    )

    selfboot_mix_dir: str = Field(
        default="audio_latents_mix_selfboot",
        description=(
            "Data-source dir (under preprocessed_data_root) holding the cached generated mix "
            "latents, one <segment>.pt per TRAIN segment in the byte-same layout as "
            "mix_latents_dir. Built by scripts/build_selfboot_mix_cache.py. Loaded only when "
            "selfboot_mix_p > 0."
        ),
    )

    mix_consistency_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Arm CONS (decode-domain mixture-consistency loss): weight on the DECODE-domain "
            "log-mel loss tying the stems' summed LINEAR power to the mix span's, computed as "
            "L1(logsumexp([mel_stem_0, ..., mel_stem_{K-1}, log_c]), mel_mix) after decoding "
            "EACH span's x0-hat through the true (frozen) audio VAE decoder -- never in latent "
            "space (the audio VAE encoder is confirmed NON-additive by direct measurement, "
            "analysis/urmp_stem_collapse_phase0.md's encoder-linearity probe). log_c is a small "
            "per-segment floor (median of the mix's own decoded linear power, times 1e-3, the "
            "same recipe _compute_partition_losses uses) so the logsumexp never sees log(0). "
            "0.0 = off (default): every prior arm byte-identical, and strategy._audio_decoder -- "
            "which does not exist until train_jointstem.py assigns it from trainer._audio_vae -- "
            "is never touched. Requires include_mix (there is no mix span to compare against)."
        ),
    )

    stem_caption_dropout_p: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Arm B2 (video grounding), lever 1: probability of replacing ONE stem's caption with "
            "the null caption (`null_caption_path`) for a training sample. WHY: the stems' loss "
            "is satisfiable from caption + own noisy latent alone, so the video pathway is an "
            "OPTIONAL channel gradient descent lets atrophy -- the measured symptom being stems "
            "whose speech is unrelated to the lips (stem<->mix envelope correlation +0.47 in the "
            "single-pass regime vs +0.97 committed and +0.98 for ground truth). With no caption "
            "to identify its source, the span must read the video (and the mix/sibling) instead. "
            "At most ONE stem is dropped per sample, so the task stays well posed: the other "
            "caption still names its source and the mix still defines the total. 0.0 = off "
            "(every prior arm byte-identical: no extra RNG draw is taken)."
        ),
    )

    null_caption_path: str | None = Field(
        default=None,
        description=(
            "Feature-level blob substituted by `stem_caption_dropout_p`, built by "
            "scripts/build_null_caption.py from the composer's generic prompt ('One source alone "
            "- a single isolated source in the scene, with every other source removed'). It "
            "states the ROLE without naming the source, so the sample stays in distribution "
            "while the identifying information is gone. Required when the dropout is on."
        ),
    )

    visual_grounding_weight: float = Field(
        default=0.0, ge=0.0,
        description=(
            "Arm B2, lever 2: weight on the per-stem VIDEO GROUNDING loss -- an InfoNCE over "
            "temporal shifts tying each stem's energy envelope to the MOTION inside its own "
            "region of the frame ('the lips tell you who talks'). Regions come from the "
            "`stem_boxes` source (scripts/build_stem_boxes.py: the composer's own panel "
            "placements / trajectories, or block halves; all 2,397 v6.gen4 segments covered, "
            "verified by overlay). TRAINING-ONLY -- the boxes are never an inference input, so "
            "the no-masks directive holds: the model learns to find its source. The loss scores "
            "alignment at lag 0 against shifted alternatives, so it penalizes exactly the "
            "observed failure (right content, wrong time) rather than requiring a particular "
            "functional relationship between audio and motion. 0.0 = off (default)."
        ),
    )

    visual_grounding_min_motion_std: float = Field(
        default=0.0, ge=0.0,
        description=(
            "Minimum standard deviation of a region's LOG-motion envelope for the grounding "
            "loss to apply to that stem. WHY THIS EXISTS (arm B2 failure, 2026-08-05): the loss "
            "z-scores the motion envelope, so a region with no motion structure is divided by a "
            "near-zero sd and becomes pure noise presented as a confident target. Measured on "
            "the roster, the log-motion sd is 0.677 for dialogue faces but 0.257 for music, and "
            "76% of MUSIC stems are below 0.3 (a bowing arm barely moves inside a block region) "
            "versus 2% of dialogue stems. B2 ran with an effective threshold of 1e-12 -- i.e. no "
            "gate -- and its music mixes drifted 13 dB down over 1250 steps while dialogue "
            "stayed within 2 dB of ground truth. 0.3 keeps the cue where the video actually "
            "carries timing and drops it where it does not. 0.0 = no gate (B2's behaviour)."
        ),
    )

    visual_grounding_max_shift: int = Field(
        default=4, ge=1,
        description=(
            "Number of temporal shifts each way used as InfoNCE negatives for the grounding "
            "loss, in VIDEO LATENT frames (~301 ms each -- the video latent runs at 3.32 Hz "
            "against the audio's 25.2 Hz, and that gap is the hard ceiling on how finely this "
            "pathway can resolve timing). 4 covers +-1.2 s, comfortably wider than a dialogue "
            "turn boundary."
        ),
    )

    stem_boxes_dir: str = Field(
        default="stem_boxes",
        description="Directory of per-stem region boxes; read when visual_grounding_weight>0 or "
                    "positional_spatial_mask is set.",
    )

    positional_spatial_mask: bool = Field(
        default=False,
        description=(
            "ARM POSMASK. When a sample draws the POSITIONAL text register, restrict that sample's "
            "stem spans to attend only the video tokens inside their own box, via an additive "
            "pre-softmax bias on `video_to_audio_attn` (stemgen/spatial_mask_gate.py). WHY: the "
            "positional prompt already NAMES a location ('the instrument on the left'), but "
            "nothing makes the audio query read that part of the frame -- the model is free to "
            "satisfy the caption from global context and never bind the words to the pixels. This "
            "makes the geometry the prompt asserts an actual constraint on where the stem may "
            "look. Applies ONLY on positional draws: a semantic or short draw names no location, "
            "so masking there would be inventing a claim the text does not make, and those samples "
            "get an all-zero bias row (a no-op -- softmax is shift-invariant). A stem whose box is "
            "`valid=False` is also left ungated rather than masked to nothing. The mix span is "
            "never gated: it legitimately hears the whole frame. False = the holder is never "
            "touched and every prior arm is byte-identical."
        ),
    )

    positional_spatial_mask_eps: float = Field(
        default=1e-4, gt=0.0, lt=1.0,
        description=(
            "Floor inside log(clamp(mask, eps, 1)) for the positional mask, i.e. how hard the "
            "outside-the-box suppression is: log(1e-4) = -9.2 logits below an in-box key. Not a "
            "hard mask on purpose -- the box is a rectangle on a 16x24 latent grid, so its edge is "
            "approximate, and an infinite penalty would make a slightly-wrong box unrecoverable."
        ),
    )

    mix_consistency_detach_mix: bool = Field(
        default=False,
        description=(
            "Arm CONS direction. False (default) = the historical two-way loss: gradient flows "
            "into the stems AND into the mix span, so the constraint can be satisfied by pulling "
            "the mix DOWN toward a generic stem sum -- harmless when the mix is frozen (arm N) "
            "but actively dangerous once the mix is trained, where the mix is itself a product. "
            "True = detach the mix's decoded log-mel (and the floor derived from it), making CONS "
            "a ONE-DIRECTIONAL obligation: the stems must add up to the mix, never the reverse. "
            "On sep-pattern samples the mix x0-hat is the clean GT (sigma 0 -> no gradient path "
            "anyway), so True simply extends that same anchor semantics to the noisy samples. "
            "User decision 2026-08-04 for the pure-generation arm."
        ),
    )

    mix_consistency_power: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Exponent p for combining the stems inside arm CONS: "
            "(1/p) * logsumexp(p * [mel_stem_0, ..., mel_stem_{K-1}, log_c]). The comparison is "
            "always PHASE-FREE -- the audio VAE's mel is log-MAGNITUDE "
            "(torchaudio MelSpectrogram(power=1.0) then log(), ltx_core audio_vae/ops.py) and the "
            "vocoder invents phase downstream, so nothing here can or should depend on it. "
            "p = 1.0 (default, and the setting the pure-generation arm ships with per the user's "
            "2026-08-04 directive to compare MAGNITUDES) adds the stems' magnitudes and is "
            "bit-for-bit the historical CONS combination. p = 2.0 would instead add POWERS "
            "(|M|^2 = |S_0|^2 + |S_1|^2), the incoherent-source rule. The knob exists so the rule "
            "can be CALIBRATED against ground truth (scripts/check_mix_sum_rule.py) rather than "
            "assumed -- not because 2.0 is expected."
        ),
    )

    stem_decorrelation_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Arm DECORR (stem-pair decorrelation loss): weight on a covariance/decorrelation "
            "term between stem0 and stem1's DECODED log-mel spectrograms, penalizing when the "
            "two stems carry too-similar (duplicated) content -- the measured winner-take-all / "
            "near-identical-duplicate failure that adherence-only losses (CONS) cannot see, since "
            "two duplicate half-mix stems still sum back to the mix correctly. Computed as the "
            "differentiable Pearson correlation between each stem's flattened decoded mel (mean-"
            "centered, dot product over norms, same op as analyze_sum_final.py's numpy corr()), "
            "then loss = relu(corr)^2 -- ONLY positive correlation (content similarity) is "
            "penalized; negative correlation is neither penalized nor rewarded, since this is "
            "music where both stems may legitimately be simultaneously ACTIVE (same timing) "
            "without being the SAME content -- activity overlap is not the failure mode, content "
            "duplication is. Does NOT require include_mix (the comparison is stem0-vs-stem1 only, "
            "never touches the mix span) and does not require CONS to be active -- keeps its own "
            "same-step stash (self._decorr_state) and decode call, independent of "
            "mix_consistency_weight. 0.0 = off (default): every prior arm byte-identical, and "
            "strategy._audio_decoder -- shared with CONS but assigned by train_jointstem.py under "
            "an OR'd gate -- is never touched. Requires num_stems == 2 (raises NotImplementedError "
            "otherwise; the pairwise formula does not generalize to K > 2 stems without deciding "
            "which pairs to penalize)."
        ),
    )

    mix_channel_conditioning: bool = Field(
        default=False,
        description=(
            "Arm CHAN (mix-channel conditioning): make the mix span an UNAVOIDABLE, "
            "train+inference-symmetric conditioning channel by injecting it directly into every "
            "stem span's INPUT token -- stem_input[t] += W(mix_token[t]), W a zero-initialized "
            "nn.Linear (stemgen/mix_channel_adapter.py, MixChannelAdapter) so the arm is an EXACT "
            "no-op at construction. NOT wired inside prepare_training_inputs (that would only "
            "reach the training forward, not validation/generation, which never calls this "
            "function): train_jointstem.py installs the adapter as a wrap on the ONE shared "
            "audio_patchify_proj submodule instead, the same pattern arms H/K/N/A2V use to apply "
            "identically to training and every sampling pass. False (default) means the gate is "
            "never armed -- arms A-M byte-identical. Requires include_mix."
        ),
    )

    rollout_training: bool = Field(
        default=False,
        description=(
            "Arm ROLL (Self-Forcing-lite training rollout; Self Forcing arXiv 2506.08009's core "
            "trick, one rung up from arm I's mix-only self-bootstrap and arm IP's one-shot input "
            "perturbation): each training step runs `rollout_steps - 1` forward passes under "
            "torch.no_grad(), each stepping the STEM spans (and the shared-sigma video stream) "
            "one grid point closer to clean with the model's OWN prediction "
            "(per_token_euler_update), then a FINAL forward pass WITH gradients at the "
            "resulting self-generated (not GT-noised) state -- the only pass contributing to "
            "the backward graph. Trains the stems under their actual inference-time conditional "
            "(a self-generated history) instead of only the teacher-forced one. The velocity "
            "target v = eps - x0 is sigma-free, so the window-start audio_targets/"
            "audio_loss_mask stay the correct loss target at the rolled-to final sigma "
            "unchanged. The MIX span (when include_mix) is never advanced -- held at whatever "
            "prepare_training_inputs drew at the window's start (respecting mix_sigma_mode/ "
            "mix_lead_alpha/sep_pattern_p exactly), which is EXACT rather than approximate "
            "because the mix span already carries no loss and no LoRA gradient in every config "
            "this arm is built against (mix_loss_weight=0, frozen_mix_lora). Installed by "
            "train_jointstem.py (stemgen/rollout_step.py), replacing trainer._training_step -- "
            "NOT expressible inside prepare_training_inputs/compute_loss's one-forward-pass "
            "contract. False (default) = every prior arm byte-identical, _training_step "
            "untouched."
        ),
    )

    rollout_steps: int = Field(
        default=4,
        ge=2,
        description=(
            "K in arm ROLL: total forward passes per training step (K-1 no-grad rollout steps "
            "+ 1 gradient step). The K sigma grid points come from a fixed LTX2Scheduler grid "
            "(rollout_grid_steps + 1 points), with the window's LAST point snapped to a sigma "
            "freshly drawn from the real timestep sampler each step (not fixed to the "
            "schedule's tail) so the gradient step's sigma marginal matches ordinary "
            "(non-rollout) training's marginal -- otherwise every ROLL gradient step would land "
            "at one fixed sigma for the whole run, confounding a ROLL-vs-teacher-forced "
            "comparison with a training-sigma-coverage difference. Ignored when "
            "rollout_training=False."
        ),
    )

    rollout_grid_steps: int = Field(
        default=30,
        ge=2,
        description=(
            "Number of LTX2Scheduler grid intervals (grid_steps+1 sigma points, 1 down to 0) "
            "the arm-ROLL window is drawn from. Built with the default token-count anchor (no "
            "per-batch latent), so it differs slightly (fixed shift) from the true per-batch "
            "validation grid; acceptable since the window position itself is randomized per "
            "step, not fixed to specific indices. Ignored when rollout_training=False."
        ),
    )



    def _alt_suffixes(self) -> list[str]:
        """The NON-primary register suffixes this run draws from, in a fixed order."""
        if self.register_mix:
            return [s for s in sorted(self.register_mix) if s]
        sfx = self.positional_conditions_suffix
        return [sfx] if sfx and self.positional_prompt_p > 0.0 else []

    def get_data_sources(self) -> dict[str, str]:
        """Which preprocessed dirs this arm reads. Abstract on the CONFIG as of LTX-2
        v1.2.0 (it used to live on the strategy); the strategy delegates here so there
        is one definition. Load-bearing: PrecomputedDataset SILENTLY drops any segment
        missing from any listed source, so every entry is gated on what reads it."""
        sources = {"latents": "latents", "conditions": "conditions"}
        for k in range(self.num_stems):
            sources[f"{self.stem_latents_dir_prefix}{k}"] = f"audio_stem{k}"
            sources[f"{self.stem_audio_conditions_dir_prefix}{k}"] = f"conditions_stem_audio{k}"
        if self.include_mix:
            sources[self.mix_latents_dir] = "audio_mix"
            sources[self.scene_audio_conditions_dir] = "conditions_scene_audio"
        if self.selfboot_mix_p > 0.0:
            sources[self.selfboot_mix_dir] = "audio_mix_selfboot"
        if (self.partition_loss_weight > 0.0 or self.mix_span_energy_weight > 0.0
                or self.per_source_energy_weight > 0.0):
            sources[self.mix_energy_dir] = "mix_energy"
        if self.per_source_energy_weight > 0.0:
            sources[self.stem_energy_dir] = "stem_energy"
        if self.per_source_energy_weight_urmp is not None:
            sources[self.domain_flag_dir] = "domain_flag"
        if self.visual_grounding_weight > 0.0 or self.positional_spatial_mask:
            sources[self.stem_boxes_dir] = "stem_boxes"
        # SECOND TEXT REGISTER, registered only when the router is actually on, so a dataset that
        # does not carry the suffixed dirs is unaffected (PrecomputedDataset would otherwise drop
        # every sample for a missing source).
        if self.register_avail_dir and self.register_mix:
            sources[self.register_avail_dir] = "register_avail"
        for sfx in self._alt_suffixes():
            sources[f"conditions{sfx}"] = f"conditions{sfx}"
            for k in range(self.num_stems):
                sources[f"{self.stem_audio_conditions_dir_prefix}{k}{sfx}"] = \
                    f"conditions_stem_audio{k}{sfx}"
            if self.include_mix:
                sources[f"{self.scene_audio_conditions_dir}{sfx}"] = \
                    f"conditions_scene_audio{sfx}"
        return sources


class JointStemStrategy(TrainingStrategy):
    """Co-generate K speaker stems (+ optional mix) with per-channel block-masked text."""

    config: JointStemConfig

    def __init__(self, config: JointStemConfig):
        super().__init__(config)
        if config.mix_loss_weight > 0.0 and not config.include_mix:
            raise ValueError(
                "mix_loss_weight>0 requires include_mix=True: there is no mix span to supervise")
        if config.mix_span_energy_weight > 0.0 and not config.include_mix:
            raise ValueError(
                "mix_span_energy_weight>0 requires include_mix=True: there is no mix span "
                "whose energy envelope could be tied")
        if config.mix_consistency_weight > 0.0 and not config.include_mix:
            raise ValueError(
                "mix_consistency_weight>0 (arm CONS) requires include_mix=True: there is no "
                "mix span to decode-compare the summed stems against")
        if config.stem_decorrelation_weight > 0.0 and config.num_stems != 2:
            # Unlike every include_mix-gated arm above, DECORR deliberately has NO include_mix
            # requirement -- the comparison is stem0-vs-stem1 only, never the mix span. What it
            # DOES require is exactly two stems: the pairwise correlation formula has no defined
            # generalization to K > 2 (which pairs? averaged how?), so fail fast rather than
            # silently penalizing only the first two of K stems.
            raise NotImplementedError(
                f"stem_decorrelation_weight>0 (arm DECORR) only implements the pairwise "
                f"stem0-vs-stem1 correlation; num_stems={config.num_stems} != 2 would silently "
                "ignore any additional stems")
        if config.mix_lead_alpha < 1.0 and not config.include_mix:
            raise ValueError(
                "mix_lead_alpha<1 requires include_mix=True: there is no mix span to lead "
                "the stems' schedule")
        if config.mix_sigma_mode == "independent" and not config.include_mix:
            raise ValueError(
                "mix_sigma_mode='independent' (arm G) requires include_mix=True: there is no "
                "mix span whose sigma could be drawn independently")
        if config.sep_pattern_p > 0.0 and not config.include_mix:
            raise ValueError(
                "sep_pattern_p>0 (arm M) requires include_mix=True: there is no mix span to "
                "present as the observed track")
        if config.positional_spatial_mask:
            # Both of these would produce a run that trains cleanly and is simply the baseline arm
            # under a different name -- the failure mode a fraction in the log catches only after
            # the GPU-hours are spent. Catch them at construction instead.
            if not config.register_mix:
                raise ValueError(
                    "positional_spatial_mask requires register_mix: the mask is keyed on a sample "
                    "having drawn the POSITIONAL register, and without the router there is no draw")
            if float(config.register_mix.get("_pos", 0.0)) <= 0.0:
                raise ValueError(
                    f"positional_spatial_mask is on but register_mix gives '_pos' weight "
                    f"{config.register_mix.get('_pos', 0.0)} -- the positional register would "
                    "never be drawn and the mask would never fire")
        if config.span_pe_layout == "concat_mix_first":
            if not config.include_mix:
                raise ValueError(
                    "span_pe_layout='concat_mix_first' requires include_mix=True: the layout is "
                    "defined as [mix | stem0 | stem1] on the time axis, and without a mix span "
                    "there is no first slice to anchor it")
            if config.ref_time_offset != 0.0:
                raise ValueError(
                    f"span_pe_layout='concat_mix_first' and ref_time_offset="
                    f"{config.ref_time_offset} are two competing time-shift mechanisms on the "
                    "same axis; use one or the other")
        # Realized gate rates, reported every step once the arm is on. Initialized here so the
        # attribute exists before the first prepare_training_inputs, not conjured on first use.
        self._posmask_frac = 0.0
        self._posmask_sample_frac = 0.0
        self._span_pe_ranges_printed = False
        self._mix_sigma_mean = 0.0       # arm SEP03: realized mix sigma (post-clamp, pre-sep)
        self._mix_eq_stem_frac = 0.0     # arm SEP03: how often the clamp bit (mix == stem sigma)
        if config.a2v_mix_only and not config.include_mix:
            raise ValueError(
                "a2v_mix_only requires include_mix=True: there is no mix span for video to "
                "listen to")
        if config.frozen_mix_lora:
            if not config.include_mix:
                raise ValueError(
                    "frozen_mix_lora (arm N) requires include_mix=True: there is no mix span "
                    "whose computation could be frozen")
            if config.span_attention_topology != "mix_protected":
                raise ValueError(
                    "frozen_mix_lora requires span_attention_topology='mix_protected': frozen "
                    "weights reading LoRA-influenced stem activations would silently break the "
                    "checkpoint-0 guarantee the arm exists to provide")
            if config.mix_loss_weight > 0.0:
                raise ValueError(
                    "frozen_mix_lora with mix_loss_weight>0 is dead compute: the mix pathway "
                    "carries no LoRA parameters, so its loss has zero gradient")
        if config.selfboot_mix_p > 0.0:
            if not config.include_mix:
                raise ValueError(
                    "selfboot_mix_p>0 (arm I) requires include_mix=True: there is no mix span to "
                    "replace with a generated one")
            if config.mix_loss_weight > 0.0:
                raise ValueError(
                    "selfboot_mix_p>0 with mix_loss_weight>0 would supervise the mix span against "
                    "a mix the model generated itself (self-distillation collapse), and adds mix "
                    "CONTENT supervision -- forbidden by design")
            if config.mix_span_energy_weight > 0.0:
                raise ValueError(
                    "selfboot_mix_p>0 with mix_span_energy_weight>0 ties the GENERATED mix span's "
                    "energy envelope to the GT mix envelope -- that label no longer describes the "
                    "span's content")
        if config.per_source_energy_weight_urmp is not None and config.per_source_energy_weight <= 0.0:
            raise ValueError(
                "per_source_energy_weight_urmp is set but per_source_energy_weight is 0 -- there "
                "is no dialogue-side loss for the URMP value to override; set "
                "per_source_energy_weight_urmp=0.0 has no effect if the base weight is already 0")
        if config.span_axis_pairs > 0:
            num_spans = config.num_stems + int(config.include_mix)
            if len(config.span_axis_coords) != num_spans:
                raise ValueError(
                    f"span_axis_coords has {len(config.span_axis_coords)} entries but the audio "
                    f"sequence has {num_spans} spans (num_stems={config.num_stems}, "
                    f"include_mix={config.include_mix}) -- one coordinate per span, in sequence "
                    "order [stem0, stem1 (, mix)]")
        if config.mix_channel_conditioning and not config.include_mix:
            raise ValueError(
                "mix_channel_conditioning (arm CHAN) requires include_mix=True: there is no mix "
                "span to inject into the stems' input tokens")
        if config.rollout_training and (
                config.partition_loss_weight > 0.0 or config.mix_span_energy_weight > 0.0
                or config.per_source_energy_weight > 0.0):
            raise ValueError(
                "rollout_training=True (arm ROLL) is incompatible with the partition/mix-span-"
                "energy/per-source-energy losses: they read self._partition_state, stashed by "
                "prepare_training_inputs from the WINDOW-START noisy_spans/span_sigmas, which is "
                "stale once the rollout has advanced audio_pred to a LATER (rolled-to) sigma -- "
                "the x0-hat inversion `noisy - sigma*v_pred` would silently use the wrong "
                "noisy/sigma pair and corrupt the loss. Not implemented; set these weights to 0.")
        # NOTE: mix_consistency_weight (arm CONS) is deliberately ABSENT from these guards as of
        # 2026-07-30 -- see refresh_cons_state_after_rollout(). The combination is now correct by
        # construction AND fails loud if the refresh is ever skipped, so it does not need a ban.
        if config.rollout_training and config.stem_decorrelation_weight > 0.0:
            # Same staleness hazard as the partition family above: _decorr_state is stashed by
            # prepare_training_inputs from the WINDOW-START noisy_spans/span_sigmas, which the
            # rollout leaves behind once audio_pred has advanced to a LATER (rolled-to) sigma --
            # the x0-hat inversion `noisy - sigma*v_pred` would silently use the wrong noisy/sigma
            # pair. (mix_consistency_weight/arm CONS had the identical exposure through
            # _cons_state; FIXED 2026-07-30 -- rollout_step.py now calls
            # strategy.refresh_cons_state_after_rollout() before compute_loss, and the CONS loss
            # REFUSES to run on a window-start stash, so the combination cannot silently corrupt.
            # DECORR is fixable the same way and stays guarded only because nothing has tested it.)
            raise ValueError(
                "rollout_training=True (arm ROLL) is incompatible with stem_decorrelation_weight"
                ">0 (arm DECORR): it reads self._decorr_state, stashed by prepare_training_inputs "
                "from the WINDOW-START noisy_spans/span_sigmas, which is stale once the rollout "
                "has advanced audio_pred to a LATER (rolled-to) sigma. Not implemented; set "
                "stem_decorrelation_weight to 0.")

        # Arm CHAN: stash point only, populated by train_jointstem.py AFTER the trainer/transformer
        # exist (the adapter needs a real device to live on) -- NEVER read inside
        # prepare_training_inputs; the actual injection is wired onto the transformer's own
        # audio_patchify_proj submodule (stemgen/mix_channel_adapter.py) so it fires identically on
        # the training forward and every validation/sampling forward. None whenever
        # mix_channel_conditioning is False.
        self._mix_channel_adapter: Any = None

        # Arm K: span-level allow matrix [S, S] for the audio self-attention, or None for the
        # legacy "full" topology (None => Modality.attention_mask stays None => the UNMASKED
        # kernel, byte-identical to arms A-H). Built ONCE here, in the strategy that owns the
        # config, and handed to the sampler by train_jointstem.py so validation can never disagree
        # with training about the topology. Fails fast on 'mix_protected' without include_mix.
        self._span_allow = build_span_allow_matrix(
            num_stems=config.num_stems,
            include_mix=config.include_mix,
            topology=config.span_attention_topology,
            allow_sibling=config.span_attention_sibling,
        )

        # Same-step stash for the partition loss (pattern: iclora_mix.py _mix_loss_state).
        self._partition_state: dict[str, Any] | None = None
        # Arm CONS: a SEPARATE same-step stash, independently toggleable from the partition family
        # above (different readout -- true decode vs. the frozen probe -- and a different gate).
        self._cons_state: dict[str, Any] | None = None
        # Arm DECORR: its OWN same-step stash, independent of both the partition family AND CONS
        # above -- DECORR never needs a mix span, so it must not be coupled to _cons_state's
        # include_mix-gated lifecycle.
        self._decorr_state: dict[str, Any] | None = None
        # Shared by arms CONS and DECORR: the audio VAE decoder handle. None until
        # train_jointstem.py assigns trainer._audio_vae here (gated behind
        # mix_consistency_weight > 0 OR stem_decorrelation_weight > 0 in that script), so the
        # attribute always exists but is never touched by any arm that doesn't use it.
        self._audio_decoder = None
        # Arm B2: its OWN same-step stash (video latent + per-stem boxes) and the null caption
        # the dropout substitutes -- both independent of every stash above, so grounding and
        # caption dropout stay separately toggleable.
        self._ground_state: dict[str, Any] | None = None
        self._caption_dropout_frac = 0.0
        self._null_caption: dict[str, Tensor] | None = None
        if config.stem_caption_dropout_p > 0.0:
            if not config.null_caption_path:
                raise ValueError(
                    "stem_caption_dropout_p > 0 requires null_caption_path "
                    "(build it with scripts/build_null_caption.py)")
            blob = torch.load(config.null_caption_path, map_location="cpu", weights_only=False)
            self._null_caption = {"embeds": blob["audio_prompt_embeds"],
                                  "mask": blob["audio_prompt_attention_mask"]}
            print(f"[jointstem] ARM B2 stem-caption dropout p={config.stem_caption_dropout_p} "
                  f"-- one stem's caption per sample replaced by: "
                  f"{str(blob['prompt'])[:70]!r}", flush=True)
        self.last_loss_components: dict[str, float] = {}
        self._loss_call_count = 0
        self._selfboot_frac = 0.0        # arm I: realized fraction of swapped mix spans, last step
        self._sep_pattern_frac = 0.0     # arm M: realized fraction of separation-pattern samples

        if self._partition_active:
            if config.partition_readout == "decoder":
                raise NotImplementedError(
                    "partition_readout='decoder' (true VAE decode) is a pre-registered "
                    "ablation not implemented in this pass -- use 'probe'")
            if not config.partition_probe_path:
                raise ValueError("partition/mix-energy loss active but partition_probe_path "
                                 "is empty -- run build_mix_energy.py and point to its output")
            probe = torch.load(config.partition_probe_path, map_location="cpu",
                               weights_only=True)
            self._probe_w = probe["w"].float()                       # [3*D_token]
            self._probe_b = float(probe["b"])
            self._probe_mu = probe["mu"].float()
            self._probe_sd = probe["sd"].float()
            print(f"[jointstem] energy probe loaded from {config.partition_probe_path} "
                  f"(dim {self._probe_w.numel()}, fit held-out R^2 "
                  f"{float(probe['r2_heldout']):.4f})", flush=True)

    @property
    def span_allow_matrix(self) -> Tensor | None:
        """Arm K span-level allow matrix ``[S, S]`` (None under the legacy 'full' topology).

        The SINGLE source of truth for the topology: ``prepare_training_inputs`` expands it into
        ``Modality.attention_mask``, and ``train_jointstem.py`` hands this same object to
        ``JointStemSampler.joint_span_allow`` so the validation gate masks exactly what training
        masked. Never re-derive it at the sampling site.
        """
        return self._span_allow

    @property
    def _partition_active(self) -> bool:
        """Any probe-based energy loss active (partition, mix-span tie, or per-source).

        All three share the frozen probe, the GT mix energy (the c floor), and the
        same-step stash -- so one gate loads them.
        """
        return (self.config.partition_loss_weight > 0.0
                or self.config.mix_span_energy_weight > 0.0
                or self.config.per_source_energy_weight > 0.0)

    @property
    def requires_audio(self) -> bool:
        return True

    def _route_prompt_register(self, batch: dict, batch_size: int, device) -> None:
        """Swap the text conditions for the alternate register on a per-sample coin flip, in place.

        ALL FOUR text tensors switch together -- video, scene/mix and both stems -- because a
        sample whose video says one thing and whose stems say another re-creates the video/mix text
        split v7.gen was built to remove. Doing it once here, rather than at each read site, makes
        that a property of the code instead of a thing to remember.

        `positional_prompt_p = 0` takes NO RNG draw, so every prior arm keeps its exact random
        stream -- the same discipline as `sep_pattern_p` and `stem_caption_dropout_p`.

        THE DRAW IS PER VISIT, not per segment: it happens here, on every forward pass, so the same
        clip revisited in a later epoch can take a different register. That is what makes the extra
        registers an AUGMENTATION rather than a fixed three-way partition of the roster -- and the
        difference is invisible in the loss curves, so it is verified directly by
        `scripts/test_register_is_runtime.py` (one segment, 40 fresh loads, all three registers
        drawn).

        ONLY TENSOR FIELDS ARE SWAPPED. A condition blob also carries its `prompt` STRING, and that
        string keeps the PRIMARY register's text whatever was routed. Nothing in the strategy or
        the trainer reads it (they consume `*_prompt_embeds` / `*_prompt_attention_mask`, which do
        switch), so this is inert for training -- but any future diagnostic that reads
        `batch[...]["prompt"]` to report which register fired would report 'semantic' every single
        time and look like a collapsed router. Read the embeddings, or `_register_frac`.
        """
        self._alt_register_frac = 0.0
        self._register_frac: dict[str, float] = {}
        self._register_avail_frac: dict[str, float] = {}
        # WHICH register each sample drew, for the consumers that need more than the aggregate
        # fractions above. Arm POSMASK gates per sample on this: the spatial mask may only apply
        # where the POSITIONAL text actually fired. Left None on every path that takes no draw, so
        # a consumer cannot mistake "no router" for "everyone drew the primary register".
        self._register_pick: Tensor | None = None
        self._register_order: list[str] | None = None
        alts = self._alt_suffixes()
        if not alts:
            return
        keys = ["conditions", *(f"conditions_stem_audio{k}" for k in range(self.config.num_stems))]
        if self.config.include_mix:
            keys.append("conditions_scene_audio")

        if self.config.register_mix:
            # N-WAY: one categorical draw per sample over ["", *alts], weights from the config.
            order = [""] + alts
            w = torch.tensor([float(self.config.register_mix.get(s, 0.0)) for s in order],
                             dtype=torch.float, device=device)
            if float(w.sum()) <= 0:
                raise RuntimeError(f"register_mix has no positive weight: "
                                   f"{self.config.register_mix}")
            wb = w.expand(batch_size, -1)
            # PER-SAMPLE AVAILABILITY. Zeroing a register's weight for the samples that cannot
            # express it is all the "reweighting" needs to be: multinomial normalises each row, so
            # a segment whose short register collides simply draws semantic-or-positional at 1/2
            # each instead of 1/3 each. The alternative -- dropping the segment from the roster --
            # is what this exists to avoid.
            av = batch.get("register_avail")
            if av is not None:
                a = av["avail"] if isinstance(av, dict) else av
                a = a.to(device=device, dtype=torch.float).view(batch_size, -1)
                if a.shape[1] != len(order):
                    raise RuntimeError(
                        f"register_avail has width {a.shape[1]} but register_mix declares "
                        f"{len(order)} registers {order} -- the availability vectors were built "
                        "for a different mixture and their columns would not line up")
                wb = wb * a
                if bool((wb.sum(dim=1) <= 0).any()):
                    raise RuntimeError(
                        "a sample has NO available register: its availability vector zeroes every "
                        "register that register_mix gives weight to. The primary register must be "
                        "available on every segment.")
                self._register_avail_frac = {(s or "semantic"): float(a[:, j].mean())
                                             for j, s in enumerate(order)}
            pick = torch.multinomial(wb, 1).squeeze(1)     # [B]
            self._register_pick, self._register_order = pick, list(order)
            self._register_frac = {(s or "semantic"): float((pick == j).float().mean())
                                   for j, s in enumerate(order)}
            self._alt_register_frac = float((pick > 0).float().mean())
            for key in keys:
                merged = dict(batch[key])
                for field, val in batch[key].items():
                    if not torch.is_tensor(val):
                        continue
                    out = val
                    for j, s in enumerate(order):
                        if j == 0:
                            continue
                        alt = batch.get(f"{key}{s}")
                        if alt is None:
                            raise RuntimeError(
                                f"register_mix names {s!r} but the batch has no '{key}{s}' -- "
                                "every suffix's condition dirs must live in the same root")
                        if field not in alt:
                            continue
                        sel = (pick == j).view(-1, *([1] * (val.dim() - 1)))
                        out = torch.where(sel, alt[field].to(val.dtype), out)
                    merged[field] = out
                batch[key] = merged
            return

        # Two-way legacy path, kept byte-identical for the arms configured with it.
        sfx = alts[0]
        use_alt = torch.rand(batch_size, device=device) < self.config.positional_prompt_p
        self._alt_register_frac = float(use_alt.float().mean())
        for key in keys:
            alt = batch.get(f"{key}{sfx}")
            if alt is None:
                raise RuntimeError(
                    f"positional_prompt_p > 0 but the batch has no '{key}{sfx}' -- the suffixed "
                    "condition dirs must live in the same preprocessed root")
            merged = dict(batch[key])
            for field, val in batch[key].items():
                if not torch.is_tensor(val) or field not in alt:
                    continue
                sel = use_alt.view(-1, *([1] * (val.dim() - 1)))
                merged[field] = torch.where(sel, alt[field].to(val.dtype), val)
            batch[key] = merged

    def get_data_sources(self) -> dict[str, str]:
        """v1.2.0 declares this abstract on the CONFIG; delegate so there is one definition."""
        return self.config.get_data_sources()

    def _alt_suffixes(self) -> list[str]:
        """Delegates: the suffix set is a pure function of config fields, and get_data_sources --
        abstract on the CONFIG in LTX-2 v1.2.0 -- needs it there too. One definition, both callers."""
        return self.config._alt_suffixes()          # noqa: SLF001

    # ------------------------------------------------------------------ inputs

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        latents = batch["latents"]
        video_latents = self._video_patchifier.patchify(latents["latents"])
        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()
        fps = latents["fps"][0].item() if latents.get("fps") is not None else DEFAULT_FPS

        batch_size = video_latents.shape[0]
        device, dtype = video_latents.device, video_latents.dtype

        # ---- PROMPT REGISTER ROUTER (dual-register training) -------------------------------------
        # Swap the text conditions for the alternate register on a per-sample coin flip, ONCE and
        # up front, so every downstream read -- the video context below, the stem block contexts
        # and the mix's scene context -- sees the same choice by construction. Routing them at
        # their individual read sites would let a future edit switch three of four and silently
        # recreate the video/mix text split.
        #
        # p = 0 takes no RNG draw, keeping every prior arm's random stream byte-identical -- the
        # same discipline as `sep_pattern_p` and `stem_caption_dropout_p`.
        self._route_prompt_register(batch, batch_size, device)

        # ---- video stream (joint_denoise, first-frame-conditioned; along for the ride) ----------
        target_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_latents.shape[1],
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )
        sigmas = timestep_sampler.sample_for(video_latents)
        # VIDEO PINNED CLEAN -- and ONLY the video. `sigmas` is the SHARED timestep: the stem spans
        # take it at :1266 (`span_sigmas = [sigmas.view(-1) for _ in stem_tokens]`) and the audio
        # Modality at :1362. Zeroing `sigmas` itself therefore trains every STEM at sigma 0 too,
        # which is not separation at all but a degenerate "predict the noise direction from a clean
        # latent" task. That bug shipped for ~25 minutes on 2026-08-12 and was caught from the loss
        # curve alone: the pinned run fell smoothly (151.8 -> 129.8, no variance) while the correct
        # unpinned run scattered between 38 and 160 as its sampled sigma moved. A separate
        # `video_sigmas` keeps the pin on the stream it is meant for.
        #
        # `video_sigmas is sigmas` when the flag is off, so an unpinned arm is bit-identical to
        # before this change -- no extra draw, no reordering.
        video_sigmas = torch.zeros_like(sigmas) if self.config.video_sigma_zero else sigmas
        sigmas_expanded = video_sigmas.view(-1, 1, 1)
        video_noise = torch.randn_like(video_latents)
        video_targets = video_noise - video_latents
        source_positions = self._get_video_positions(
            num_frames=num_frames, height=height, width=width, batch_size=batch_size,
            fps=fps, device=device,
        )
        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise
        noisy_video = torch.where(
            target_conditioning_mask.unsqueeze(-1), video_latents, noisy_video)
        video_timesteps = self._create_per_token_timesteps(
            target_conditioning_mask, video_sigmas.squeeze())
        video_loss_mask = ~target_conditioning_mask

        video_modality = Modality(
            enabled=True,
            sigma=video_sigmas,
            latent=noisy_video,
            timesteps=video_timesteps,
            positions=source_positions,
            context=batch["conditions"]["video_prompt_embeds"],
            context_mask=batch["conditions"]["prompt_attention_mask"],
        )

        # ---- audio stream: [stem0 | stem1 (| mix)] -- SHARED timestep, all spans denoised --------
        # Every span carries velocity noise at the SAME per-element sigma (standard shared-timestep
        # diffusion). Stems always bear a loss; the mix span bears one only in Arm B.
        stem_latents = torch.stack(
            [batch[f"audio_stem{k}"]["latents"] for k in range(self.config.num_stems)]
        )  # [K, B, C, T, F]
        stem_tokens = [self._audio_patchifier.patchify(stem_latents[k])
                       for k in range(self.config.num_stems)]

        span_latents: list[Tensor] = list(stem_tokens)
        if self.config.include_mix:
            mix_latents = batch["audio_mix"]["latents"]              # [B, C, T, F]
            if self.config.selfboot_mix_p > 0.0:
                # ARM I -- SELF-BOOTSTRAPPING: with probability p, this sample's mix span
                # conditions on the model's OWN cached generated mix instead of the ground truth
                # -- the same distribution the span holds at inference. Per SAMPLE and per STEP
                # (JEN-1 Composer Sec. 4.3's p = 0.5), so every segment is seen both ways and the
                # GT mix stays an anchor. The swap happens BEFORE noising, so the span keeps
                # training's exact (1 - s) * x0 + s * eps convention at its own sigma, and the
                # velocity target v = eps - x0 below picks the swapped tensor up automatically
                # (numerically inert: the mix span is masked out of the loss whenever
                # mix_loss_weight = 0, which the __init__ guard forces here).
                boot_latents = batch["audio_mix_selfboot"]["latents"]
                if (boot_latents.shape != mix_latents.shape
                        or boot_latents.dtype != mix_latents.dtype):
                    raise ValueError(
                        f"selfboot mix latent {tuple(boot_latents.shape)}/{boot_latents.dtype} != "
                        f"GT mix latent {tuple(mix_latents.shape)}/{mix_latents.dtype} -- rebuild "
                        "the cache with build_selfboot_mix_cache.py")
                use_boot = (torch.rand(batch_size, device=mix_latents.device)
                            < self.config.selfboot_mix_p)            # [B]
                mix_latents = torch.where(
                    use_boot.view(-1, 1, 1, 1), boot_latents, mix_latents)
                self._selfboot_frac = use_boot.float().mean().item()
            span_latents.append(self._audio_patchifier.patchify(mix_latents))

        span_lens = [tok.shape[1] for tok in span_latents]

        # Time-aligned audio positions per span; the mix span (last, when present) may be offset.
        span_positions: list[Tensor] = []
        for i, span_len in enumerate(span_lens):
            pos = self._get_audio_positions(
                num_time_steps=span_len, batch_size=batch_size, device=device)
            is_mix = self.config.include_mix and i == len(span_lens) - 1
            if is_mix and self.config.ref_time_offset != 0.0:
                pos = pos + self.config.ref_time_offset
            if self.config.span_pe_layout == "concat_mix_first":
                # ARM CATPE: virtual per-span time offsets -- one concatenated timeline, mix
                # first, SEQUENCE order untouched. D is derived from the positions the patchifier
                # actually produced (token pitch * span length), never hardcoded: the pitch is a
                # function of sample_rate/hop/downsample and this must keep being right if any of
                # those change. RoPE evaluates midpoints, so offsetting both [start, end) bounds
                # by a scalar shifts a token's phase exactly (same mechanics as ref_time_offset).
                # Pitch from tokens 1->2, NOT 0->1: the first token is causal-clamped (its start
                # is floored at 0), so the 0->1 midpoint gap is 0.025 s while the steady pitch is
                # 0.04 s -- deriving from token 0 would shrink every offset by 37%.
                if span_len < 3:
                    raise ValueError(f"span_len={span_len} too short to derive the token pitch")
                dt = float(pos[0, 0, 2].mean() - pos[0, 0, 1].mean())
                span_seconds = span_len * dt
                offset = 0.0 if is_mix else (i + 1) * span_seconds
                pos = pos + offset
            span_positions.append(pos)
        if self.config.span_pe_layout == "concat_mix_first" and not self._span_pe_ranges_printed:
            # Printed from the REALIZED tensors, once: the claim worth checking is what the model
            # actually sees, not what the formula intended.
            names = [f"stem{k}" for k in range(len(span_lens) - 1)] + ["mix"]
            spans = "  ".join(
                f"{n} {float(p[0, 0, 0, 0]):.2f}-{float(p[0, 0, -1, 1]):.2f}s"
                for n, p in zip(names, span_positions))
            print(f"[jointstem] ARM CATPE span PE ranges (sequence order): {spans}", flush=True)
            self._span_pe_ranges_printed = True

        # Per-span sigma: every stem span at the drawn (primary) sigma. The mix span (last, when
        # present) depends on mix_sigma_mode:
        #   * 'leads'       (arm F): sigma_mix = mix_lead_alpha * sigma -- a DETERMINISTIC fixed-
        #                            alpha map of the stem sigma.
        #   * 'independent' (arm G): sigma_mix drawn INDEPENDENTLY from the SAME timestep sampler
        #                            as the primary draw (a second i.i.d. sample_for on the SAME
        #                            tensor the primary draw used, so identical shifted-logit-normal
        #                            marginal), decoupled from the stem sigma. Every (sigma_stem,
        #                            sigma_mix) pair becomes in-distribution.
        # The velocity target v = eps - x0 is sigma-FREE, so each span stays a valid flow-matching
        # objective at its own sigma -- only the noising level and the per-token timesteps change.
        # In 'leads' mode with mix_lead_alpha = 1.0 this is byte-identical to the historical shared
        # schedule (same values, same randn order); 'leads' never draws a second sigma, so arms A-F
        # keep their exact RNG stream.
        span_sigmas: list[Tensor] = [sigmas.view(-1) for _ in stem_tokens]
        sep_flags: Tensor | None = None
        if self.config.include_mix:
            if self.config.mix_sigma_mode in ("independent", "independent_leq"):
                mix_sigma = timestep_sampler.sample_for(video_latents).view(-1)
                if self.config.mix_sigma_mode == "independent_leq":
                    # Arm SEP03: stems first, then the mix clamped to them -- the mix LEADS by
                    # construction. Same sampler, same RNG order as 'independent' (the clamp is
                    # applied after the identical second draw), so the two modes differ only in
                    # the pairs they expose, never in the random stream.
                    mix_sigma = torch.minimum(mix_sigma, sigmas.view(-1))
                    self._mix_sigma_mean = float(mix_sigma.mean())
                    self._mix_eq_stem_frac = float((mix_sigma == sigmas.view(-1)).float().mean())
            else:
                mix_sigma = self.config.mix_lead_alpha * sigmas.view(-1)
            # Arm M SEPARATION pattern (sep_pattern_p, MGE-LDM track-aware timesteps): per
            # sample, force sigma_mix = 0 -- the generic noising below then yields the CLEAN GT
            # mix tokens ((1-0)*x0 + 0*eps) and per-token timestep 0, i.e. an OBSERVED track.
            # The loss mask (below) removes the mix from the loss for exactly these samples, so
            # the pattern trains "stems given a committed mix" and nothing else. p = 0 skips the
            # Bernoulli draw entirely, keeping every prior arm's RNG stream byte-identical.
            if self.config.sep_pattern_p > 0.0:
                sep_flags = (torch.rand(mix_sigma.shape[0], device=mix_sigma.device)
                             < self.config.sep_pattern_p)             # [B]
                mix_sigma = torch.where(sep_flags, torch.zeros_like(mix_sigma), mix_sigma)
                self._sep_pattern_frac = sep_flags.float().mean().item()
            span_sigmas.append(mix_sigma)

        noisy_spans: list[Tensor] = []
        target_spans: list[Tensor] = []
        for tok, span_sigma in zip(span_latents, span_sigmas):
            noise = torch.randn_like(tok)
            span_sigma_expanded = span_sigma.view(-1, 1, 1)
            # Arm IP (DDPM-IP, arXiv:2301.11706): the network INPUT is noised with a perturbed
            # eps' = eps + gamma*xi, the velocity TARGET stays v = eps - x0 -- the input error
            # simulates the sampler's own prediction error (root-caused exposure bias). At
            # sigma = 0 the perturbation vanishes with the noise term itself (arm M clean mix).
            noise_in = noise
            if self.config.input_perturbation_gamma > 0.0:
                noise_in = noise + self.config.input_perturbation_gamma * torch.randn_like(tok)
            noisy_spans.append((1 - span_sigma_expanded) * tok + span_sigma_expanded * noise_in)
            target_spans.append(noise - tok)                        # velocity

        combined_latent = torch.cat(noisy_spans, dim=1)
        combined_positions = torch.cat(span_positions, dim=2)
        audio_targets = torch.cat(target_spans, dim=1)
        # Per-token timesteps [B, T_total]: each span's OWN sigma on its tokens -- this is how
        # the staggered schedule reaches the DiT (per-token AdaLN scale-shift on
        # Modality.timesteps). Modality.sigma below stays the STEM sigma (B,), matching the
        # iclora_mix precedent for independent mix timesteps (used only for the AV
        # cross-attention/prompt AdaLN, which take one scalar per batch element).
        audio_timesteps = torch.cat(
            [ss.view(-1, 1).expand(-1, tok.shape[1])
             for ss, tok in zip(span_sigmas, span_latents)],
            dim=1).contiguous()

        # ---- per-channel TEXT: concat the blocks + block-diagonal cross-attn mask ---------------
        block_contexts = [batch[f"conditions_stem_audio{k}"]["audio_prompt_embeds"]
                          for k in range(self.config.num_stems)]
        block_masks = [batch[f"conditions_stem_audio{k}"]["audio_prompt_attention_mask"]
                       for k in range(self.config.num_stems)]

        # ---- arm B2 lever 1: STEM CAPTION DROPOUT ------------------------------------------------
        # Replace ONE stem's caption (never both -- the task must stay well posed) with the null
        # caption, so that span cannot identify its source from text and has to read the video
        # and the mix instead. p = 0 takes no RNG draw at all, keeping prior arms byte-identical.
        self._caption_dropout_frac = 0.0
        if self.config.stem_caption_dropout_p > 0.0 and self.config.num_stems > 1:
            if self._null_caption is None:
                raise RuntimeError(
                    "stem_caption_dropout_p > 0 but no null caption was loaded -- set "
                    "joint_stem.null_caption_path (build it with scripts/build_null_caption.py)")
            drop = torch.rand(batch_size, device=device) < self.config.stem_caption_dropout_p
            which = torch.randint(self.config.num_stems, (batch_size,), device=device)
            null_e = self._null_caption["embeds"].to(device=device,
                                                     dtype=block_contexts[0].dtype)
            null_m = self._null_caption["mask"].to(device=device, dtype=block_masks[0].dtype)
            for k in range(self.config.num_stems):
                sel = (drop & (which == k)).view(-1, *([1] * (block_contexts[k].dim() - 1)))
                block_contexts[k] = torch.where(sel, null_e.expand_as(block_contexts[k]),
                                                block_contexts[k])
                selm = (drop & (which == k)).view(-1, *([1] * (block_masks[k].dim() - 1)))
                block_masks[k] = torch.where(selm, null_m.expand_as(block_masks[k]),
                                             block_masks[k])
            self._caption_dropout_frac = float(drop.float().mean())
        if self.config.include_mix:
            block_contexts.append(batch["conditions_scene_audio"]["audio_prompt_embeds"])
            block_masks.append(batch["conditions_scene_audio"]["audio_prompt_attention_mask"])

        audio_context = torch.cat(block_contexts, dim=1)            # [B, sum(S_i), D]
        audio_context_mask = build_block_diagonal_text_bias(
            span_lens=span_lens, block_masks=block_masks,
            dtype=audio_context.dtype, device=audio_context.device)  # [B,1,sum(T),sum(S)]

        # ---- arm K: asymmetric cross-span SELF-attention mask (native Modality route) ------------
        # None under the legacy "full" topology, in which case Modality.attention_mask stays None
        # and ltx_core takes the UNMASKED attention kernel -- byte-identical to arms A-H.
        audio_span_mask = build_span_allow_mask(
            span_lens=span_lens, allow=self._span_allow, batch=batch_size,
            dtype=combined_latent.dtype, device=device)              # [B, T, T] in {0, 1} or None

        audio_modality = Modality(
            enabled=True,
            latent=combined_latent,
            sigma=sigmas,
            timesteps=audio_timesteps,
            positions=combined_positions,
            context=audio_context,
            context_mask=audio_context_mask,
            attention_mask=audio_span_mask,
        )

        # Loss mask: every stem span True; the mix span True only when supervised (Arm B/R1) --
        # and NEVER on an arm-M separation-pattern sample, where the mix is an OBSERVED track
        # (clean context, not a target; supervising it there would just re-teach the identity).
        mix_supervised = self.config.include_mix and self.config.mix_loss_weight > 0.0
        loss_pieces = [torch.ones(batch_size, tok.shape[1], dtype=torch.bool, device=device)
                       for tok in stem_tokens]
        if self.config.include_mix:
            mix_row = torch.full((batch_size,), mix_supervised, dtype=torch.bool, device=device)
            if sep_flags is not None:
                mix_row = mix_row & ~sep_flags
            loss_pieces.append(mix_row.view(-1, 1).expand(-1, span_lens[-1]))
        audio_loss_mask = torch.cat(loss_pieces, dim=1)

        # ---- arm D family: stash the same-step tensors the partition loss needs -----------------
        # (noisy spans + sigma to invert v_pred -> x0-hat; GT mix energy as the coupling target).
        # Consumed by THIS step's compute_loss and cleared there -- never reused across steps.
        if self._partition_active:
            e_mix = batch["mix_energy"]["energy"]                    # [B, T] (25 Hz grid)
            for i, span_len in enumerate(span_lens):
                if e_mix.shape[1] != span_len:
                    raise ValueError(
                        f"mix energy length {e_mix.shape[1]} != span {i} token count "
                        f"{span_len} -- energy grid misaligned with the latent grid "
                        "(rebuild with build_mix_energy.py)")
            e_stem = None
            if self.config.per_source_energy_weight > 0.0:
                e_stem = batch["stem_energy"]["energy"]              # [B, K, T] (25 Hz grid)
                if (e_stem.shape[1] != self.config.num_stems
                        or e_stem.shape[2] != span_lens[0]):
                    raise ValueError(
                        f"stem energy shape {tuple(e_stem.shape)} != "
                        f"[B, {self.config.num_stems}, {span_lens[0]}] -- rebuild with "
                        "build_stem_energy.py")
            is_urmp = None
            if self.config.per_source_energy_weight_urmp is not None:
                is_urmp = batch["domain_flag"]["is_urmp"].float()       # [B], 1.0 = URMP segment
            self._partition_state = {
                "noisy_spans": noisy_spans,
                "span_sigmas": span_sigmas,
                "e_mix": e_mix,
                "e_stem": e_stem,
                "span_lens": span_lens,
                "is_urmp": is_urmp,
            }
        else:
            self._partition_state = None

        # ---- arm CONS: stash the same-step tensors the decode-domain consistency loss needs -----
        # (noisy spans + sigma to invert v_pred -> x0-hat per span). Kept as its OWN stash, separate
        # from _partition_state above, so the two mechanisms stay independently toggleable -- CONS
        # reads the TRUE VAE decode, not the frozen probe, and needs no GT energy data source at all.
        if self.config.mix_consistency_weight > 0.0:
            self._cons_state = {
                "noisy_spans": noisy_spans,
                "span_sigmas": span_sigmas,
                "span_lens": span_lens,
                # Provenance marker, NOT decoration. Under rollout_training the (noisy, sigma) pair
                # stashed here is the WINDOW-START state, while audio_pred will be the velocity at
                # the ROLLED-TO sigma -- inverting one against the other yields a wrong x0-hat and a
                # silently corrupt loss. rollout_step.py must call
                # refresh_cons_state_after_rollout() before compute_loss; this flag lets the loss
                # REFUSE to run otherwise instead of returning a plausible number.
                "sigma_source": "window_start",
            }
        else:
            self._cons_state = None

        # ---- arm DECORR: same-step tensors for the stem0-vs-stem1 decorrelation loss -------------
        # Independent stash from _cons_state above -- DECORR never reads the mix span, so its
        # lifecycle must not be coupled to include_mix / mix_consistency_weight.
        if self.config.stem_decorrelation_weight > 0.0:
            self._decorr_state = {
                "noisy_spans": noisy_spans,
                "span_sigmas": span_sigmas,
                "span_lens": span_lens,
            }
        else:
            self._decorr_state = None

        # ---- arm POSMASK: the positional prompt's location becomes an attention constraint -------
        # Set-or-clear UNCONDITIONALLY while the arm is on. The holder in spatial_mask_gate is
        # module-global and persists until overwritten, so a step that merely SKIPPED this would
        # silently reuse the previous step's bias -- and with one span layout throughout, a stale
        # bias shape-matches and applies without any error to say so.
        if self.config.positional_spatial_mask:
            set_video_key_bias(self._positional_key_bias(batch, latents["latents"], span_lens))
        # ---- arm B2 lever 2: stash what the video-grounding loss needs (own stash, as above) ----
        # The CLEAN video latent is the alignment target: a training-time reference (like CONS's
        # true decoder), never an inference input. The audio side is inverted from audio_pred in
        # compute_loss, so only the noising state has to be carried here.
        if self.config.visual_grounding_weight > 0.0:
            self._ground_state = {
                "noisy_spans": noisy_spans,
                "span_sigmas": span_sigmas,
                "span_lens": span_lens,
                "video_clean": latents["latents"],            # [B, C, T, H, W], pre-patchify
                "boxes": batch["stem_boxes"]["boxes"],        # [B, K, T, 4] normalized xyxy
                "boxes_valid": batch["stem_boxes"]["valid"],  # [B, K]
            }
        else:
            self._ground_state = None

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
        )

    def _positional_key_bias(self, batch: dict, video: Tensor, span_lens: list[int]) -> Tensor:
        """Per-audio-query video-key bias for arm POSMASK. [B, 1, T_audio, S_video].

        A stem span is gated only where BOTH hold: that sample drew the positional register, and
        that stem's box is marked valid. Everywhere else the mask is all-ones, whose log is 0 --
        and a bias that is constant across the video keys is a no-op, because softmax is
        shift-invariant. So "ungated" and "no gate at all" are the same computation, which is what
        makes one unconditional code path safe for every sample in the batch.

        The rasterisation deliberately reuses `_compute_visual_grounding_loss`'s convention (the
        half-open `x0 <= x < x1` test on cell-corner coordinates), so the mask and the grounding
        loss can never disagree about which latent cells are "inside" the same box.
        """
        boxes = batch["stem_boxes"]["boxes"].to(device=video.device).float()   # [B, K, T, 4]
        valid = batch["stem_boxes"]["valid"].to(device=video.device).bool()    # [B, K]
        b, _c, t_v, h, w = video.shape
        num_stems = self.config.num_stems
        if boxes.shape[1] < num_stems:
            raise ValueError(f"stem_boxes carries {boxes.shape[1]} boxes but the strategy has "
                             f"{num_stems} stems")
        if boxes.shape[2] != t_v:
            raise ValueError(f"stem_boxes has {boxes.shape[2]} frames but the video latent has "
                             f"{t_v} -- the box grid was built for a different geometry")

        pick, order = self._register_pick, self._register_order
        if pick is None or order is None:
            raise RuntimeError(
                "positional_spatial_mask needs the register router: no per-sample draw was "
                "recorded. Set register_mix (with a positional register) -- without it there is "
                "no positional draw to key the mask on.")
        if "_pos" not in order:
            raise RuntimeError(f"positional_spatial_mask is on but register_mix declares {order}, "
                               "which has no '_pos' register: the mask could never fire.")
        is_pos = (pick == order.index("_pos")).to(device=video.device)         # [B]

        ys = torch.arange(h, device=video.device).view(1, 1, 1, h, 1) / h
        xs = torch.arange(w, device=video.device).view(1, 1, 1, 1, w) / w
        x0, y0 = boxes[..., 0:1].unsqueeze(-1), boxes[..., 1:2].unsqueeze(-1)
        x1, y1 = boxes[..., 2:3].unsqueeze(-1), boxes[..., 3:4].unsqueeze(-1)
        region = ((xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)).float()     # [B, K, T, H, W]

        # A gated stem with an EMPTY region would suppress every video key equally, which is the
        # same no-op as ungated but arrives there by accident. Fall back explicitly instead, so an
        # unusable box is recorded as ungated rather than silently behaving like one.
        gate = is_pos.view(b, 1) & valid[:, :num_stems] & (region.flatten(2).sum(-1) > 0)  # [B, K]
        self._posmask_frac = float(gate.float().mean())
        self._posmask_sample_frac = float(gate.any(dim=1).float().mean())

        masks: list[Tensor | None] = []
        for k in range(num_stems):
            g = gate[:, k].view(b, 1, 1, 1).float()
            masks.append(g * region[:, k] + (1.0 - g))                        # [B, T, H, W]
        if self.config.include_mix:
            masks.append(None)                       # the mix legitimately hears the whole frame
        if len(masks) != len(span_lens):
            raise ValueError(f"built {len(masks)} span masks for {len(span_lens)} audio spans")
        return spatial_masks_to_per_query_bias(
            masks, span_lens, eps=self.config.positional_spatial_mask_eps)

    # ------------------------------------------------------------------- loss

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Masked velocity loss over the loss-bearing audio spans (+ optional video). Returns [B,]."""
        if audio_pred is None or inputs.audio_targets is None:
            raise ValueError("joint stem generation requires audio predictions")

        # Per-token loss weight over [stem0|stem1(|mix)]: every stem token weight 1; the mix span
        # (present only when include_mix) weighted by mix_loss_weight. The loss MASK already zeroes
        # the mix in Arm C (mix_loss_weight=0), so scaling by the weight is exact for A/B/C alike.
        weight = inputs.audio_loss_mask.float().unsqueeze(-1)        # [B, T_total, 1]
        if self.config.include_mix and self.config.mix_loss_weight != 1.0:
            # All spans share one latent length, so the trailing mix span is T_total/(num_stems+1).
            t_total = audio_pred.shape[1]
            mix_len = t_total // (self.config.num_stems + 1)
            weight = weight.clone()
            weight[:, t_total - mix_len:, :] *= self.config.mix_loss_weight
        err = (audio_pred - inputs.audio_targets).pow(2).mul(weight)
        audio_loss = err.sum(dim=[-2, -1]) / weight.sum(dim=[-2, -1]).clamp(min=1e-8)

        self.last_loss_components = {"loss_velocity": audio_loss.detach().mean().item()}
        if self.config.selfboot_mix_p > 0.0:
            self.last_loss_components["selfboot_frac"] = self._selfboot_frac
        if self.config.sep_pattern_p > 0.0:
            self.last_loss_components["sep_pattern_frac"] = self._sep_pattern_frac
        if self.config.mix_sigma_mode == "independent_leq":
            # The arm's identity is "mix at or below the stems"; a run where the clamp never
            # bites (or always does) is a different experiment wearing the same name.
            self.last_loss_components["mix_sigma_mean"] = self._mix_sigma_mean
            self.last_loss_components["mix_eq_stem_frac"] = self._mix_eq_stem_frac
        if self.config.stem_caption_dropout_p > 0.0:
            # Realized dropout rate: with batch size 1 this reads 0 or 1 per step and should
            # average to stem_caption_dropout_p over the run -- log it so a silently inert
            # lever (the failure mode that hides best) is visible from step one.
            self.last_loss_components["caption_dropout_frac"] = self._caption_dropout_frac
        if self.config.positional_spatial_mask:
            # How often the mask actually FIRED. Same argument as every fraction above, and it
            # bites harder here: the arm's whole identity is this gate, and a run where it never
            # fired -- a register that is never drawn, boxes that are all invalid, an availability
            # vector that zeroes the positional column -- produces a perfectly healthy loss curve
            # that is simply the baseline arm under a different name.
            self.last_loss_components["posmask_stem_frac"] = self._posmask_frac
            self.last_loss_components["posmask_sample_frac"] = self._posmask_sample_frac
        if self.config.register_mix:
            # The realized split per register. With batch size 1 each reads 0 or 1 per step and
            # must average to the configured weights -- without it a three-way mix that silently
            # collapsed to one register would look exactly like one that worked.
            for name, frac in (self._register_frac or {}).items():
                self.last_loss_components[f"reg_{name.lstrip('_')}"] = frac
            # And how often each register was even ELIGIBLE. Without this the drawn fractions are
            # uninterpretable: short reading 0.28 instead of 0.33 could mean the router is broken
            # or could mean 7% of segments cannot express it, and those two need different fixes.
            for name, frac in (self._register_avail_frac or {}).items():
                self.last_loss_components[f"avail_{name.lstrip('_')}"] = frac
        elif self.config.positional_prompt_p > 0.0:
            # Same argument as the line above, and the dual-register arm's ENTIRE identity rests on
            # it: with batch size 1 this reads 0 or 1 per step and must average to
            # positional_prompt_p over the run. Without it a router that silently never fired --
            # or always fired -- would look exactly like one that worked, for 23 hours.
            self.last_loss_components["alt_register_frac"] = self._alt_register_frac
        if self._partition_state is not None:
            extra = self._compute_partition_losses(audio_pred)
            if not torch.isfinite(extra).all():
                raise FloatingPointError(
                    f"partition/mix-energy loss not finite: {extra} "
                    f"(components {self.last_loss_components})")
            audio_loss = audio_loss + extra

        if self._cons_state is not None:
            cons = self._compute_mix_consistency_loss(audio_pred)
            if not torch.isfinite(cons).all():
                raise FloatingPointError(
                    f"mix consistency loss (arm CONS) not finite: {cons} "
                    f"(components {self.last_loss_components})")
            audio_loss = audio_loss + cons

        if self._decorr_state is not None:
            decorr = self._compute_stem_decorrelation_loss(audio_pred)
            if not torch.isfinite(decorr).all():
                raise FloatingPointError(
                    f"stem decorrelation loss (arm DECORR) not finite: {decorr} "
                    f"(components {self.last_loss_components})")
            audio_loss = audio_loss + decorr

        if self._ground_state is not None:
            ground = self._compute_visual_grounding_loss(audio_pred)
            if not torch.isfinite(ground).all():
                raise FloatingPointError(
                    f"visual grounding loss (arm B2) not finite: {ground} "
                    f"(components {self.last_loss_components})")
            audio_loss = audio_loss + ground

        self._loss_call_count += 1
        if self._loss_call_count <= 8 and len(self.last_loss_components) > 1:
            comp = ", ".join(f"{k}={v:.4f}" for k, v in self.last_loss_components.items())
            print(f"[jointstem] loss components (call {self._loss_call_count}): {comp}",
                  flush=True)

        if self.config.video_loss_weight == 0.0:
            return audio_loss

        video_mask = inputs.video_loss_mask.unsqueeze(-1).float()
        video_loss = (video_pred - inputs.video_targets).pow(2).mul(video_mask)
        video_loss = video_loss.sum(dim=[-2, -1]) / video_mask.sum(dim=[-2, -1]).clamp(min=1e-8)
        return audio_loss + self.config.video_loss_weight * video_loss

    def _compute_partition_losses(self, audio_pred: Tensor) -> Tensor:
        """Energy losses: partition (SS3.1) + mix-span tie (SS3.4/arm D2) + per-source (arm E). [B,].

        Float32 outside autocast; every log(E + c) computed as logsumexp([logE_hat, log c])
        -- exactly the log-of-sum form, overflow-immune on off-manifold x0-hats.
        """
        state = self._partition_state
        self._partition_state = None                                 # same-step consumption
        span_lens: list[int] = state["span_lens"]
        device = audio_pred.device

        with torch.autocast(device_type=device.type, enabled=False):
            w = self._probe_w.to(device)
            mu = self._probe_mu.to(device)
            sd = self._probe_sd.to(device)
            # Stem spans all share sigma (span_sigmas[0]); the mix span (last) may sit at
            # sigma_mix = mix_lead_alpha * sigma (arm F). Each span's x0-hat inversion and
            # sigma weight below use ITS OWN sigma; with mix_lead_alpha = 1.0 all are equal.
            sigmas = state["span_sigmas"][0].float()                 # [B] (stem sigma)
            e_mix = state["e_mix"].float()                           # [B, T]
            # Floor `c` is meant to be "small relative to this segment's TYPICAL SPEECH energy".
            # The median over ALL frames was a proxy for that and silently breaks once a segment
            # is more than half silence: the median lands on a zero frame, c becomes 0, and
            # log(0) = -inf poisons the loss. Measured on celebvhq_dialogue_v3, where turns are
            # shorter and carry inserted lead-ins: 21 of 400 segments (5%) have median(E_mix) == 0
            # exactly, and training died at step 42 with loss_per_source = inf (the fail-fast
            # guard below caught it rather than training through it).
            # The faithful version of the same intent is the median over ACTIVE frames, which
            # cannot collapse while the segment contains any speech at all. The absolute floor is
            # a last-resort guard for a fully silent mix, which should never reach training.
            active = e_mix > 0
            e_active = e_mix.masked_fill(~active, float("nan"))
            c = 1e-3 * e_active.nanmedian(dim=1, keepdim=True).values  # [B, 1] per segment
            c = torch.nan_to_num(c, nan=0.0).clamp_min(1e-12)
            log_c = c.log()
            log_gt = (e_mix + c).log()                               # [B, T]
            sigma_weight = (1.0 - sigmas).pow(2)                     # [B]

            def span_log_energy(idx: int) -> Tensor:
                """Frozen ridge probe on span idx's x0-hat -> log mel-power energy [B, T]."""
                start = sum(span_lens[:idx])
                v_pred = audio_pred[:, start:start + span_lens[idx], :].float()
                span_sig = state["span_sigmas"][idx].float().view(-1, 1, 1)
                x0_hat = state["noisy_spans"][idx].float() - span_sig * v_pred
                pad = torch.cat([x0_hat[:, :1], x0_hat, x0_hat[:, -1:]], dim=1)
                feats = torch.cat([pad[:, :-2], pad[:, 1:-1], pad[:, 2:]], dim=-1)
                if feats.shape[-1] != w.numel():
                    raise ValueError(f"probe dim {w.numel()} != feature dim {feats.shape[-1]}")
                return (feats - mu).div(sd) @ w + self._probe_b      # [B, T]

            extra = torch.zeros_like(sigma_weight)                   # [B]
            need_stem_log_e = (self.config.partition_loss_weight > 0.0
                               or self.config.per_source_energy_weight > 0.0)
            stem_log_e = ([span_log_energy(k) for k in range(self.config.num_stems)]
                          if need_stem_log_e else None)               # each [B, T]
            if self.config.partition_loss_weight > 0.0:
                log_sum = torch.logsumexp(
                    torch.stack(stem_log_e + [log_c.expand_as(stem_log_e[0])]), dim=0)
                partition_raw = (log_sum - log_gt).pow(2).mean(dim=1)  # [B]
                partition_loss = (self.config.partition_loss_weight * sigma_weight
                                  * partition_raw)
                extra = extra + partition_loss
                self.last_loss_components["loss_partition"] = (
                    partition_loss.detach().mean().item())
                self.last_loss_components["loss_partition_raw"] = (
                    partition_raw.detach().mean().item())

            if self.config.per_source_energy_weight > 0.0:
                # Arm E: each stem's probe energy vs that stem's OWN GT envelope. GT-silent
                # slices have E_k_GT ~ 0, so their target is log(c) -- the anti-absorption
                # force: a stem carrying the sibling's voice in its GT-silent half pays here.
                e_stem = state["e_stem"].float()                     # [B, K, T]
                per_source_terms = []
                for k in range(self.config.num_stems):
                    log_pred_k = torch.logsumexp(
                        torch.stack([stem_log_e[k], log_c.expand_as(stem_log_e[k])]), dim=0)
                    log_gt_k = (e_stem[:, k] + c).log()              # [B, T]
                    per_source_terms.append((log_pred_k - log_gt_k).pow(2).mean(dim=1))
                per_source_raw = torch.stack(per_source_terms).mean(dim=0)  # [B]; mean_{k,t}
                # G-fix-domain: per-SAMPLE weight when per_source_energy_weight_urmp overrides the
                # URMP half (root cause: the frozen probe is dialogue-only and confirmed biased on
                # music -- see the field's docstring). None -> the original scalar, byte-identical.
                if self.config.per_source_energy_weight_urmp is not None:
                    is_urmp = state["is_urmp"]                          # [B]
                    ps_weight = torch.where(
                        is_urmp > 0.5,
                        torch.full_like(is_urmp, self.config.per_source_energy_weight_urmp),
                        torch.full_like(is_urmp, self.config.per_source_energy_weight))
                else:
                    ps_weight = self.config.per_source_energy_weight
                per_source_loss = ps_weight * sigma_weight * per_source_raw
                extra = extra + per_source_loss
                self.last_loss_components["loss_per_source"] = (
                    per_source_loss.detach().mean().item())
                self.last_loss_components["loss_per_source_raw"] = (
                    per_source_raw.detach().mean().item())

            if self.config.mix_span_energy_weight > 0.0:
                mix_log_e = span_log_energy(len(span_lens) - 1)      # mix = last span
                log_pred = torch.logsumexp(
                    torch.stack([mix_log_e, log_c.expand_as(mix_log_e)]), dim=0)
                mix_energy_raw = (log_pred - log_gt).pow(2).mean(dim=1)  # [B]
                # Weight by the MIX span's own sigma (= stem sigma unless mix-leads is on).
                mix_sigma_weight = (1.0 - state["span_sigmas"][-1].float()).pow(2)
                mix_energy_loss = (self.config.mix_span_energy_weight * mix_sigma_weight
                                   * mix_energy_raw)
                extra = extra + mix_energy_loss
                self.last_loss_components["loss_mix_energy"] = (
                    mix_energy_loss.detach().mean().item())
                self.last_loss_components["loss_mix_energy_raw"] = (
                    mix_energy_raw.detach().mean().item())

            self.last_loss_components["partition_sigma_weight"] = (
                sigma_weight.detach().mean().item())
        return extra

    def refresh_cons_state_after_rollout(
        self, audio_latent: Tensor, stem_sigma: float, span_len: int
    ) -> None:
        """Re-point arm CONS's stash at the ROLLED state, so its x0-hat inversion is valid.

        WHY THIS EXISTS. `_compute_mix_consistency_loss` recovers each span's x0-hat as
        `noisy_span - sigma * v_pred`. Without a rollout that is exact: both terms come from the same
        `prepare_training_inputs` call. With `rollout_training=True` the stems have been advanced to a
        LATER sigma by `rollout_step.py` before the gradient-carrying forward pass, so `audio_pred` is
        the velocity at the rolled sigma while the stash still holds the window-start noisy tokens and
        window-start sigma. Pairing them produces a wrong x0-hat -- and because CONS then decodes that
        x0-hat through the real VAE and compares log-mels, the result is a finite, plausible-looking
        loss that is simply measuring the wrong signal. `joint_stems.py`'s DECORR guard names this
        exposure explicitly and left it unguarded ("a pre-existing gap"); this method closes it.

        The MIX span is deliberately NOT touched. `rollout_step.py` never advances it (it carries no
        LoRA gradient under `frozen_mix_lora` and no loss under `mix_loss_weight=0`), so its
        window-start (noisy, sigma) pair is still the pair `audio_pred` was computed from -- including
        the `sep_pattern_p` samples where that sigma is exactly 0 and the x0-hat is therefore the clean
        ground-truth mix, which is the case CONS most needs to be right about.
        """
        state = self._cons_state
        if state is None:
            raise RuntimeError(
                "refresh_cons_state_after_rollout() called but _cons_state is None -- "
                "prepare_training_inputs must run first and mix_consistency_weight must be > 0")
        span_lens: list[int] = state["span_lens"]
        num_stems = self.config.num_stems
        if any(length != span_len for length in span_lens[:num_stems]):
            raise ValueError(
                f"rollout span_len={span_len} disagrees with the stashed stem span lengths "
                f"{span_lens[:num_stems]} -- refusing to re-point the stash onto a different "
                "tokenisation")
        noisy = list(state["noisy_spans"])
        sigmas = list(state["span_sigmas"])
        for k in range(num_stems):
            start = k * span_len
            noisy[k] = audio_latent[:, start:start + span_len, :]
            sigmas[k] = torch.full_like(sigmas[k], stem_sigma)
        state["noisy_spans"] = noisy
        state["span_sigmas"] = sigmas
        state["sigma_source"] = "rolled"

    def _compute_mix_consistency_loss(self, audio_pred: Tensor) -> Tensor:
        """Arm CONS: decode-domain mixture-consistency loss. [B,].

        Inverts each span's x0-hat (same construction as ``_compute_partition_losses``),
        decodes stem0/.../stem{K-1}/mix through the TRUE (frozen) audio VAE decoder --
        ``self._audio_decoder``, wired in by train_jointstem.py -- and ties them in LOG-mel space:

            loss_cons = L1( logsumexp([mel_stem_0, ..., mel_stem_{K-1}, log_c]), mel_mix )

        logsumexp is log(sum(exp(.))), i.e. the log of a LINEAR-domain sum -- the physically
        correct way to combine mel POWER across incoherent sources, never a naive sum of the log
        values themselves. ``log_c`` is a small per-segment floor (the SAME "median over active
        frames" recipe _compute_partition_losses uses, but computed from the MIX's own decoded
        linear power -- CONS has no precomputed GT energy source at all, unlike the probe-based
        losses). The mix span's x0-hat is the ANCHOR (arm N makes it base-model-trustworthy by
        construction): the comparison target, not something this loss ever pulls on.

        Float32 outside autocast, exactly like _compute_partition_losses; the decoder call itself
        casts to bf16 internally (matching its own frozen bf16 weights, mirroring
        joint_stem_sampler.py's parity_decode), then casts the result back to float32 immediately.
        """
        state = self._cons_state
        self._cons_state = None                                       # same-step consumption
        if self.config.rollout_training and state.get("sigma_source") != "rolled":
            # Fail loud rather than return a finite, plausible, WRONG loss. See
            # refresh_cons_state_after_rollout() for the mechanism.
            raise RuntimeError(
                "arm CONS + rollout_training: _cons_state still holds the WINDOW-START "
                f"(noisy, sigma) pair (sigma_source={state.get('sigma_source')!r}) while audio_pred "
                "is the velocity at the ROLLED-TO sigma. The x0-hat inversion would be wrong and the "
                "decode-domain loss would look fine and measure the wrong signal. rollout_step.py "
                "must call strategy.refresh_cons_state_after_rollout() before compute_loss().")
        if self._audio_decoder is None:
            raise RuntimeError(
                "mix_consistency_weight>0 (arm CONS) but strategy._audio_decoder was never set "
                "-- train_jointstem.py must assign strategy._audio_decoder = trainer._audio_vae "
                "right after constructing the trainer, gated behind mix_consistency_weight > 0")
        span_lens: list[int] = state["span_lens"]
        num_stems = self.config.num_stems
        mix_idx = len(span_lens) - 1
        device = audio_pred.device

        with torch.autocast(device_type=device.type, enabled=False):

            def span_x0_hat(idx: int) -> Tensor:
                start = sum(span_lens[:idx])
                v_pred = audio_pred[:, start:start + span_lens[idx], :].float()
                span_sig = state["span_sigmas"][idx].float().view(-1, 1, 1)
                return state["noisy_spans"][idx].float() - span_sig * v_pred

            def decode_log_mel(idx: int) -> Tensor:
                """x0-hat tokens [B, span_len, D] -> log-mel [B, C_out, T', F'] via the true VAE
                decode. channels=8, mel_bins=16 is the audio latent shape this codebase's VAE
                always uses (verified against 5+ existing call sites, e.g.
                stemgen/strategy.py's consistency_loss, stemgen/sampler.py's slot reconstruction,
                and scripts/measure_audio_decoder_cost.py's z_channels=8 decoder build -- not a
                guess).

                Re-asserts device placement on every call (cheap no-op if already correct):
                `validation_sampler.py` moves this SAME decoder object to GPU before its own use
                and back to CPU after (lines ~809/814, its own memory-management convention), and
                step-0 validation runs BEFORE the first real training step -- a one-time `.to()`
                at strategy setup is not enough; two crashes tonight ("Input type
                (CUDABFloat16Type) and weight type (CPUBFloat16Type)") were this exact staleness,
                not a setup bug."""
                x0_hat = span_x0_hat(idx)
                latent = self._audio_patchifier.unpatchify(
                    x0_hat.to(torch.bfloat16),
                    AudioLatentShape(batch=x0_hat.shape[0], channels=8,
                                      frames=span_lens[idx], mel_bins=16))
                self._audio_decoder = self._audio_decoder.to(latent.device)
                return self._audio_decoder(latent).float()

            stem_mels = [decode_log_mel(k) for k in range(num_stems)]
            mix_mel = decode_log_mel(mix_idx)
            if self.config.mix_consistency_detach_mix:
                # One-directional obligation: the mix is the TARGET, never pulled toward the stem
                # sum. Detached BEFORE log_c is derived from it, so the floor is a constant too.
                mix_mel = mix_mel.detach()

            # Floor `c`: the SAME "small fraction of typical power" intent as
            # _compute_partition_losses, computed ENTIRELY IN LOG SPACE to avoid the exp-overflow
            # this loss originally risked (mix_mel.exp() on an off-manifold high-sigma decode can
            # overflow float32). median commutes with any monotonic transform, so
            # log(median(exp(mel))) == median(mel) exactly -- computing the median directly on
            # mel and adding log(1e-3) gives the IDENTICAL floor value as the round-trip-through-
            # exp version, never materializing exp() over the full tensor. No GT energy source to
            # read a floor scale from here (unlike the probe-based losses), so the floor is
            # derived from the mix's own decoded power, same as before.
            log_c = (mix_mel.flatten(1).median(dim=1, keepdim=True).values + math.log(1e-3)
                     ).view(-1, *([1] * (mix_mel.dim() - 1)))    # broadcastable to mel shape

            # Combine the stems at exponent p: (1/p) * logsumexp(p * mel). The domain is log-
            # MAGNITUDE mel throughout (phase-free by construction), and p = 1 -- the default and
            # what this arm ships -- sums those magnitudes, bit-for-bit the historical CONS
            # combination. p = 2 would sum powers instead; see mix_consistency_power. Scaling
            # inside the logsumexp keeps everything in log space, preserving the overflow safety
            # the floor construction was written for.
            p = float(self.config.mix_consistency_power)
            log_sum_stems = (1.0 / p) * torch.logsumexp(
                p * torch.stack(stem_mels + [log_c.expand_as(mix_mel)]), dim=0)
            reduce_dims = list(range(1, mix_mel.dim()))
            cons_raw = (log_sum_stems - mix_mel).abs().mean(dim=reduce_dims)  # [B]
            cons_loss = self.config.mix_consistency_weight * cons_raw

        self.last_loss_components["loss_cons"] = cons_loss.detach().mean().item()
        self.last_loss_components["loss_cons_raw"] = cons_raw.detach().mean().item()
        return cons_loss

    def _compute_visual_grounding_loss(self, audio_pred: Tensor) -> Tensor:
        """Arm B2: tie each stem's activity to the MOTION in its own region of the frame. [B,].

        For stem k: take its x0-hat energy envelope (pooled onto the video latent grid) and the
        motion energy of the video inside stem k's box, then score their agreement at lag 0
        against agreement at shifted lags, as an InfoNCE:

            loss = -log softmax_over_shifts( <a_k, m_k(shift)> / tau )[shift = 0]

        WHY A SHIFT CONTRAST rather than a correlation or a regression: the failure we measured
        is "right content, wrong time", so the objective should reward being aligned MORE than
        being misaligned -- not any particular functional map from motion to loudness (a mouth
        can move without sound, an instrument can ring after the bow stops). This also makes the
        loss scale-free: both envelopes are z-scored, so it cannot be satisfied by matching
        levels, which is CONS's job.

        Everything is masked by the per-stem `valid` flag, and by whether the envelopes carry any
        variance at all -- a silent stem or a static region has no timing to align, and forcing
        one would be inventing supervision.
        """
        state = self._ground_state
        self._ground_state = None                                     # same-step consumption
        span_lens: list[int] = state["span_lens"]
        video = state["video_clean"].float()                          # [B, C, T, H, W]
        boxes = state["boxes"].float()                                # [B, K, T, 4]
        valid = state["boxes_valid"]                                  # [B, K]
        b, _c, t_v, h, w = video.shape
        num_stems = self.config.num_stems
        shifts = list(range(-self.config.visual_grounding_max_shift,
                            self.config.visual_grounding_max_shift + 1))

        with torch.autocast(device_type=audio_pred.device.type, enabled=False):
            # Video side: per-frame motion energy inside each stem's box. Frame differences make
            # this a MOTION cue (a moving mouth / bowing arm), not an appearance cue.
            diff = (video[:, :, 1:] - video[:, :, :-1]).pow(2).mean(dim=1)      # [B, T-1, H, W]
            diff = torch.cat([diff[:, :1], diff], dim=1)                        # [B, T, H, W]
            ys = torch.arange(h, device=video.device).view(1, 1, 1, h, 1) / h
            xs = torch.arange(w, device=video.device).view(1, 1, 1, 1, w) / w
            x0, y0 = boxes[..., 0:1].unsqueeze(-1), boxes[..., 1:2].unsqueeze(-1)
            x1, y1 = boxes[..., 2:3].unsqueeze(-1), boxes[..., 3:4].unsqueeze(-1)
            region = (((xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1))
                      .float())                                                 # [B, K, T, H, W]
            area = region.flatten(3).sum(-1).clamp(min=1.0)                      # [B, K, T]
            motion = (region * diff.unsqueeze(1)).flatten(3).sum(-1) / area      # [B, K, T]

            # Audio side: x0-hat energy per stem, pooled onto the same T frames.
            env = []
            for k in range(num_stems):
                start = sum(span_lens[:k])
                v_pred = audio_pred[:, start:start + span_lens[k], :].float()
                x0_hat = state["noisy_spans"][k].float() - \
                    state["span_sigmas"][k].float().view(-1, 1, 1) * v_pred
                e = x0_hat.pow(2).mean(dim=-1)                                   # [B, span_len]
                e = torch.nn.functional.adaptive_avg_pool1d(
                    e.unsqueeze(1), t_v).squeeze(1)                              # [B, T]
                env.append(torch.log(e + 1e-8))
            audio_env = torch.stack(env, dim=1)                                  # [B, K, T]

            def z(x: Tensor) -> Tensor:
                return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)

            a = z(audio_env)
            m = z(torch.log(motion + 1e-8))
            sims = []
            for s in shifts:
                ms = torch.roll(m, shifts=s, dims=-1)
                sims.append((a * ms).mean(dim=-1))                               # [B, K]
            logits = torch.stack(sims, dim=-1) / 0.1                             # [B, K, S]
            target = torch.full((b, num_stems), shifts.index(0), device=logits.device)
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, len(shifts)), target.reshape(-1), reduction="none"
            ).view(b, num_stems)

            # Mask: invalid boxes, silent stems, and regions whose motion carries no timing.
            # The motion test is on the LOG envelope -- the same quantity that gets z-scored --
            # because that is what decides whether the target is signal or amplified noise.
            log_motion_std = torch.log(motion + 1e-8).std(dim=-1)
            live = ((audio_env.std(dim=-1) > 1e-3)
                    & (log_motion_std > self.config.visual_grounding_min_motion_std)
                    & valid.bool())
            denom = live.float().sum(dim=1).clamp(min=1.0)
            per_sample = (ce * live.float()).sum(dim=1) / denom
            loss = self.config.visual_grounding_weight * per_sample

        self.last_loss_components["loss_ground"] = float(loss.detach().mean())
        self.last_loss_components["ground_live_frac"] = float(live.float().mean())
        return loss

    def _compute_stem_decorrelation_loss(self, audio_pred: Tensor) -> Tensor:
        """Arm DECORR: covariance-based decorrelation loss between stem0 and stem1. [B,].

        Decodes stem0/stem1's x0-hats through the TRUE (frozen) audio VAE decoder -- the SAME
        x0-hat-inversion + unpatchify + decode mechanics as ``_compute_mix_consistency_loss``
        (``x0_hat = noisy_span - sigma * v_pred``), kept as an INDEPENDENT stash/readout
        (``self._decorr_state``, never ``self._cons_state``) so this loss never requires
        ``include_mix`` or arm CONS to be active -- it only ever touches stem0 and stem1.

        Computes the differentiable Pearson correlation between each batch element's flattened
        decoded log-mel spectrograms (mean-center, dot product, normalize by norms -- exactly the
        numpy ``corr()`` in ``.scripts/analyze_sum_final.py``, reimplemented in torch with
        gradients flowing back through the frozen decoder into both stems' x0-hats):

            corr = (mel0 - mean(mel0)) . (mel1 - mean(mel1)) / (||mel0 - mean|| ||mel1 - mean|| + eps)
            loss_decorr = stem_decorrelation_weight * relu(corr)^2

        Only POSITIVE correlation (duplicated content) is penalized -- negative correlation is
        left alone, since this is music where both stems may legitimately be simultaneously
        ACTIVE (same onset/offset timing) without carrying the SAME content; activity overlap is
        not the failure mode this arm targets, content duplication is.

        Float32 outside autocast, exactly like the other decode-domain losses; the decoder call
        itself casts to bf16 internally then back to float32 immediately.
        """
        state = self._decorr_state
        self._decorr_state = None                                    # same-step consumption
        if self._audio_decoder is None:
            raise RuntimeError(
                "stem_decorrelation_weight>0 (arm DECORR) but strategy._audio_decoder was never "
                "set -- train_jointstem.py must assign strategy._audio_decoder = "
                "trainer._audio_vae right after constructing the trainer, gated behind "
                "stem_decorrelation_weight > 0 (or mix_consistency_weight > 0, whichever is set)")
        span_lens: list[int] = state["span_lens"]
        device = audio_pred.device

        with torch.autocast(device_type=device.type, enabled=False):

            def span_x0_hat(idx: int) -> Tensor:
                start = sum(span_lens[:idx])
                v_pred = audio_pred[:, start:start + span_lens[idx], :].float()
                span_sig = state["span_sigmas"][idx].float().view(-1, 1, 1)
                return state["noisy_spans"][idx].float() - span_sig * v_pred

            def decode_log_mel(idx: int) -> Tensor:
                # Re-assert device placement every call: validation_sampler.py moves this SAME
                # decoder object to GPU before its own use and back to CPU after (its own
                # memory-management convention), and step-0 validation runs before the first
                # real training step -- a one-time `.to()` at strategy setup goes stale. Two
                # crashes tonight ("Input type (CUDABFloat16Type) and weight type
                # (CPUBFloat16Type)") were exactly this, on CONS's identical decode helper.
                x0_hat = span_x0_hat(idx)
                latent = self._audio_patchifier.unpatchify(
                    x0_hat.to(torch.bfloat16),
                    AudioLatentShape(batch=x0_hat.shape[0], channels=8,
                                      frames=span_lens[idx], mel_bins=16))
                self._audio_decoder = self._audio_decoder.to(latent.device)
                return self._audio_decoder(latent).float()

            mel0 = decode_log_mel(0).flatten(1)                       # [B, C*T'*F']
            mel1 = decode_log_mel(1).flatten(1)
            mel0 = mel0 - mel0.mean(dim=1, keepdim=True)
            mel1 = mel1 - mel1.mean(dim=1, keepdim=True)
            dot = (mel0 * mel1).sum(dim=1)
            norm0 = mel0.norm(dim=1)
            norm1 = mel1.norm(dim=1)
            corr = dot / (norm0 * norm1 + 1e-8)                       # [B], matches numpy corr()
            decorr_raw = torch.relu(corr).pow(2)                      # [B]
            decorr_loss = self.config.stem_decorrelation_weight * decorr_raw

        self.last_loss_components["loss_decorr"] = decorr_loss.detach().mean().item()
        self.last_loss_components["loss_decorr_raw"] = decorr_raw.detach().mean().item()
        self.last_loss_components["stem_corr"] = corr.detach().mean().item()
        return decorr_loss

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "strategy": "joint_stem_generation",
            "num_stems": self.config.num_stems,
            "include_mix": self.config.include_mix,
            "mix_loss_weight": self.config.mix_loss_weight,
            "ref_time_offset": self.config.ref_time_offset,
            "mix_lead_alpha": self.config.mix_lead_alpha,
            "mix_sigma_mode": self.config.mix_sigma_mode,
            "sep_pattern_p": self.config.sep_pattern_p,
            "input_perturbation_gamma": self.config.input_perturbation_gamma,
            "frozen_mix_lora": self.config.frozen_mix_lora,
            "a2v_mix_only": self.config.a2v_mix_only,
            "span_axis_pairs": self.config.span_axis_pairs,
            "span_axis_omega": self.config.span_axis_omega,
            "span_axis_coords": list(self.config.span_axis_coords),
            # Arm K: a mask mismatch between training and sampling is a SILENT quality bug, so the
            # checkpoint records the topology it was trained under.
            "span_attention_topology": self.config.span_attention_topology,
            "span_attention_sibling": self.config.span_attention_sibling,
            "partition_loss_weight": self.config.partition_loss_weight,
            "partition_readout": self.config.partition_readout,
            "mix_span_energy_weight": self.config.mix_span_energy_weight,
            "per_source_energy_weight": self.config.per_source_energy_weight,
            "per_source_energy_weight_urmp": self.config.per_source_energy_weight_urmp,
            # Arm I: which mix distribution the stems were trained to read (GT vs the model's own).
            "selfboot_mix_p": self.config.selfboot_mix_p,
            "selfboot_mix_dir": self.config.selfboot_mix_dir,
            # Arm CONS: decode-domain mixture-consistency weight (0 = off) + the two knobs that
            # change WHAT it optimizes (direction and magnitude-vs-power combination), so a
            # checkpoint can never be re-scored under a different rule than it trained on.
            "mix_consistency_weight": self.config.mix_consistency_weight,
            "mix_consistency_detach_mix": self.config.mix_consistency_detach_mix,
            "mix_consistency_power": self.config.mix_consistency_power,
            # Arm DECORR: stem0-vs-stem1 decorrelation weight (0 = off).
            "stem_decorrelation_weight": self.config.stem_decorrelation_weight,
            # Arm CHAN: mix-channel conditioning (0 = off, the audio_patchify_proj gate never armed).
            "mix_channel_conditioning": self.config.mix_channel_conditioning,
        }
