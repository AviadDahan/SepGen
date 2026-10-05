"""SepGen: single-pass GENERATION sampler for [stem0 | stem1 (| mix)].

Validation headline: from text + first frame, denoise ALL audio spans from noise in ONE joint pass
(the trained task), then decode each span to its own waveform. Per-channel text is enforced with
the block-diagonal gate on ``audio_attn2`` (see ``text_block_gate.py``): the concatenated positive
audio context ``[stem0_text | stem1_text (| scene_text)]`` is stashed as ``joint_audio_context`` and
each span attends only to its own block. The CFG negative pass uses the single-block negative prompt
and is left unmasked by the gate.

Reuses the frozen trainer's ``_run_denoising`` (scheduler, CFG, STG) unchanged -- only the audio
sequence layout, the positive context, and the block bias are ours. Video follows the joint_denoise
path (first-frame-conditioned generation), matching the mix-conditioned validation so only the audio
task differs.

ARM F (mix-leads, ``mix_lead_alpha < 1``) swaps the frozen loop for
``_run_denoising_mix_leads``: same LTX2Scheduler grid, same CFG **and STG** composition (both are
x0-space deltas, schedule-agnostic, so STG stays ENABLED), but the audio timesteps are PER-TOKEN --
stem tokens at sigma_i, mix tokens at ``mix_lead_alpha * sigma_i`` -- exactly the (sigma, alpha*sigma)
pairs the arm-F strategy trains on. The frozen loop cannot express this: it applies ONE scalar sigma
to every audio token (``timesteps = sigma * denoise_mask``) and its Euler step calls
``to_velocity``, which collapses sigma with ``.item()``. ``X0Model`` already converts velocity ->
x0-hat with the per-token ``Modality.timesteps`` (model.py ``to_denoised`` broadcast), so only the
Euler update needs the per-token form (``per_token_euler_update``). The mix span is initialised at
``alpha * noise`` (clean initial latent is zeros), the level-``alpha*sigma_0 = alpha`` marginal
under a zero-mean prior -- the one unavoidable bootstrap approximation of the multiplicative map,
since training never shows the mix above ``alpha``. Both spans reach sigma = 0 together at the
final grid point, the mix having led by the factor ``alpha`` the whole way; tokens whose sigma hits
0 early (never with this grid, defensively handled anyway) are held fixed.

ARM G (per-span INDEPENDENT training sigma) makes the schedule a FREE INFERENCE-TIME CHOICE:
once ``sigma_mix`` is drawn i.i.d. from the stems' own marginal, every ``(sigma_stem, sigma_mix)``
pair is in-distribution, so any mix track can be sampled post-hoc without retraining (Diffusion
Forcing 2407.01392 §4). The staggered loop therefore accepts an OPTIONAL explicit
``mix_sigma_track`` -- the mix span's sigma at every grid point -- instead of only the
multiplicative ``alpha * sigma`` map. ``mix_sigma_track = None`` keeps the arm-F/G multiplicative
path bit-for-bit unchanged (same ``build_span_sigma_scale`` multiply, same mix init scale
``alpha``; note ``sigmas[0] = 0.99999994``, NOT exactly 1.0, so deriving the init scale from
``track[0]`` would NOT be byte-identical -- hence the two paths stay separate). Track builders:
``build_mix_sigma_track_multiplicative`` / ``_step_lead`` / ``_early_finish``, all validated by
``validate_mix_sigma_track`` (monotone, non-negative, mix never behind the stems, ends at 0).

ARM K (asymmetric cross-span SELF-attention mask) is armed here for BOTH loops via the module-level
``audio_span_gate`` wrapper on ``audio_attn1``: the mix span attends only itself, stems attend own
(+ mix, + sibling when enabled). Unlike the text gate there is NO positive/negative length routing
-- this gate's key axis is the AUDIO token axis, identical on the positive, CFG-negative and
STG-perturbed passes, and the mask is a property of the sequence layout rather than the prompt, so
applying it identically on all three is exactly correct. ``joint_span_allow = None`` (legacy)
leaves the gate a pass-through.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import (
    CFGGuider,
    MultiModalGuider,
    MultiModalGuiderParams,
    STGGuider,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import X0Model
from ltx_core.types import LatentState

from stemgen import a2v_mix_gate, lora_span_gate
from stemgen.audio_span_gate import (
    build_span_attention_bias,
    clear_span_attention_bias,
    current_span_attention_bias,
    set_span_attention_bias,
)
from stemgen.av_reference_sampler import AVReferenceSampler
from stemgen.spatial_mask_gate import clear_video_key_bias
from stemgen.text_block_gate import (
    build_block_diagonal_text_bias,
    clear_text_block_bias,
    current_text_block_bias,
    set_text_block_bias,
)


def build_span_sigma_scale(
    num_stems: int,
    span_len: int,
    mix_lead_alpha: float,
    device: torch.device,
) -> Tensor:
    """Per-token sigma multiplier ``[1, (num_stems+1)*span_len, 1]`` for ``[stem0|stem1|mix]``.

    Stem tokens get 1.0; the trailing mix span gets ``mix_lead_alpha``. Multiplying the scalar
    grid sigma by this vector yields the arm-F per-token timesteps (stems at sigma, mix at
    alpha*sigma) -- the same map the strategy trains with. ``num_stems = 0`` is the two-phase
    pass-1 mix-only layout: the stem block is empty and the whole sequence is the mix span.
    """
    if not 0.0 < mix_lead_alpha <= 1.0:
        raise ValueError(f"mix_lead_alpha must be in (0, 1], got {mix_lead_alpha}")
    if num_stems < 0 or span_len < 1:
        raise ValueError(f"need num_stems>=0 and span_len>=1, got {num_stems}, {span_len}")
    return torch.cat([
        torch.ones(1, num_stems * span_len, 1, device=device, dtype=torch.float32),
        torch.full((1, span_len, 1), float(mix_lead_alpha), device=device,
                   dtype=torch.float32),
    ], dim=1)


def build_mix_sigma_track_multiplicative(sigmas: Tensor, alpha: float) -> Tensor:
    """Multiplicative mix-lead track ``sigma_mix(i) = alpha * sigma(i)`` -- the arm-F map.

    Provided so the sweep can express EVERY schedule as an explicit track; the sampler's default
    path (``mix_sigma_track = None``) still uses ``build_span_sigma_scale`` so arm F/G output
    stays bit-identical.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    return alpha * sigmas.detach().to(torch.float32).cpu()


def build_mix_sigma_track_step_lead(sigmas: Tensor, lead_steps: int) -> Tensor:
    """Fixed-STEP mix lead: ``sigma_mix(i) = sigma(min(i + lead_steps, K))``.

    The mix walks the scheduler's OWN grid, merely shifted by ``lead_steps`` grid points, so
    every sigma it occupies is one the stems also occupy and each pair is a plain pair of grid
    points. This is a genuinely different mechanism from the multiplicative lead, not a
    reparametrisation of it: on the LTX2 grid the multiplicative lead is LARGEST at sigma ~ 1
    (0.2 at alpha = 0.8) and vanishes at the end, whereas the step lead is nearly flat at the top
    (sigma_0 .. sigma_5 span only 0.03) and largest at the bottom, where the grid accelerates.

    The tail is implied rather than chosen: once ``i + lead_steps > K`` the mix sits at sigma = 0
    and ``per_token_euler_update`` holds it, so the last ``lead_steps`` grid points are a
    finish-early-then-hold phase and inherit the sigma = 0 caveat documented on
    ``build_mix_sigma_track_early_finish``.
    """
    if lead_steps < 1:
        raise ValueError(f"lead_steps must be >= 1, got {lead_steps}")
    s = sigmas.detach().to(torch.float32).cpu()
    last = s.shape[0] - 1
    if lead_steps > last:
        raise ValueError(f"lead_steps {lead_steps} exceeds the grid's {last} steps")
    idx = torch.clamp(torch.arange(s.shape[0]) + lead_steps, max=last)
    return s[idx]


def build_mix_sigma_track_early_finish(sigmas: Tensor, finish_fraction: float) -> Tensor:
    """Mix finishes early, then HOLDS clean: ``sigma_mix(i) = sigma(min(round(i / f), K))``.

    The mix walks the same grid at ``1 / f`` speed, reaches sigma = 0 at grid point
    ``round(f * K)``, and is held there (``per_token_euler_update`` freezes sigma = 0 tokens)
    while the stems finish against a fully committed, frozen leader.

    CAVEAT -- this is the ONE schedule in the sweep that is not strictly in-distribution, and it
    must be reported as such. Arm G's training sigmas come from
    ``ShiftedLogitNormalTimestepSampler`` (third_party/LTX-2/.../ltx_trainer/timestep_samplers.py):
    values below ``eps = 1e-3`` are REFLECTED to ``2*eps - x`` and the uniform fallback is
    ``(1 - eps) * rand + eps``, so both branches return >= 1e-3 and sigma_mix = 0 is NEVER trained.
    The frozen loop never queries the model at sigma = 0 either (it iterates ``sigmas[:-1]``, last
    query at 0.1). Held mix tokens are therefore conditioned at an AdaLN timestep 1e-3 below
    anything seen in training -- a small but real boundary extrapolation, unlike the
    multiplicative and step-lead tracks whose interior sigmas are all trained.
    """
    if not 0.0 < finish_fraction < 1.0:
        raise ValueError(f"finish_fraction must be in (0, 1), got {finish_fraction}")
    s = sigmas.detach().to(torch.float32).cpu()
    last = s.shape[0] - 1
    idx = torch.clamp(
        torch.round(torch.arange(s.shape[0], dtype=torch.float32) / finish_fraction).long(),
        max=last)
    return s[idx]


def steps_matching_alpha(sigmas: Tensor, alpha: float) -> int:
    """Grid-point lead whose MEAN sigma gap matches the multiplicative ``alpha`` lead's.

    Removes the free parameter from the step-lead schedule: rather than picking a lead length by
    taste, pick the one that leads by the same average amount as the reference multiplicative
    schedule, so the two tracks differ in SHAPE (constant-ratio vs constant-index) at matched
    magnitude and the comparison isolates the shape.
    """
    s = sigmas.detach().to(torch.float32).cpu()
    target = float((s - alpha * s).mean())
    last = s.shape[0] - 1
    gaps = [float((s - build_mix_sigma_track_step_lead(s, n)).mean()) for n in range(1, last + 1)]
    return 1 + min(range(len(gaps)), key=lambda i: abs(gaps[i] - target))


def validate_mix_sigma_track(mix_sigma_track: Tensor, sigmas: Tensor) -> Tensor:
    """Fail-fast checks on an explicit mix sigma track against the scheduler grid it runs on.

    A track that silently disagrees with the grid (wrong step count, mix noisier than the stems,
    non-monotone, not landing clean) would produce plausible-looking audio from an untrained map,
    which is exactly the failure the sweep exists to rule out -- so every property is asserted.
    """
    t = mix_sigma_track.detach().to(torch.float32).cpu().reshape(-1)
    s = sigmas.detach().to(torch.float32).cpu().reshape(-1)
    if t.shape != s.shape:
        raise ValueError(
            f"mix_sigma_track has {t.shape[0]} grid points but the scheduler grid has "
            f"{s.shape[0]} (num_inference_steps + 1) -- build the track from the SAME grid")
    if not torch.isfinite(t).all():
        raise ValueError("mix_sigma_track contains non-finite values")
    if (t < 0).any():
        raise ValueError(f"mix_sigma_track must be non-negative, min {float(t.min())}")
    if (t[:-1] < t[1:]).any():
        raise ValueError("mix_sigma_track must be non-increasing (a mix that re-noises is not a lead)")
    if (t > s + 1e-6).any():
        worst = int(torch.argmax(t - s))
        raise ValueError(
            f"mix_sigma_track must never be BEHIND the stems: at grid point {worst} "
            f"mix {float(t[worst]):.4f} > stem {float(s[worst]):.4f}")
    if float(t[-1]) != 0.0:
        raise ValueError(f"mix_sigma_track must end clean (0.0), got {float(t[-1])}")
    return t


def build_span_sigma_grid(
    sigmas: Tensor,
    num_stems: int,
    span_len: int,
    mix_lead_alpha: float,
    mix_sigma_track: Tensor | None,
    device: torch.device,
) -> Tensor:
    """Per-token sigma at EVERY grid point, ``[K+1, 1, (num_stems+1)*span_len, 1]``.

    ``mix_sigma_track = None`` reproduces the multiplicative path exactly: the returned grid is
    ``sigmas[i] * build_span_sigma_scale(...)``, the same float32 product the loop used inline
    before, so arm F/G sampling is bit-identical. With a track, stem rows carry the grid sigma and
    the mix rows carry the track.
    """
    s = sigmas.to(device=device, dtype=torch.float32).reshape(-1, 1, 1, 1)
    if mix_sigma_track is None:
        scale = build_span_sigma_scale(
            num_stems=num_stems, span_len=span_len, mix_lead_alpha=mix_lead_alpha, device=device)
        return s * scale.unsqueeze(0)
    track = validate_mix_sigma_track(mix_sigma_track, sigmas).to(device).reshape(-1, 1, 1, 1)
    stem_rows = s.expand(-1, 1, num_stems * span_len, 1)
    mix_rows = track.expand(-1, 1, span_len, 1)
    return torch.cat([stem_rows, mix_rows], dim=2).contiguous()


def per_token_euler_update(
    latent: Tensor,
    denoised: Tensor,
    sigma_tok: Tensor,
    sigma_next_tok: Tensor,
) -> Tensor:
    """Flow-matching Euler step with PER-TOKEN sigma: ``x + (x - x0) * dsigma / sigma``.

    Exact per-token analogue of the frozen ``EulerDiffusionStep`` (which only accepts a scalar
    sigma via ``to_velocity``'s ``.item()``): on-manifold states stay on-manifold at their own
    sigma track. Tokens whose sigma is already 0 are held fixed (their span is done).

    Args:
        latent / denoised: ``[B, T, D]`` current sample and x0-hat.
        sigma_tok / sigma_next_tok: ``[B, T, 1]`` (or broadcastable) current and next sigma.
    """
    x = latent.to(torch.float32)
    x0 = denoised.to(torch.float32)
    sig = sigma_tok.to(torch.float32)
    dsig = sigma_next_tok.to(torch.float32) - sig
    safe_sig = torch.where(sig > 0, sig, torch.ones_like(sig))
    stepped = x + (x - x0) * (dsig / safe_sig)
    return torch.where(sig > 0, stepped, x).to(latent.dtype)


class JointStemGenerationRequest:
    """Marker switching the sampler into joint K-stem generation for one validation prompt.

    ``mix_lead_alpha`` < 1 selects the arm-F mix-leads staggered loop
    (``_run_denoising_mix_leads``); 1.0 keeps the frozen shared-sigma ``_run_denoising``.
    Must equal the training config's ``joint_stem.mix_lead_alpha``.

    ``teacher_force_mix_latent`` (DIAGNOSTIC probe, default None = behavior unchanged):
    patchified GROUND-TRUTH mix tokens ``[1, span_len, D]`` (AudioPatchifier layout
    ``b t (c f)``). Requires ``include_mix`` and ``mix_lead_alpha < 1``. The arm-F loop then
    CLAMPS the mix span to the GT mix latent noised at its own declared level
    ``alpha * sigma`` -- training's convention ``(1 - s) * x0 + s * eps``
    (joint_stems.py mix-span noising) with ONE fixed ``eps`` drawn from
    ``teacher_force_seed`` and held for the whole trajectory. The clamp is applied to the
    INITIAL state (sigma_0, where the default bootstrap has zero content) and after every
    per-token Euler update (sigma_{i+1}); the final grid point (sigma = 0) leaves the clean
    GT mix latent. This deliberately reduces arm F to a mix-conditioned separator: it
    isolates whether the gibberish stems come from the self-generated (unsupervised) mix
    CONTENT (H-TF) or from the (sigma, alpha*sigma) pairing itself (H0). An experimental
    schedule, not used by the paper.

    ``mix_sigma_track`` (arm-G schedule sweep, default None = behaviour unchanged): explicit mix
    sigma at every scheduler grid point, ``[num_inference_steps + 1]``, built from the SAME
    ``LTX2Scheduler`` grid the sampler will use (mismatch fails loud in
    ``validate_mix_sigma_track``). Setting it selects the staggered loop at any
    ``mix_lead_alpha`` and OVERRIDES the multiplicative map, so schedules the fixed alpha cannot
    express -- a fixed-STEP lead, or a mix that finishes early and holds -- become plain
    inference-time choices on an already-trained arm-G checkpoint. Requires ``include_mix``.
    """

    def __init__(self, ref_time_offset: float = 0.0, mix_lead_alpha: float = 1.0,
                 teacher_force_mix_latent: Tensor | None = None,
                 teacher_force_seed: int = 0,
                 mix_sigma_track: Tensor | None = None,
                 energy_parity_guidance: bool = False,
                 two_phase: bool = False,
                 mix_only: bool = False,
                 span_pe_layout: str = "aligned"):
        # ARM CATPE: must equal the training config's joint_stem.span_pe_layout, exactly like
        # ref_time_offset -- a sampler whose PE layout differs from training renders from
        # positions the weights never saw, silently.
        self.span_pe_layout = span_pe_layout
        self.ref_time_offset = ref_time_offset
        self.mix_lead_alpha = mix_lead_alpha
        self.teacher_force_mix_latent = teacher_force_mix_latent
        self.teacher_force_seed = teacher_force_seed
        self.mix_sigma_track = mix_sigma_track
        # MIX-ONLY render (two-phase pass 1, 2026-08-02): NO stem spans in the sequence at
        # all -- the audio layout is a single mix span and the pass is pure base computation
        # (LoRA span gate re-armed to zero the delta sequence-wide, A2V gate vacuous). The
        # method returns (None, None): only the mix LATENT is produced, for direct handoff
        # to pass 2; nothing is decoded. Internal to two_phase -- not a standalone mode.
        self.mix_only = mix_only
        if mix_only and (two_phase or teacher_force_mix_latent is not None
                         or mix_sigma_track is not None or energy_parity_guidance):
            raise ValueError("mix_only is the two-phase pass-1 internal mode; it cannot be "
                             "combined with two_phase / teacher_force_mix_latent / "
                             "mix_sigma_track / energy_parity_guidance")
        # TWO-PHASE COMMITTED render (mixfirst arm A1, 2026-08-02): pass 1 = the plain joint
        # render (mix span from the frozen base, free-running); pass 2 = stems re-denoised
        # from fresh noise with pass 1's OWN mix latent committed clean at sigma 0 (an
        # all-zeros mix_sigma_track + the teacher-force clamp — the Test-A on_gen machinery
        # with a direct latent handoff instead of the wav round-trip). This is the arm's
        # TRAINED task (sep_pattern_p 1.0 trains only (sigma_stem, 0) pairs), so validation
        # under two_phase renders the capability the run actually optimizes. Mutually
        # exclusive with an explicit teacher_force_mix_latent / mix_sigma_track.
        self.two_phase = two_phase
        if two_phase and (teacher_force_mix_latent is not None or mix_sigma_track is not None):
            raise ValueError("two_phase builds its own clamp + track; do not pass "
                             "teacher_force_mix_latent / mix_sigma_track alongside it")
        # Frozen-anchor energy-parity guidance (stemgen/energy_parity.py): per sampling step,
        # gradient-nudge the STEM spans' x0-hat through the audio VAE decoder until their summed
        # mel power matches the frozen mix span's (parameter-free Gauss-Newton; ratio preserved).
        # Requires include_mix; only meaningful under arm N (frozen mix = trustworthy anchor).
        self.energy_parity_guidance = energy_parity_guidance


@dataclass(frozen=True)
class DevGuidance:
    """Per-modality guidance for the NON-DISTILLED (dev) checkpoint, defaults mirroring the
    upstream LTX-2.3 one-stage pipeline (``ltx_pipelines/utils/constants.py`` LTX_2_3_PARAMS):
    video CFG 3.0, audio CFG 7.0, STG 1.0, CFG-rescale 0.7. Composed via the upstream
    ``MultiModalGuider.calculate`` (same math as ``ti2vid_one_stage``), not the hand-rolled
    delta pair. STG block selection stays in the trainer validation config (``stg_blocks``;
    use [28] for LTX-2.3 per upstream, NOT the LTX-2.0 default [29]).

    ``modality_scale`` (upstream default 3.0, exposed on the CLI as ``--a2v-guidance-scale`` /
    ``--v2a-guidance-scale``) defaults to 1.0 = OFF here so every run predating 2026-07-29 stays
    bit-reproducible. It is now FULLY WIRED (2026-07-30): setting it non-1.0 makes
    ``_build_modality_perturbation_config`` run the 4th isolated-modality pass and feed it as
    ``uncond_modality``, matching upstream. Before this it was inert even if set, because
    ``uncond_modality`` was hard-wired to ``cond``.
    READ THIS BEFORE ENABLING IT ON A MASK-GATED RUN: the isolated-modality pass
    is the prediction with BOTH cross-attentions skipped in ALL blocks
    (``denoisers.py:120-133``: SKIP_A2V_CROSS_ATTN + SKIP_V2A_CROSS_ATTN, ``blocks=None``), and
    ``video_to_audio_attn`` is exactly where the spatial mask applies its log-bias. So the audio
    guider's ``(modality_scale - 1) * (cond - uncond_modality)`` term amplifies
    (mask-gated prediction - no-video-conditioning prediction): it is a second inference-time lever
    acting ON the selection mechanism, not beside it. The CLI help calls this a lipsync knob; for
    stem separation it is not. Enabling it on a headline number requires the modality 1.0 / 3.0 /
    mask-off-at-3.0 three-way, or the mask's contribution is confounded. Costs a 4th transformer
    pass per step (~33%)."""

    video_cfg_scale: float = 3.0
    audio_cfg_scale: float = 7.0
    stg_scale: float = 1.0
    rescale_scale: float = 0.7
    modality_scale: float = 1.0
    # CUSTOM NULL-BRANCH GUIDANCE: a third x0 branch whose
    # "null" is MECHANISTIC rather than a negative prompt, composed into the AUDIO
    # prediction only as custom_weight * (cond - null), inside the rescale.
    #   "nomix"     null = stems' audio_attn1 masked off the mix span (mix-anchored
    #               guidance: steers TOWARD content the committed mix supports).
    #   "scenetext" null = every span's text routed to the SCENE block (mix-artifact
    #               repulsion: steers AWAY from scene-generic content in the stem spans).
    #   "swapprompt" null = the two stems' text blocks CROSSED (stem0 span reads stem1's
    #               prompt and vice versa; scene unchanged): steers each stem AWAY from
    #               what the SIBLING's prompt would render in its span (anti-leakage).
    #   "base"      null = the FROZEN BASE model (LoRA delta zeroed on every token for
    #               that forward; autoguidance, Karras 2024): amplifies exactly what the
    #               separation LoRA learned vs the base's leakage-prone generic render.
    # Defaults keep both OFF; the branch activates only when custom_variant is set AND
    # custom_weight != 0, so every existing run stays byte-identical. Costs a third
    # transformer pass per step. Only _run_denoising_mix_leads implements it.
    #   "nomixlat"  null = the MIX SPAN'S LATENT replaced by its trajectory-fixed noise
    #               (declared at the step's sigma): the strongest "mix absent" contrast —
    #               removes the mix CONTENT, not just the stems' attention to it.
    #   "nosib"     null = each stem's attention to the SIBLING span masked (self + mix
    #               kept): the delta amplifies what inter-stem communication contributes
    #               (complement of "nomix").
    # custom_sigma_min/max restrict the null branch to a sigma window (content commits
    # at HIGH sigma; window [0.5, 1.0] guides only commitment steps and lets late steps
    # polish acoustics unguided).
    custom_variant: str | None = None
    custom_weight: float = 0.0
    custom_sigma_min: float = 0.0
    custom_sigma_max: float = 1.0
    off_canonical: tuple[str, ...] = ()
    """Names of the four pinned scalars this run sets away from LTX_2_3_PARAMS ON PURPOSE.

    The pin in ``joint_sep_sampler._run_denoising`` exists to catch SILENT DRIFT (this module is
    edited by more than one experiment), not to forbid a deliberate ablation. Naming a scalar here
    turns its hard failure into a loud warning and records the intent in the run's ``args.json``;
    anything NOT named still raises, and naming a scalar that is actually at its canonical value
    also raises, so this cannot become a blanket mute."""


class JointStemSampler(AVReferenceSampler):
    """Denoise [stem0 | stem1 (| mix)] jointly from noise with per-channel block-masked text.

    Per-prompt state, set by the entrypoint before each generate call:
        ``joint_audio_context``  concat positive audio context  [1, sum(S_i), D]
        ``joint_block_masks``    list of per-block binary key masks, each [1, S_i]
        ``joint_num_stems``      K
        ``joint_include_mix``    whether a mix span trails the stems
        ``joint_span_allow``     arm-K span allow matrix [S, S] (None = legacy full attention),
                                 taken verbatim from the TRAINING strategy so the sampled topology
                                 can never disagree with the trained one
        ``joint_dev_guidance``   ``DevGuidance`` for the non-distilled checkpoint, or None. When
                                 set, EVERY joint denoising pass (aligned included) routes through
                                 ``_run_denoising_mix_leads`` so the per-modality composition
                                 applies uniformly; None keeps the legacy single-scale
                                 CFGGuider/STGGuider path byte-identical (distilled-era repro).
    """

    joint_audio_context = None
    joint_block_masks: list | None = None
    joint_num_stems = 2
    joint_include_mix = False
    joint_span_allow: Tensor | None = None
    joint_dev_guidance: DevGuidance | None = None
    # OBSERVED VIDEO (localize.py separation mode, 2026-08-04): patchified REAL video
    # tokens [1, Tv, D]. When set, the video stream is presented fully clean with
    # denoise_mask = 0 on every token -- the loop's own conventions then keep it at
    # timestep 0 and clamped (the all-frames extension of first-frame conditioning;
    # mirrors the mix span's committed clamp on the audio side). None = generate video
    # as before (behavior unchanged).
    joint_observed_video: Tensor | None = None
    # SCHEDULE-TRACKING observed video (2026-08-04, same day): holding the video clean at
    # timestep 0 starves the t2v readout, whose signal lives in the HIGH-noise attention
    # regime (design-set decomposition: t2v gmed 0.65 -> 0.47 under the clean hold while
    # v2a was flat). With this flag the observed video is instead CLAMPED to
    # (1 - sigma_i) * x0_obs + sigma_i * eps_fixed at every grid point -- the video
    # analogue of the mix span's teacher-force clamp: still fully observed (its clean
    # content anchors every step), but presented at the step's own noise level, exactly
    # the probe regime the readout statistics were built on. First-frame-conditioned
    # tokens (denoise_mask 0) stay clean throughout, as in the probe.
    joint_observed_video_schedule: bool = False
    # PHASE-12 x0-GUIDANCE HOOK (2026-08-07): optional callable
    # (denoised_audio, num_stems, span_len, sigma) -> denoised_audio, applied to the
    # composed x0-hat right after parity, before the conditioning blend. Not used by the
    # paper;
    # None = exact current behavior.
    joint_x0_guidance = None
    # Set by _generate_joint_stems after every render (consumed by the two-phase pass).
    last_joint_audio_latent: Tensor | None = None
    last_joint_span_len: int | None = None

    @staticmethod
    def _build_modality_perturbation_config() -> BatchedPerturbationConfig:
        """The isolated-modality ("mod") pass: BOTH cross-attentions skipped in EVERY block.

        Byte-for-byte the perturbation set upstream builds for its own bimodal-guidance pass
        (``ltx_pipelines/utils/denoisers.py``, the ``"mod"`` entry), so enabling
        ``modality_scale`` reproduces the canonical LTX-2.3 pipeline rather than a look-alike.
        ``blocks=None`` means ALL blocks, not none -- STG's per-block list does not apply here.
        """
        return BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=[
            Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
            Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
        ])])

    @torch.no_grad()
    def generate(self, config, device="cuda"):
        device = torch.device(device) if isinstance(device, str) else device
        self._validate_config(config)
        # ARM POSMASK: drop any video-key bias the last TRAINING step stashed. The holder in
        # spatial_mask_gate is module-global and the gate wraps the same transformer the trainer
        # uses, so a leftover bias would be injected into every validation forward -- and because
        # the audio span layout is identical here, it would shape-match and apply with nothing
        # raised. Validation has no register draw and no per-sample boxes, so the correct state is
        # ungated; training re-sets the bias on its next step.
        clear_video_key_bias()
        reqs = [c for c in (getattr(self, "audio_conditionings", None) or [])
                if isinstance(c, JointStemGenerationRequest)]
        if reqs and reqs[0].two_phase:
            return self._generate_two_phase(config, device, reqs[0])
        if reqs:
            return self._generate_joint_stems(config, device, reqs[0])
        if self.joint_dev_guidance is not None:
            # The inherited path composes guidance with the single-scale config values --
            # under a dev checkpoint that silently drops the per-modality guidance this run
            # was configured for. No current caller reaches here (every validation prompt
            # carries a JointStemGenerationRequest); if one ever does, fail loudly.
            raise RuntimeError(
                "JointStemSampler.generate: non-joint render requested while joint_dev_guidance "
                "is set -- the inherited sampler would use single-scale guidance, not the dev "
                "per-modality composition. Route this render through a JointStemGenerationRequest "
                "or extend dev guidance to the inherited path first.")
        return super().generate(config, device)

    def _generate_two_phase(self, config, device, request: JointStemGenerationRequest):
        """Committed two-phase render: generate the scene, then extract its parts.

        Pass 1: MIX-ONLY joint render at the aligned schedule — the audio sequence is a
        single mix span (NO stem spans are generated), computed
        by pure base weights alongside the video, and nothing is decoded — only the mix
        latent is kept.
        Pass 2: stems re-denoise from fresh noise with pass 1's mix latent clamped clean at
        sigma 0 at every grid point (all-zeros track) — the trained sep-pattern condition.
        Nothing pre-existing enters; both passes share the caller's config/seed/contexts.
        """
        if not self.joint_include_mix:
            raise RuntimeError("two_phase requires include_mix: there is no mix span to commit")
        pass1 = JointStemGenerationRequest(
            ref_time_offset=request.ref_time_offset, mix_lead_alpha=1.0, mix_only=True,
            span_pe_layout=request.span_pe_layout)
        self._generate_joint_stems(config, device, pass1)
        lat = self.last_joint_audio_latent
        span_len = self.last_joint_span_len
        mix_tokens = lat[:, -span_len:].to(torch.float32)
        committed_track = torch.zeros(int(config.num_inference_steps) + 1)
        pass2 = JointStemGenerationRequest(
            ref_time_offset=request.ref_time_offset, mix_lead_alpha=1.0,
            teacher_force_mix_latent=mix_tokens,
            teacher_force_seed=request.teacher_force_seed,
            mix_sigma_track=committed_track,
            span_pe_layout=request.span_pe_layout)
        return self._generate_joint_stems(config, device, pass2)

    def _generate_joint_stems(self, config, device, request: JointStemGenerationRequest):
        if self.joint_audio_context is None or self.joint_block_masks is None:
            raise RuntimeError(
                "JointStemSampler.generate: joint_audio_context / joint_block_masks were not set "
                "for this prompt")
        mix_lead_alpha = float(request.mix_lead_alpha)
        if not 0.0 < mix_lead_alpha <= 1.0:
            raise ValueError(f"mix_lead_alpha must be in (0, 1], got {mix_lead_alpha}")
        if mix_lead_alpha < 1.0 and not self.joint_include_mix:
            raise RuntimeError(
                "mix_lead_alpha<1 requires include_mix: there is no mix span to lead")
        mix_sigma_track = request.mix_sigma_track
        if mix_sigma_track is not None and not self.joint_include_mix:
            raise RuntimeError(
                "mix_sigma_track requires include_mix: there is no mix span to schedule")
        mix_only = request.mix_only
        if mix_only and not self.joint_include_mix:
            raise RuntimeError("mix_only requires include_mix: the render would be empty")
        # Local span layout: mix_only (two-phase pass 1) drops the stem spans entirely.
        num_stems = 0 if mix_only else self.joint_num_stems

        v_pos, _a_pos_cached, v_neg, a_neg = self._get_prompt_embeddings(config, device)
        a_pos = self.joint_audio_context.to(device)
        if mix_only:
            # Pass 1 audio context = the SCENE text block only (the trailing block of the
            # per-span concat). With a single span and a single block there is nothing to
            # route, so no text-block bias is stashed either -- a 1024-wide bias would alias
            # the negative-prompt length, which install_text_block_gate forbids by design.
            # Result: upstream single-prompt render semantics, exactly the plain base pass.
            scene_w = int(self.joint_block_masks[-1].shape[1])
            a_pos = a_pos[:, -scene_w:]

        generator = torch.Generator(device=device).manual_seed(config.seed)
        noiser = GaussianNoiser(generator=generator)

        # Video: generated with optional first-frame conditioning (joint_denoise semantics).
        video_tools = self._create_video_latent_tools(config)
        video_clean = video_tools.create_initial_state(device=device, dtype=torch.bfloat16)
        if config.condition_image is not None:
            video_clean = self._apply_image_conditioning(
                video_clean, config.condition_image, config, device)
        if self.joint_observed_video is not None:
            obs = self.joint_observed_video.to(device=video_clean.latent.device,
                                               dtype=video_clean.latent.dtype)
            if obs.shape != video_clean.latent.shape:
                raise ValueError(
                    f"joint_observed_video shape {tuple(obs.shape)} != video token grid "
                    f"{tuple(video_clean.latent.shape)} -- encode the video at the "
                    "configured resolution/frame count first")
            if self.joint_observed_video_schedule:
                # schedule-tracking mode: keep the conditioning denoise_mask; the actual
                # per-step (1-s)x0 + s*eps clamp is applied inside the mirror loop, which
                # owns the sigma grid. The noiser call below is a placeholder state that
                # the grid-0 clamp immediately overwrites.
                video_clean = replace(video_clean, latent=obs, clean_latent=obs)
                video_state = noiser(latent_state=video_clean, noise_scale=1.0)
            else:
                video_clean = replace(video_clean, latent=obs, clean_latent=obs,
                                      denoise_mask=torch.zeros_like(
                                          video_clean.denoise_mask))
                video_state = video_clean
        else:
            video_state = noiser(latent_state=video_clean, noise_scale=1.0)

        # Audio: K stem spans (+ optional mix span), all noised, all denoised. Every span reuses the
        # audio grid; the mix span (last, when present) is shifted by ref_time_offset.
        audio_tools = self._create_audio_latent_tools(config)
        n_spans = num_stems + (1 if self.joint_include_mix else 0)
        span_clean = []
        span_noised = []
        for i in range(n_spans):
            clean = audio_tools.create_initial_state(device=device, dtype=torch.bfloat16)
            is_mix = self.joint_include_mix and i == n_spans - 1
            if is_mix and request.ref_time_offset != 0.0:
                clean = replace(clean, positions=clean.positions + request.ref_time_offset)
            if request.span_pe_layout == "concat_mix_first":
                # ARM CATPE, mirroring joint_stems.prepare_training_inputs exactly: one timeline,
                # mix first, sequence order untouched -- mix +0, stem k +(k+1)*D. Pitch from
                # tokens 1->2 (token 0 is causal-clamped; 0->1 reads 0.025 s, steady pitch 0.04).
                # mix_only (two-phase pass 1) falls out correctly: its single span IS the mix,
                # offset 0. Positions must be float32 -- `dtype` above applies to the LATENT only
                # (tools.py builds positions via get_patch_grid_bounds, no dtype arg), and a bf16
                # cast would quantize a 9.12 s offset by ~0.005 s.
                pos = clean.positions
                if pos.dtype != torch.float32:
                    raise RuntimeError(f"span PE offsets need float32 positions, got {pos.dtype}")
                if pos.shape[2] < 3:
                    raise RuntimeError(f"span of {pos.shape[2]} tokens too short to derive pitch")
                dt = float(pos[0, 0, 2].mean() - pos[0, 0, 1].mean())
                offset = 0.0 if is_mix else (i + 1) * (pos.shape[2] * dt)
                if offset != 0.0:
                    clean = replace(clean, positions=pos + offset)
            span_clean.append(clean)
            # Mix-leads: the mix span starts at its declared level alpha*sigma_0 = alpha, i.e.
            # alpha * noise (clean initial latent is zeros; same randn draw, same RNG stream) --
            # the level-alpha marginal under a zero-mean prior. Stems start at pure noise.
            # With an explicit track the same rule reads the track's own first grid point. The
            # alpha branch is kept EXACTLY as it was rather than folded into track[0]: the grid's
            # sigma_0 is 0.99999994, not 1.0, so track[0] = alpha*sigma_0 != alpha and arm F/G
            # sampling would stop being bit-identical.
            if is_mix and mix_sigma_track is not None:
                span_scale = float(mix_sigma_track.reshape(-1)[0])
            elif is_mix and mix_lead_alpha < 1.0:
                span_scale = mix_lead_alpha
            else:
                span_scale = 1.0
            span_noised.append(noiser(latent_state=clean, noise_scale=span_scale))
        span_len = span_clean[0].latent.shape[1]

        joint_clean = LatentState(
            latent=torch.cat([s.latent for s in span_clean], dim=1),
            denoise_mask=torch.cat([s.denoise_mask for s in span_clean], dim=1),
            positions=torch.cat([s.positions for s in span_clean], dim=2),
            clean_latent=torch.cat([s.clean_latent for s in span_clean], dim=1),
        )
        joint_state = LatentState(
            latent=torch.cat([s.latent for s in span_noised], dim=1),
            denoise_mask=joint_clean.denoise_mask,
            positions=joint_clean.positions,
            clean_latent=joint_clean.clean_latent,
        )

        # DIAGNOSTIC teacher-forced-mix probe (default off): GT mix tokens + the ONE fixed
        # noise draw the staggered loop will clamp the mix span with at every step.
        teacher_force_mix = None
        if request.teacher_force_mix_latent is not None:
            if not self.joint_include_mix:
                raise RuntimeError(
                    "teacher_force_mix_latent requires include_mix (no mix span to clamp)")
            x0_gt = request.teacher_force_mix_latent.to(device=device, dtype=torch.float32)
            if x0_gt.dim() == 2:
                x0_gt = x0_gt.unsqueeze(0)
            expected = (1, span_len, int(joint_state.latent.shape[2]))
            if tuple(x0_gt.shape) != expected:
                raise ValueError(
                    f"teacher_force_mix_latent shape {tuple(x0_gt.shape)} != {expected} -- "
                    "patchified GT mix tokens must match the mix span token count exactly")
            tf_gen = torch.Generator(device=device).manual_seed(int(request.teacher_force_seed))
            eps_fixed = torch.randn(x0_gt.shape, generator=tf_gen, device=device,
                                    dtype=torch.float32)
            teacher_force_mix = (x0_gt, eps_fixed)

        # Block-diagonal text bias for this generation's span lengths, then arm the gate.
        # mix_only runs with NO biases: the context is already the scene block alone and the
        # sequence is a single span, so both gates stay pass-through (None), matching the
        # frozen sampler's own context_mask=None convention for a plain single-prompt render.
        block_bias = None if mix_only else build_block_diagonal_text_bias(
            span_lens=[span_len] * n_spans,
            block_masks=[m.to(device) for m in self.joint_block_masks],
            dtype=a_pos.dtype, device=device)
        # Arm K: additive cross-span SELF-attention bias (None = legacy full attention, in which
        # case the gate stays a pass-through). The WRAPPER route is used here because BOTH
        # validation loops are reached from this call site -- the FROZEN _run_denoising (whose
        # audio Modality is built inside third_party) and our _run_denoising_mix_leads -- and the
        # bias is a property of the sequence layout, so it must apply identically on the positive,
        # CFG-negative and STG-perturbed passes. It does: the loops replace() only `context`, and
        # the wrapper is pass-agnostic.
        span_bias = None if mix_only else build_span_attention_bias(
            span_lens=[span_len] * n_spans,
            allow=self.joint_span_allow,
            batch=int(joint_state.latent.shape[0]),
            dtype=joint_state.latent.dtype, device=device)
        set_text_block_bias(block_bias)
        set_span_attention_bias(span_bias)
        # mix_only re-arms the two span-count-keyed GLOBAL gates for the 1-span layout (they
        # slice the audio token axis into num_spans parts and would cut into the mix span
        # otherwise): the A2V mix-only gate is vacuous (ALL audio is the mix) -> disabled;
        # the arm-N LoRA span gate, when armed, zeroes the delta on span 0 of 1 = the whole
        # sequence -> the pass is pure base computation, exactly the frozen-mix contract.
        # Snapshots restored in the finally, so pass 2 / non-two-phase renders are untouched.
        prev_a2v = dict(a2v_mix_gate._STATE)      # noqa: SLF001
        prev_lora = dict(lora_span_gate._STATE)   # noqa: SLF001
        if mix_only:
            a2v_mix_gate.clear_a2v_mix_gate()
            if prev_lora["enabled"]:
                lora_span_gate.set_lora_span_gate(num_spans=1, gated_span=0)
        try:
            # The per-token loop is taken for the arm-F stagger, for an explicit mix sigma
            # track, AND for the teacher-forced probe at any alpha: at alpha = 1 with no track
            # its per-token timesteps/Euler reduce exactly to the frozen shared-sigma loop, so
            # the only difference is the clamp.
            # Energy-parity guidance needs the per-token loop (it edits the composed x0-hat
            # before the Euler update); at alpha=1 with no track it is the frozen loop's exact
            # twin, so routing through it changes nothing but the correction itself.
            parity_decode = None
            if request.energy_parity_guidance:
                if not self.joint_include_mix:
                    raise RuntimeError(
                        "energy_parity_guidance requires include_mix (mix span is the anchor)")
                parity_tools = audio_tools
                parity_proto = span_clean[0]

                def parity_decode(tokens: Tensor) -> Tensor:
                    state = replace(parity_proto, latent=tokens.to(torch.bfloat16))
                    state = parity_tools.unpatchify(state)
                    return self._audio_decoder(state.latent)

                self._audio_decoder.to(device)

            # joint_dev_guidance forces the mirror loop for the ALIGNED schedule too: at
            # mix_lead_alpha=1.0 the span-sigma grid is identically sigma_i (shared-sigma
            # schedule, same math as the frozen _run_denoising), so routing through the mirror
            # changes ONLY the guidance composition -- per-modality dev guidance then applies
            # uniformly to every joint render instead of silently reverting to the legacy
            # single-scale path on aligned renders.
            if (mix_lead_alpha < 1.0 or mix_sigma_track is not None
                    or teacher_force_mix is not None or parity_decode is not None
                    or self.joint_dev_guidance is not None):
                video_state, joint_state = self._run_denoising_mix_leads(
                    config=config,
                    video_state=video_state,
                    audio_state=joint_state,
                    video_clean_state=video_clean,
                    audio_clean_state=joint_clean,
                    v_ctx_pos=v_pos,
                    a_ctx_pos=a_pos,
                    v_ctx_neg=v_neg,
                    a_ctx_neg=a_neg,
                    device=device,
                    num_stems=num_stems,
                    span_len=span_len,
                    mix_lead_alpha=mix_lead_alpha,
                    teacher_force_mix=teacher_force_mix,
                    mix_sigma_track=mix_sigma_track,
                    parity_decode=parity_decode,
                )
            else:
                video_state, joint_state = self._run_denoising(
                    config=config,
                    video_state=video_state,
                    audio_state=joint_state,
                    video_clean_state=video_clean,
                    audio_clean_state=joint_clean,
                    v_ctx_pos=v_pos,
                    a_ctx_pos=a_pos,
                    v_ctx_neg=v_neg,
                    a_ctx_neg=a_neg,
                    device=device,
                )
        finally:
            clear_text_block_bias()
            clear_span_attention_bias()
            if mix_only:
                a2v_mix_gate._STATE.update(prev_a2v)      # noqa: SLF001
                lora_span_gate._STATE.update(prev_lora)   # noqa: SLF001

        # Final denoised audio tokens [B, n_spans*span_len, D] — the two-phase committed
        # render reads the mix span out of this (direct latent handoff, no wav round-trip).
        self.last_joint_audio_latent = joint_state.latent.detach()
        self.last_joint_span_len = span_len

        if mix_only:
            # Two-phase pass 1: only the mix latent above is consumed (handed straight to
            # pass 2's clamp); decoding the video + mix here would be discarded work.
            return None, None

        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        video_output = self._decode_video(video_state, device, config.tiled_decoding)

        # Decode each span. Stems keyed generated_stem{k}; the mix span (B/C) keyed generated_mix.
        slots: dict[str, torch.Tensor] = {}
        for k in range(num_stems):
            slots[f"generated_stem{k}"] = self._decode_audio_span(
                joint_state, audio_tools, k * span_len, span_len, device)
        if self.joint_include_mix:
            slots["generated_mix"] = self._decode_audio_span(
                joint_state, audio_tools, num_stems * span_len, span_len, device)
        self.last_slot_audio = slots
        # Primary returned track (muxed into the mp4): the mix span in B/C, else the summed stems.
        if self.joint_include_mix:
            primary = slots["generated_mix"]
        else:
            stems = [slots[f"generated_stem{k}"] for k in range(num_stems)]
            t = min(s.shape[-1] for s in stems)
            primary = sum(s[..., :t] for s in stems)
        return video_output, primary

    def _run_denoising_mix_leads(  # noqa: PLR0913
        self,
        config,
        video_state: LatentState,
        audio_state: LatentState,
        video_clean_state: LatentState,
        audio_clean_state: LatentState,
        v_ctx_pos: Tensor,
        a_ctx_pos: Tensor,
        v_ctx_neg: Tensor | None,
        a_ctx_neg: Tensor | None,
        device: torch.device,
        num_stems: int,
        span_len: int,
        mix_lead_alpha: float,
        teacher_force_mix: tuple[Tensor, Tensor] | None = None,
        mix_sigma_track: Tensor | None = None,
        parity_decode=None,
    ) -> tuple[LatentState, LatentState]:
        """Arm-F staggered denoising: mix span at ``alpha * sigma_i``, stems at ``sigma_i``.

        Line-for-line mirror of the frozen ``ValidationSampler._run_denoising`` (same
        LTX2Scheduler grid, same CFG + STG x0-space composition, same conditioning clamp,
        same video Euler step) with exactly two changes, both audio-only:

          1. ``Modality.timesteps`` is per-token: ``sigma_i * span_sigma_scale`` where the
             scale is 1 on stem tokens and ``mix_lead_alpha`` on the mix span -- the (sigma,
             alpha*sigma) pairs the arm-F strategy trains on, at every step. ``Modality.sigma``
             stays the STEM sigma ``(B,)``, matching training (iclora_mix precedent).
             ``X0Model`` then converts velocity -> x0-hat with these per-token timesteps.
          2. The audio Euler update is ``per_token_euler_update`` (per-token sigma and dt;
             the frozen ``EulerDiffusionStep``/``to_velocity`` collapse sigma to a scalar via
             ``.item()``). Tokens at sigma 0 are held fixed.

        Both spans hit sigma = 0 together at the last grid point (grid ends at exactly 0), the
        mix having been alpha-ahead throughout, so the mix's coarse structure commits first.

        ARM-G SCHEDULE SWEEP: pass ``mix_sigma_track`` (``[K+1]``, same grid) to replace the
        multiplicative map with an arbitrary validated mix track -- a fixed-STEP lead, or a mix
        that reaches 0 early and is held there by ``per_token_euler_update``. Nothing else in the
        loop changes: the per-token timesteps and per-token Euler already accept any sigma pair,
        which is precisely why independent-per-span-sigma training turns the schedule into a free
        inference-time decision. ``None`` keeps the multiplicative map bit-for-bit.
        """
        if audio_state.latent.shape[1] != (num_stems + 1) * span_len:
            raise ValueError(
                f"audio length {audio_state.latent.shape[1]} != (num_stems+1)*span_len = "
                f"{(num_stems + 1) * span_len} -- mix-leads needs [stem0..stemK-1|mix]")

        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=config.num_inference_steps).to(device).float()
        stepper = EulerDiffusionStep()
        dev_g = self.joint_dev_guidance
        if dev_g is not None:
            # Non-distilled (dev) checkpoint: per-modality guidance composed by the upstream
            # MultiModalGuider (ti2vid_one_stage math). Same 3 passes/step as the legacy path.
            cfg_guider = stg_guider = None
            video_guider = MultiModalGuider(params=MultiModalGuiderParams(
                cfg_scale=dev_g.video_cfg_scale, stg_scale=dev_g.stg_scale,
                rescale_scale=dev_g.rescale_scale, modality_scale=dev_g.modality_scale))
            audio_guider = MultiModalGuider(params=MultiModalGuiderParams(
                cfg_scale=dev_g.audio_cfg_scale, stg_scale=dev_g.stg_scale,
                rescale_scale=dev_g.rescale_scale, modality_scale=dev_g.modality_scale))
            dev_need_neg = (dev_g.video_cfg_scale != 1.0 or dev_g.audio_cfg_scale != 1.0)
            dev_need_stg = dev_g.stg_scale != 0.0
            stg_perturbation_config = (
                self._build_stg_perturbation_config(config) if dev_need_stg else None)
            # Canonical LTX-2.3 BIMODAL guidance (modality_scale 3.0, upstream default). Costs a
            # 4th transformer pass per step (~33%): the isolated-modality prediction with BOTH
            # cross-attentions skipped in every block, matching the upstream "mod" pass in
            # ltx_pipelines/utils/denoisers.py. Without it the guider's
            # (modality_scale - 1) * (cond - uncond_modality) term is inert.
            dev_need_mod = video_guider.do_isolated_modality_generation() or \
                audio_guider.do_isolated_modality_generation()
            mod_perturbation_config = (
                self._build_modality_perturbation_config() if dev_need_mod else None)
            # CUSTOM NULL-BRANCH GUIDANCE (see DevGuidance): validate + precompute once.
            dev_custom = dev_g.custom_variant
            if dev_custom is not None and dev_g.custom_weight == 0.0:
                raise ValueError(
                    "custom_variant set with custom_weight 0 -- a no-op request; drop the "
                    "variant or set a non-zero weight (args.json must not misreport)")
            if dev_custom is None and dev_g.custom_weight != 0.0:
                raise ValueError("custom_weight set without custom_variant")
            if dev_custom is not None and num_stems == 0:
                # Two-phase pass 1 (mix_only) has no stem spans to guide; the variant
                # applies to pass 2. Skip the branch here rather than raise, so
                # committed-gen + variant works end to end (verifier finding 2026-08-07).
                dev_custom = None
            g_null_span_bias = g_null_ctx = g_null_text_bias = null_pert = None
            if dev_custom in ("nomix", "nosib"):
                if self.joint_span_allow is None or not self.joint_include_mix:
                    raise RuntimeError(
                        f"{dev_custom} guidance needs include_mix and a span allow matrix")
                null_allow = self.joint_span_allow.clone()
                if dev_custom == "nomix":
                    null_allow[:num_stems, num_stems] = 0.0  # stems no longer attend the mix
                else:  # nosib: sibling blocked, self + mix kept
                    for _i in range(num_stems):
                        for _j in range(num_stems):
                            if _i != _j:
                                null_allow[_i, _j] = 0.0
                g_null_span_bias = build_span_attention_bias(
                    span_lens=[span_len] * (num_stems + 1), allow=null_allow,
                    batch=int(audio_state.latent.shape[0]),
                    dtype=audio_state.latent.dtype, device=device)
            elif dev_custom == "scenetext":
                if not self.joint_block_masks or len(self.joint_block_masks) != num_stems + 1:
                    raise RuntimeError(
                        "scenetext guidance needs per-span block masks with a trailing "
                        "scene block")
                scene_w = int(self.joint_block_masks[-1].shape[1])
                scene_ctx = a_ctx_pos[:, -scene_w:]
                g_null_ctx = scene_ctx.repeat(1, num_stems + 1, 1)
                g_null_text_bias = build_block_diagonal_text_bias(
                    span_lens=[span_len] * (num_stems + 1),
                    block_masks=[self.joint_block_masks[-1].to(device)] * (num_stems + 1),
                    dtype=a_ctx_pos.dtype, device=device)
            elif dev_custom == "swapprompt":
                if not self.joint_block_masks or len(self.joint_block_masks) != num_stems + 1:
                    raise RuntimeError(
                        "swapprompt guidance needs per-span block masks with a trailing "
                        "scene block")
                if num_stems != 2:
                    raise RuntimeError("swapprompt is defined for exactly 2 stems")
                w0 = int(self.joint_block_masks[0].shape[1])
                w1 = int(self.joint_block_masks[1].shape[1])
                g_null_ctx = torch.cat(
                    [a_ctx_pos[:, w0:w0 + w1], a_ctx_pos[:, :w0], a_ctx_pos[:, w0 + w1:]],
                    dim=1)
                g_null_text_bias = build_block_diagonal_text_bias(
                    span_lens=[span_len] * (num_stems + 1),
                    block_masks=[self.joint_block_masks[1].to(device),
                                 self.joint_block_masks[0].to(device),
                                 self.joint_block_masks[-1].to(device)],
                    dtype=a_ctx_pos.dtype, device=device)
            elif dev_custom == "base":
                if not lora_span_gate._STATE["enabled"]:      # noqa: SLF001
                    raise RuntimeError(
                        "base (autoguidance) null needs the LoRA span gate installed "
                        "(frozen_mix_lora run) -- without it the null would equal cond")
            elif dev_custom == "nomixlat":
                if teacher_force_mix is None:
                    raise RuntimeError(
                        "nomixlat null needs the teacher-forced mix (separation regime): "
                        "it replaces the committed mix latent with its trajectory noise")
            elif dev_custom is not None:
                raise ValueError(f"unknown custom_variant {dev_custom!r}")
            if dev_custom is not None:
                # All-ones no-op masks; the localization capture hook checks
                # `perturbations is not None` BEFORE context width, so the null forward is
                # classified "perturbed" and never counted as a second positive pass.
                null_pert = BatchedPerturbationConfig(
                    perturbations=[PerturbationConfig.empty()])
        else:
            cfg_guider = CFGGuider(config.guidance_scale)
            stg_guider = STGGuider(config.stg_scale)
            stg_perturbation_config = (
                self._build_stg_perturbation_config(config) if stg_guider.enabled() else None)

        # Per-token sigma at every grid point, [K+1, 1, T_audio, 1]. With mix_sigma_track=None
        # this is exactly sigma_i * build_span_sigma_scale(...), the product the loop formed
        # inline before, so the arm-F/G multiplicative path is unchanged to the bit.
        span_sigma_grid = build_span_sigma_grid(
            sigmas=sigmas, num_stems=num_stems, span_len=span_len,
            mix_lead_alpha=mix_lead_alpha, mix_sigma_track=mix_sigma_track, device=device)

        # Teacher-forced-mix probe (diagnostic; None on every headline path): overwrite the
        # mix span with the GT mix noised at its declared level alpha*sigma, training's
        # convention (1-s)*x0 + s*eps with the trajectory-fixed eps. Applied to the INITIAL
        # state (sigma_0) and after every Euler update (sigma_{i+1}); the final grid point
        # (sigma = 0) leaves the clean GT mix latent in place.
        mix_start = num_stems * span_len

        def _clamp_mix_to_gt(state: LatentState, grid_index: int) -> LatentState:
            x0_gt, eps_fixed = teacher_force_mix
            # The mix span's DECLARED level at this grid point, read off the same per-token grid
            # the model is conditioned with -- identical to the old mix_lead_alpha * sigma on the
            # multiplicative path, and correct for an explicit track too.
            s = span_sigma_grid[grid_index, 0, -1, 0]
            clamped = ((1.0 - s) * x0_gt + s * eps_fixed).to(state.latent.dtype)
            return replace(state, latent=torch.cat(
                [state.latent[:, :mix_start], clamped], dim=1))

        if teacher_force_mix is not None:
            audio_state = _clamp_mix_to_gt(audio_state, 0)

        # SCHEDULE-TRACKING observed video: the video analogue of the mix clamp above --
        # (1 - sigma_i) * x0_obs + sigma_i * eps_fixed at every grid point, with the ONE
        # trajectory-fixed eps. Conditioned tokens (denoise_mask 0, e.g. the first frame)
        # stay clean, matching the probe regime the readout statistics were fit on.
        video_sched_clamp = None
        if self.joint_observed_video is not None and self.joint_observed_video_schedule:
            vgen = torch.Generator(device=device).manual_seed(int(config.seed) + 7919)
            v_obs = video_clean_state.latent.to(torch.float32)
            v_eps = torch.randn(v_obs.shape, generator=vgen, device=device,
                                dtype=torch.float32)
            v_mask = video_state.denoise_mask

            def video_sched_clamp(state: LatentState, grid_index: int) -> LatentState:
                s = sigmas[grid_index]
                lat = (1.0 - s) * v_obs + s * v_eps
                lat = lat * v_mask + v_obs * (1 - v_mask)
                return replace(state, latent=lat.to(state.latent.dtype))

            video_state = video_sched_clamp(video_state, 0)

        video = Modality(
            enabled=True,
            latent=video_state.latent,
            sigma=sigmas[0].repeat(video_state.latent.shape[0]),
            timesteps=video_state.denoise_mask,
            positions=video_state.positions,
            context=v_ctx_pos,
            context_mask=None,
        )
        audio = Modality(
            enabled=True,
            latent=audio_state.latent,
            sigma=sigmas[0].repeat(audio_state.latent.shape[0]),
            timesteps=audio_state.denoise_mask,
            positions=audio_state.positions,
            context=a_ctx_pos,
            context_mask=None,
        )

        self._transformer.to(device)
        x0_model = X0Model(self._transformer)

        with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
            for step_idx, sigma in enumerate(sigmas[:-1]):
                audio_sigma_tok = span_sigma_grid[step_idx] * audio_state.denoise_mask
                audio_sigma_next_tok = span_sigma_grid[step_idx + 1] * audio_state.denoise_mask

                video = replace(
                    video,
                    latent=video_state.latent,
                    sigma=sigma.repeat(video_state.latent.shape[0]),
                    timesteps=sigma * video_state.denoise_mask,
                    positions=video_state.positions,
                )
                audio = replace(
                    audio,
                    latent=audio_state.latent,
                    sigma=sigma.repeat(audio_state.latent.shape[0]),
                    timesteps=audio_sigma_tok,
                    positions=audio_state.positions,
                )

                pos_video, pos_audio = x0_model(video=video, audio=audio, perturbations=None)

                if dev_g is not None:
                    neg_video = neg_audio = None
                    if dev_need_neg and v_ctx_neg is not None:
                        video_neg = replace(video, context=v_ctx_neg)
                        audio_neg = replace(audio, context=a_ctx_neg)
                        neg_video, neg_audio = x0_model(
                            video=video_neg, audio=audio_neg, perturbations=None)
                    perturbed_video = perturbed_audio = None
                    if stg_perturbation_config is not None:
                        perturbed_video, perturbed_audio = x0_model(
                            video=video, audio=audio, perturbations=stg_perturbation_config)
                    mod_video = mod_audio = None
                    if mod_perturbation_config is not None:
                        mod_video, mod_audio = x0_model(
                            video=video, audio=audio, perturbations=mod_perturbation_config)
                    # CUSTOM NULL-BRANCH forward: swap exactly one module-global gate for
                    # this one sequential call, restore in finally (the outer finally still
                    # clears both holders on any exit). Video output discarded -- the custom
                    # delta is audio-only (video is conditioned clean in separation).
                    null_audio = None
                    in_custom_window = (dev_custom is not None and
                                        dev_g.custom_sigma_min <= float(sigma)
                                        <= dev_g.custom_sigma_max)
                    if not in_custom_window:
                        pass  # outside the sigma window: standard composition below
                    elif dev_custom in ("nomix", "nosib"):
                        _prev_span_bias = current_span_attention_bias()
                        set_span_attention_bias(g_null_span_bias)
                        try:
                            _null_video, null_audio = x0_model(
                                video=video, audio=audio, perturbations=null_pert)
                        finally:
                            set_span_attention_bias(_prev_span_bias)
                    elif dev_custom in ("scenetext", "swapprompt"):
                        _prev_text_bias = current_text_block_bias()
                        set_text_block_bias(g_null_text_bias)
                        try:
                            audio_null = replace(audio, context=g_null_ctx)
                            _null_video, null_audio = x0_model(
                                video=video, audio=audio_null, perturbations=null_pert)
                        finally:
                            set_text_block_bias(_prev_text_bias)
                    elif dev_custom == "base":
                        _prev_lora = dict(lora_span_gate._STATE)   # noqa: SLF001
                        lora_span_gate.set_lora_span_gate(
                            num_spans=num_stems + 1, gated_span="all")
                        try:
                            _null_video, null_audio = x0_model(
                                video=video, audio=audio, perturbations=null_pert)
                        finally:
                            lora_span_gate._STATE.update(_prev_lora)   # noqa: SLF001
                    elif dev_custom == "nomixlat":
                        _x0_gt, _eps_fixed = teacher_force_mix
                        null_lat = audio.latent.clone()
                        null_lat[:, mix_start:] = _eps_fixed.to(null_lat.dtype)
                        null_ts = audio.timesteps.clone()
                        # Declared at the GRID MAX sigma (~1): pure eps is only consistent
                        # with s=1 under the (1-s)x0 + s*eps convention, and arm-G's
                        # independent per-span sigma training makes (sigma_stem, ~1)
                        # in-distribution (audit F1, 2026-08-07).
                        null_ts[:, mix_start:] = float(sigmas[0])
                        audio_null = replace(audio, latent=null_lat, timesteps=null_ts)
                        _null_video, null_audio = x0_model(
                            video=video, audio=audio_null, perturbations=null_pert)
                    # Missing passes fall back to cond, zeroing that guidance term in
                    # MultiModalGuider.calculate (uncond_modality=cond => modality term 0 when
                    # modality_scale is 1.0).
                    denoised_video = video_guider.calculate(
                        cond=pos_video,
                        uncond_text=neg_video if neg_video is not None else pos_video,
                        uncond_perturbed=(perturbed_video if perturbed_video is not None
                                          else pos_video),
                        uncond_modality=mod_video if mod_video is not None else pos_video)
                    if null_audio is not None:
                        # Upstream 4-term composition + the null-branch delta, with the
                        # rescale applied ONCE over the whole sum (same op order as
                        # MultiModalGuider.calculate; the w->0 limit equals it exactly).
                        u = neg_audio if neg_audio is not None else pos_audio
                        p = perturbed_audio if perturbed_audio is not None else pos_audio
                        m = mod_audio if mod_audio is not None else pos_audio
                        custom_delta = dev_g.custom_weight * (pos_audio - null_audio)
                        # Mix-span rows never reach the output (sigma-0 Euler hold +
                        # clamp) but a nonzero delta there inflates pred.std() and
                        # globally shrinks the rescaled prediction (audit F2). Exactly
                        # zero for every variant except nomixlat; zeroing is universal.
                        custom_delta[:, num_stems * span_len:] = 0.0
                        pred = (pos_audio
                                + (dev_g.audio_cfg_scale - 1) * (pos_audio - u)
                                + dev_g.stg_scale * (pos_audio - p)
                                + (dev_g.modality_scale - 1) * (pos_audio - m)
                                + custom_delta)
                        if dev_g.rescale_scale != 0:
                            factor = pos_audio.std() / pred.std()
                            factor = (dev_g.rescale_scale * factor
                                      + (1 - dev_g.rescale_scale))
                            pred = pred * factor
                        denoised_audio = pred
                    else:
                        denoised_audio = audio_guider.calculate(
                            cond=pos_audio,
                            uncond_text=neg_audio if neg_audio is not None else pos_audio,
                            uncond_perturbed=(perturbed_audio if perturbed_audio is not None
                                              else pos_audio),
                            uncond_modality=mod_audio if mod_audio is not None else pos_audio)
                else:
                    denoised_video, denoised_audio = pos_video, pos_audio

                    if cfg_guider.enabled() and v_ctx_neg is not None:
                        video_neg = replace(video, context=v_ctx_neg)
                        audio_neg = replace(audio, context=a_ctx_neg)
                        neg_video, neg_audio = x0_model(
                            video=video_neg, audio=audio_neg, perturbations=None)
                        denoised_video = denoised_video + cfg_guider.delta(pos_video, neg_video)
                        denoised_audio = denoised_audio + cfg_guider.delta(pos_audio, neg_audio)

                    if stg_guider.enabled() and stg_perturbation_config is not None:
                        perturbed_video, perturbed_audio = x0_model(
                            video=video, audio=audio, perturbations=stg_perturbation_config)
                        denoised_video = denoised_video + stg_guider.delta(
                            pos_video, perturbed_video)
                        denoised_audio = denoised_audio + stg_guider.delta(
                            pos_audio, perturbed_audio)

                # Frozen-anchor energy-parity guidance (stemgen/energy_parity.py): nudge the
                # STEM spans' composed x0-hat until summed mel power matches the frozen mix
                # anchor's. Applied to the guided x0-hat, before conditioning blend + Euler.
                if parity_decode is not None:
                    from stemgen.energy_parity import parity_correct_stems
                    denoised_audio, parity_stats = parity_correct_stems(
                        denoised_audio, num_stems, span_len, parity_decode)
                    if step_idx % 10 == 0 or step_idx == len(sigmas) - 2:
                        print(f"[parity] step {step_idx} sigma {float(sigma):.3f}: "
                              f"r0 {parity_stats['r0']:+.3f} -> {parity_stats['r_final']:+.3f}",
                              flush=True)

                if self.joint_x0_guidance is not None and num_stems >= 2:
                    denoised_audio = self.joint_x0_guidance(
                        denoised_audio, num_stems, span_len, float(sigma))

                denoised_video = (
                    denoised_video * video_state.denoise_mask
                    + video_clean_state.latent.float() * (1 - video_state.denoise_mask))
                denoised_audio = (
                    denoised_audio * audio_state.denoise_mask
                    + audio_clean_state.latent.float() * (1 - audio_state.denoise_mask))

                video_state = replace(
                    video_state,
                    latent=stepper.step(
                        sample=video.latent, denoised_sample=denoised_video,
                        sigmas=sigmas, step_index=step_idx),
                )
                audio_state = replace(
                    audio_state,
                    latent=per_token_euler_update(
                        latent=audio.latent, denoised=denoised_audio,
                        sigma_tok=audio_sigma_tok, sigma_next_tok=audio_sigma_next_tok),
                )
                if teacher_force_mix is not None:
                    audio_state = _clamp_mix_to_gt(audio_state, step_idx + 1)
                if video_sched_clamp is not None:
                    video_state = video_sched_clamp(video_state, step_idx + 1)

                if self._sampling_context is not None:
                    self._sampling_context.advance_step()

        return video_state, audio_state
