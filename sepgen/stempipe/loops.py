"""Denoising loops for the joint audio sequence.

Upstream calls a loop with exactly six keyword arguments
(`sigmas, video_state, audio_state, stepper, transformer, denoiser`) and expects
`(video_state, audio_state)` back. We are handed `audio_state=None` on purpose: with no
audio spec, `DiffusionStage.__call__` skips its post-loop `clear_conditioning` +
`unpatchify`, so the joint state we return comes back RAW. Were an audio spec passed
instead, upstream would silently truncate our three spans to the length of one and hand
back stem0 relabelled as the mixture -- which is why the first thing each loop does is
assert the slot is empty.

`JointRes2sLoop` delegates every step of the actual sampling to upstream's res_2s loop, so
the RK coefficients, bong refinement and SDE injection are upstream's code, unmodified.
`JointEulerLoop` is ours only because staggered generation needs per-token sigmas, which
the stock scalar Euler step cannot express.
"""
from __future__ import annotations

from dataclasses import replace

import torch
from ltx_core.tools import AudioLatentTools
from ltx_pipelines.utils.samplers import res2s_audio_video_denoising_loop
from tqdm import tqdm

from .masks import per_token_euler_step, span_sigma_grid
from .stem_config import StemConfig


class _JointLoopBase:
    def __init__(self, *, stem: StemConfig, audio_tools: AudioLatentTools, noiser,
                 span_len: int, denoiser_ctx, device: torch.device, dtype: torch.dtype):
        self.stem = stem
        self.audio_tools = audio_tools
        self.noiser = noiser
        self.span_len = span_len
        self.denoiser_ctx = denoiser_ctx
        self.device = device
        self.dtype = dtype

    # -- state ---------------------------------------------------------------

    def _build_joint_state(self):
        """`[stem0 | stem1 | mix]`, mix LAST, one seeded noise draw per span in order.

        Each span is created from the SAME single-clip audio shape, so all three carry
        identical positions (the same 0..duration seconds). They are distinguished by the
        attention topology and the caption routing, never by their place on the time axis --
        that is what keeps every span inside the positional range the model was trained on.
        """
        spans = self.stem.num_spans
        # AudioLatentShape is a NamedTuple, so it replaces with _replace, not dataclasses.replace
        joint_shape = self.audio_tools.target_shape
        one = joint_shape._replace(frames=joint_shape.frames // spans)
        one_tools = AudioLatentTools(self.audio_tools.patchifier, one)

        # DRAW ORDER decides which sample each span gets. The layout is always
        # [stem0 | stem1 | mix], but with `mix_noise_first` the mixture's noise is drawn
        # first -- i.e. from the same position in the generator's stream that stock's single
        # audio span draws from -- so the mixture is the one stock would have produced.
        order = ([spans - 1] + list(range(spans - 1)) if self.stem.mix_noise_first
                 else list(range(spans)))
        states = [None] * spans
        for index in order:
            state = one_tools.create_initial_state(self.device, self.dtype)
            states[index] = self.noiser(latent_state=state, noise_scale=1.0)

        joint = replace(
            states[0],
            latent=torch.cat([s.latent for s in states], dim=1),
            clean_latent=torch.cat([s.clean_latent for s in states], dim=1),
            denoise_mask=torch.cat([s.denoise_mask for s in states], dim=1),
            positions=torch.cat([s.positions for s in states], dim=2),
        )
        self._assert_joint(joint, states)
        return joint

    def _assert_joint(self, joint, states) -> None:
        spans, span_len = self.stem.num_spans, self.span_len
        if joint.latent.shape[1] != spans * span_len:
            raise RuntimeError(
                f"joint audio is {joint.latent.shape[1]} tokens, expected {spans * span_len}")
        if float(joint.denoise_mask.min()) != 1.0:
            raise RuntimeError("joint audio denoise mask must be all ones (nothing is conditioning)")
        p = joint.positions
        for k in range(1, spans):
            if not torch.equal(p[..., :span_len, :], p[..., k * span_len:(k + 1) * span_len, :]):
                raise RuntimeError(f"span {k} has different positions than span 0")

    def __call__(self, *, sigmas, video_state, audio_state, stepper, transformer, denoiser):
        if audio_state is not None:
            raise RuntimeError(
                "this loop must be given audio_state=None; with an audio spec upstream would "
                "truncate the joint sequence to one span after the loop (tools.clear_conditioning)")
        joint = self._build_joint_state()
        self.denoiser_ctx.arm(joint)
        try:
            return self._run(sigmas=sigmas, video_state=video_state, joint=joint,
                             stepper=stepper, transformer=transformer, denoiser=denoiser)
        finally:
            self.denoiser_ctx.disarm()

    def _run(self, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError


class JointRes2sLoop(_JointLoopBase):
    """Second-order res_2s sampling, delegated to upstream."""

    def _run(self, *, sigmas, video_state, joint, stepper, transformer, denoiser):
        return res2s_audio_video_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=joint,
            stepper=stepper,
            transformer=transformer,
            denoiser=denoiser,
            noise_seed=-1,          # upstream's own default; the pipeline passes nothing
        )


class JointEulerLoop(_JointLoopBase):
    """First-order Euler with per-token sigmas, so the mix span can lead the sources.

    At `mix_lead_alpha == 1.0` with no explicit track this is exactly the stock Euler step
    (asserted by `tests/test_euler_reduction.py`).
    """

    def _run(self, *, sigmas, video_state, joint, stepper, transformer, denoiser):
        from ltx_pipelines.utils.helpers import post_process_latent

        grid = span_sigma_grid(
            sigmas, self.stem.num_spans, self.span_len,
            mix_lead_alpha=self.stem.mix_lead_alpha,
            mix_sigma_track=self.stem.mix_sigma_track, device=self.device,
        )
        for step_index in tqdm(range(len(sigmas) - 1), desc="euler"):
            v_res, a_res = denoiser(transformer, video_state, joint, sigmas, step_index)
            dv = post_process_latent(v_res.denoised, video_state.denoise_mask,
                                     video_state.clean_latent)
            da = post_process_latent(a_res.denoised, joint.denoise_mask, joint.clean_latent)
            video_state = replace(
                video_state,
                latent=stepper.step(video_state.latent, dv, sigmas, step_index))
            joint = replace(
                joint,
                latent=per_token_euler_step(joint.latent, da,
                                            grid[step_index] * joint.denoise_mask,
                                            grid[step_index + 1] * joint.denoise_mask))
        return video_state, joint
