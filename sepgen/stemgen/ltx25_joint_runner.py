"""Drive the joint-stem sampler from LTX-2.5's ValidationRunner lifecycle.

WHY THIS EXISTS. LTX-2 v1.2.0 deleted ``ltx_trainer.validation_sampler`` and replaced it with
``validation_runner.ValidationRunner``, which owns a different division of labour: the runner now
loads its own text encoder / VAEs, caches prompt embeddings and conditioning media at construction
time, and the trainer calls ``runner.run(transformer, step, ...)`` once per validation. The 2.3
trainer instead built a fresh ``ValidationSampler`` per validation and handed it live model handles,
which is the contract our ``JointStemSampler`` was written against.

Rather than rewrite 1,209 lines of joint-stem sampling against the new class, this bridges the two:
``ValidationRunner`` keeps everything it is good at (model lifecycle, prompt caching, output
writing, muxing, W&B, distributed work splitting) and we override exactly ONE seam --
``_generate_sample`` -- to denoise ``[stem0 | stem1 | mix]`` our way instead of upstream's single
audio sequence. Everything else in the validation path is upstream's, unmodified.

The per-sample joint state (block-masked audio context, the generation request, the arm's schedule)
is supplied by ``train_jointstem.py`` through ``audio_conditioning_provider`` -- the same hook name
and the same call signature the 2.3 build used, so the launcher's wiring is unchanged. Under 2.3
that hook was reached via an edit to ``third_party/``; here it is reached from our own subclass,
which is why the LTX-2.5 checkout carries no local patches at all.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from ltx_trainer.validation_runner import ValidationRunner

from stemgen.joint_stem_sampler import JointStemSampler
from stemgen.ltx25_validation_sampler import GenerationConfig


class JointStemValidationRunner(ValidationRunner):
    """``ValidationRunner`` whose per-sample generation is the joint multi-span audio pass.

    Class attributes are set by the launcher before the trainer is constructed (the trainer builds
    its runner inside ``__init__``, so instance-level wiring would be too late):

    ``audio_conditioning_provider``  ``(sample_index) -> list`` of conditioning items; it also
                                     stashes this cell's block-masked audio context on
                                     ``JointStemSampler`` as a side effect, exactly as under 2.3.
    ``slot_audio_callback``          optional consumer of the per-span waveforms, so a multi-stem
                                     render is auditioned stem by stem rather than only as a mux.
    ``validation_labels``            per-sample display names, used by the callback.
    """

    audio_conditioning_provider = None
    slot_audio_callback = None
    validation_labels: list[str] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # `_generate_sample` receives the sample OBJECT, not its index, but every hook the launcher
        # provides is index-addressed. Map by identity rather than by equality: two validation
        # cells may legitimately carry the same prompt and dims (e.g. one segment rendered under
        # two schedules), and `list.index` would then collapse them onto the first.
        self._sample_index = {id(s): i for i, s in enumerate(self._config.samples)}

    def _generate_sample(
        self,
        sample,
        cached_embeddings,
        cached_media,
        transformer,
        device: torch.device,
        sampling_ctx,
    ) -> tuple[Tensor | None, Tensor | None]:
        cfg = self._config
        idx = self._sample_index[id(sample)]
        width, height, num_frames = sample.video_dims or cfg.video_dims

        sampler = JointStemSampler(
            transformer=transformer,
            vae_decoder=self._vae_decoder,
            # The joint path conditions on a first frame that the runner has ALREADY encoded into
            # `cached_media`, so no encoder is needed here; passing one would invite a second,
            # divergent encode of the same pixels.
            vae_encoder=None,
            audio_decoder=self._audio_decoder,
            vocoder=self._vocoder,
            sampling_context=sampling_ctx,
        )
        # LTX-2.5's default video decoder is a DIFFUSION decoder that derives its minimum tile
        # overlaps from its own receptive fields and rejects anything smaller. The runner already
        # knows how to compute a valid layout (and sizes tiles against free VRAM); handing it over
        # keeps that geometry in one place instead of duplicating it in the vendored sampler.
        sampler.tiling_provider = self

        # First frame: taken from the runner's cache. `first_frame` is the only condition type the
        # joint validation cells carry (the legacy prompts+images config migrates to exactly that).
        condition_latent = None
        condition_pixels = None
        for cond_idx, cond in enumerate(sample.conditions):
            if cond.type != "first_frame":
                raise ValueError(
                    f"validation sample {idx} carries condition {cond.type!r}; the joint-stem "
                    "runner only knows how to present a first frame. Add explicit handling rather "
                    "than letting it be silently dropped.")
            media = cached_media.conditions.get(cond_idx)
            if media is None:
                raise ValueError(f"validation sample {idx}: first_frame condition has no cached media")
            condition_latent = media.latent.to(device)
            # The sampler gates first-frame conditioning on `condition_image is not None`; the
            # pixels are what that field means, and the latent below is what actually gets used.
            condition_pixels = media.pixels if media.pixels is not None else media.latent

        gen = GenerationConfig(
            prompt=sample.prompt,
            negative_prompt=cfg.negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=cfg.frame_rate,
            num_inference_steps=cfg.inference_steps,
            # Composition is handled per-modality by the arm's DevGuidance (video_cfg / audio_cfg /
            # rescale), which JointStemSampler routes every render through; this single-scale field
            # exists only to satisfy the shared config object and is deliberately neutral.
            guidance_scale=1.0,
            seed=sample.seed or cfg.seed,
            condition_image=condition_pixels,
            generate_audio=cfg.generate_audio,
            cached_embeddings=cached_embeddings,
            stg_scale=cfg.video_stg_scale,
            stg_blocks=cfg.stg_blocks,
        )
        # The first frame arrives as an already-encoded latent, so hand it over directly instead of
        # through `condition_image` (which would ask the sampler to encode pixels it has no encoder for).
        sampler.precomputed_condition_latent = condition_latent

        if self.audio_conditioning_provider is not None:
            sampler.audio_conditionings = self.audio_conditioning_provider(idx)

        video, audio = sampler.generate(gen, device)

        slots = getattr(sampler, "last_slot_audio", None)
        if self.slot_audio_callback is not None and slots is not None:
            # Signature is the launcher's, verbatim: it derives the segment name and its display
            # labels from prompt_index itself, so nothing extra is passed. `video_path` is accepted
            # but unused there; the muxed mp4 does not exist yet at this point in the runner's
            # lifecycle (it writes the file after _generate_sample returns), so None is the honest
            # value rather than a fabricated path.
            self.slot_audio_callback(
                step=self._current_step,
                prompt_index=idx,
                prompt=sample.prompt,
                video_path=None,
                slot_audio=slots,
                sample_rate=self._vocoder.output_sampling_rate if self._vocoder else None,
            )
        return video, audio

    def run(self, *args, **kwargs):
        # `_generate_sample` needs the step number for the per-step stem directories, and upstream
        # does not thread it down. Captured here rather than guessed.
        self._current_step = kwargs.get("step", args[1] if len(args) > 1 else 0)
        return super().run(*args, **kwargs)
