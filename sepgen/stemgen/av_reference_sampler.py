"""Trainer-native sampling with synchronized video and audio references.

The frozen trainer sampler supports either its ref-first video IC-LoRA path or SepGen's
target-first audio reference hook, but its video-reference path never applies the audio
conditionings. This subclass composes both without changing the trainer's scheduler, CFG, STG,
or any file under ``third_party/``.

``fixed_reference`` is not video IC-LoRA: the encoded source video is the sole video modality,
with denoise_mask=0 for every token. Only the audio target is denoised.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.types import LatentState
from stemgen.ltx25_validation_sampler import GenerationConfig, ValidationSampler


class MixJointGenerationRequest:
    """Marker placed in ``audio_conditionings`` to switch the sampler into GENERATION mode.

    In generation mode the mix is NOT clamped clean (as separation does via
    ``AudioConditionByReferenceLatent``); instead the audio sequence is ``[stem | mix]`` with BOTH
    spans denoised from noise -- the mix-loss regime's generation task. Deliberately has no
    ``apply_to`` so that if it ever leaked into the conditioning-application loop it would fail loud
    rather than be silently mis-handled. ``ref_time_offset`` matches the training mix RoPE offset.
    """

    def __init__(self, ref_time_offset: float = 0.0):
        self.ref_time_offset = ref_time_offset


class AVReferenceSampler(ValidationSampler):
    """ValidationSampler with matched joint/dual/fixed video layouts."""

    video_context_mode = "joint_denoise"

    @torch.no_grad()
    def generate(self, config: GenerationConfig, device="cuda"):
        device = torch.device(device) if isinstance(device, str) else device
        self._validate_config(config)
        # Mix-loss GENERATION mode: a marker in audio_conditionings denoises mix + stem jointly from
        # noise (mix from t=1) instead of the separation path that clamps the mix clean. Detected
        # before any video-mode routing so it applies whatever the video layout is.
        gen_reqs = [c for c in (getattr(self, "audio_conditionings", None) or [])
                    if isinstance(c, MixJointGenerationRequest)]
        if gen_reqs:
            return self._generate_joint_mix(config, device, gen_reqs[0])
        if self.video_context_mode == "joint_denoise":
            return super().generate(config, device)
        if config.reference_video is None:
            raise ValueError(
                f"video_context_mode={self.video_context_mode!r} requires reference_video")
        if self.video_context_mode == "dual_reference":
            return self._generate_dual_reference(config, device)
        if self.video_context_mode == "fixed_reference":
            return self._generate_fixed_reference(config, device)
        raise ValueError(f"unknown video_context_mode: {self.video_context_mode!r}")

    def _apply_audio_references(self, audio_state, audio_clean_state, audio_tools):
        for conditioning in (getattr(self, "audio_conditionings", None) or []):
            audio_state = conditioning.apply_to(audio_state, audio_tools)
            audio_clean_state = conditioning.apply_to(audio_clean_state, audio_tools)
        return audio_state, audio_clean_state

    def _prepare_reference_video(self, config, device):
        # Validation reloads the same segment once per stem prompt. Re-encoding every prompt
        # fragments enough VRAM to OOM the dual layout, while pointer-based caching is unsafe
        # because the allocator can reuse a freed tensor's data_ptr for a different segment.
        # Hash the actual CPU pixels: identical content reuses its latent; different content
        # cannot alias merely because storage addresses were recycled.
        source = config.reference_video.detach().cpu().contiguous()
        cache_key = (
            hashlib.sha256(memoryview(source.numpy())).digest(),
            tuple(source.shape),
            str(source.dtype),
            config.reference_downscale_factor,
            config.frame_rate,
            str(device),
        )
        cache = getattr(self, "_video_reference_cache", None)
        if cache is None:
            cache = {}
            self._video_reference_cache = cache
        if cache_key in cache:
            return cache[cache_key]

        pixels = self._preprocess_reference_video(config)
        latent, positions = self._encode_video(pixels, config.frame_rate, device)
        if config.reference_downscale_factor != 1:
            positions = positions.clone()
            positions[:, 1, ...] *= config.reference_downscale_factor
            positions[:, 2, ...] *= config.reference_downscale_factor
        cache[cache_key] = (pixels, latent, positions)
        return cache[cache_key]

    def _prepare_audio_target(self, config, device, noiser):
        if not config.generate_audio:
            raise ValueError("mix-conditioned stem extraction requires generate_audio=True")
        audio_tools = self._create_audio_latent_tools(config)
        audio_clean_state = audio_tools.create_initial_state(
            device=device, dtype=torch.bfloat16)
        audio_state = noiser(latent_state=audio_clean_state, noise_scale=1.0)
        audio_state, audio_clean_state = self._apply_audio_references(
            audio_state, audio_clean_state, audio_tools)
        return audio_tools, audio_state, audio_clean_state

    def _decode_target_audio(self, audio_state, audio_tools, device):
        if audio_state is None:
            raise RuntimeError("audio state disappeared during denoising")
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)
        return self._decode_audio(audio_state, device)

    def _generate_dual_reference(self, config, device):
        """Denoise video target + audio target with clean ref-first video and mix references."""
        v_pos, a_pos, v_neg, a_neg = self._get_prompt_embeddings(config, device)
        generator = torch.Generator(device=device).manual_seed(config.seed)
        noiser = GaussianNoiser(generator=generator)

        _ref_pixels, ref_latent, ref_positions = self._prepare_reference_video(config, device)
        ref_seq_len = ref_latent.shape[1]
        video_tools = self._create_video_latent_tools(config)
        target_clean = video_tools.create_initial_state(device=device, dtype=torch.bfloat16)
        if config.condition_image is not None:
            target_clean = self._apply_image_conditioning(
                target_clean, config.condition_image, config, device)

        ref_mask = torch.zeros(
            1, ref_seq_len, 1, device=device, dtype=target_clean.denoise_mask.dtype)
        video_clean = LatentState(
            latent=torch.cat([ref_latent, target_clean.latent], dim=1),
            denoise_mask=torch.cat([ref_mask, target_clean.denoise_mask], dim=1),
            positions=torch.cat([ref_positions, target_clean.positions], dim=2),
            clean_latent=torch.cat([ref_latent, target_clean.clean_latent], dim=1),
        )
        # Noise only the target before concatenating the clean reference. GaussianNoiser draws
        # random values even where denoise_mask=0, so noising the combined state would consume an
        # extra reference-sized RNG block and give audio a different seed stream than the
        # historical joint_denoise path.
        target_state = noiser(latent_state=target_clean, noise_scale=1.0)
        video_state = LatentState(
            latent=torch.cat([ref_latent, target_state.latent], dim=1),
            denoise_mask=video_clean.denoise_mask,
            positions=video_clean.positions,
            clean_latent=video_clean.clean_latent,
        )
        audio_tools, audio_state, audio_clean = self._prepare_audio_target(
            config, device, noiser)

        video_state, audio_state = self._run_denoising(
            config=config,
            video_state=video_state,
            audio_state=audio_state,
            video_clean_state=video_clean,
            audio_clean_state=audio_clean,
            v_ctx_pos=v_pos,
            a_ctx_pos=a_pos,
            v_ctx_neg=v_neg,
            a_ctx_neg=a_neg,
            device=device,
        )
        target_latent = video_state.latent[:, ref_seq_len:]
        video_output = self._decode_video_latent(target_latent, config, device)
        audio_output = self._decode_target_audio(audio_state, audio_tools, device)
        return video_output, audio_output

    def _generate_fixed_reference(self, config, device):
        """Keep the sole source-video stream clean while denoising only target audio."""
        v_pos, a_pos, v_neg, a_neg = self._get_prompt_embeddings(config, device)
        generator = torch.Generator(device=device).manual_seed(config.seed)
        noiser = GaussianNoiser(generator=generator)

        ref_pixels, ref_latent, ref_positions = self._prepare_reference_video(config, device)
        ref_mask = torch.zeros(
            *ref_latent.shape[:2], 1, device=device, dtype=torch.float32)
        video_clean = LatentState(
            latent=ref_latent,
            denoise_mask=ref_mask,
            positions=ref_positions,
            clean_latent=ref_latent,
        )
        video_state = video_clean
        # Consume exactly the target-video RNG block used by joint_denoise. The fixed video
        # remains untouched, but the following audio target starts from the same random values
        # for the same validation seed in every layout.
        rng_alignment_state = self._create_video_latent_tools(config).create_initial_state(
            device=device, dtype=torch.bfloat16)
        noiser(latent_state=rng_alignment_state, noise_scale=1.0)
        audio_tools, audio_state, audio_clean = self._prepare_audio_target(
            config, device, noiser)

        video_state, audio_state = self._run_denoising(
            config=config,
            video_state=video_state,
            audio_state=audio_state,
            video_clean_state=video_clean,
            audio_clean_state=audio_clean,
            v_ctx_pos=v_pos,
            a_ctx_pos=a_pos,
            v_ctx_neg=v_neg,
            a_ctx_neg=a_neg,
            device=device,
        )
        if not torch.equal(video_state.latent, video_clean.latent):
            max_delta = (video_state.latent - video_clean.latent).abs().max().item()
            raise RuntimeError(
                f"fixed source video changed during audio denoising (max delta {max_delta:.3e})")

        # Return the actual source frames, not a VAE reconstruction or generated replacement.
        video_output = ((ref_pixels[0].cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
        audio_output = self._decode_target_audio(audio_state, audio_tools, device)
        return video_output, audio_output

    # ------------------------------------------------------------------ generation mode

    def _generate_joint_mix(self, config, device, request: MixJointGenerationRequest):
        """GENERATION: denoise [stem | mix] jointly from noise (mix from t=1) + text + first frame.

        This is the mix-loss regime's generation task -- neither audio span is clamped; both are
        denoised on the shared inference schedule. The two spans sit at IDENTICAL RoPE positions
        (time-aligned, ref_time_offset), exactly as training presents them, so at t>0 the model
        distinguishes them only by content + per-token timestep. The primary returned track is the
        generated MIX; the generated stem is stashed on ``last_slot_audio`` for the caller.

        Video follows the joint_denoise path (first-frame-conditioned generation), matching the
        separation validation so only the audio task differs between the two modes.
        """
        v_pos, a_pos, v_neg, a_neg = self._get_prompt_embeddings(config, device)
        generator = torch.Generator(device=device).manual_seed(config.seed)
        noiser = GaussianNoiser(generator=generator)

        # Video: generated with optional first-frame conditioning (joint_denoise semantics).
        video_tools = self._create_video_latent_tools(config)
        video_clean = video_tools.create_initial_state(device=device, dtype=torch.bfloat16)
        if config.condition_image is not None:
            video_clean = self._apply_image_conditioning(
                video_clean, config.condition_image, config, device)
        video_state = noiser(latent_state=video_clean, noise_scale=1.0)

        # Audio: stem span + mix span, both noised, both denoised. The mix span reuses the stem's
        # audio grid (same length/positions) shifted by ref_time_offset, matching training where the
        # mix (context) sits at the target's positions + ref_time_offset.
        audio_tools = self._create_audio_latent_tools(config)
        if getattr(request, "span_pe_layout", "aligned") != "aligned":
            # ARM CATPE is implemented only on the joint K-stem path (JointStemSampler); this
            # 2-span [stem | mix] variant has its own layout conventions and silently rendering
            # it under aligned PEs while training used concatenated ones is exactly the kind of
            # quiet divergence this repo keeps getting burned by.
            raise NotImplementedError(
                f"span_pe_layout={request.span_pe_layout!r} is not implemented on the "
                "AV-reference 2-span path; route this render through JointStemSampler")
        stem_clean = audio_tools.create_initial_state(device=device, dtype=torch.bfloat16)
        mix_clean = audio_tools.create_initial_state(device=device, dtype=torch.bfloat16)
        mix_clean = replace(mix_clean, positions=mix_clean.positions + request.ref_time_offset)
        stem_state = noiser(latent_state=stem_clean, noise_scale=1.0)
        mix_state = noiser(latent_state=mix_clean, noise_scale=1.0)
        stem_len = stem_clean.latent.shape[1]

        joint_clean = LatentState(
            latent=torch.cat([stem_clean.latent, mix_clean.latent], dim=1),
            denoise_mask=torch.cat([stem_clean.denoise_mask, mix_clean.denoise_mask], dim=1),
            positions=torch.cat([stem_clean.positions, mix_clean.positions], dim=2),
            clean_latent=torch.cat([stem_clean.clean_latent, mix_clean.clean_latent], dim=1),
        )
        joint_state = LatentState(
            latent=torch.cat([stem_state.latent, mix_state.latent], dim=1),
            denoise_mask=joint_clean.denoise_mask,
            positions=joint_clean.positions,
            clean_latent=joint_clean.clean_latent,
        )

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

        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        video_output = self._decode_video(video_state, device, config.tiled_decoding)

        stem_audio = self._decode_audio_span(joint_state, audio_tools, 0, stem_len, device)
        mix_audio = self._decode_audio_span(joint_state, audio_tools, stem_len, stem_len, device)
        # The generated stem rides out on last_slot_audio; the trainer's slot_audio_callback writes
        # and logs it. The generated MIX is primary (muxed into the mp4 + handed to the hook).
        self.last_slot_audio = {"generated_mix": mix_audio, "generated_stem": stem_audio}
        return video_output, mix_audio

    def _decode_audio_span(self, joint_state, audio_tools, start, length, device):
        """Decode one audio span [start:start+length] of a joint [stem | mix] latent to waveform."""
        span = LatentState(
            latent=joint_state.latent[:, start:start + length],
            denoise_mask=torch.ones_like(joint_state.denoise_mask[:, start:start + length]),
            positions=joint_state.positions[:, :, start:start + length],
            clean_latent=joint_state.clean_latent[:, start:start + length],
        )
        span = audio_tools.unpatchify(span)
        return self._decode_audio(span, device)
