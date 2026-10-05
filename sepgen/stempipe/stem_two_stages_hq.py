"""Two-stage LTX-2.5 pipeline with an optional joint-stem add-on.

Structure, constructor and call inputs mirror upstream `ltx_pipelines.ti2vid_two_stages_hq`.
With `stem=None` this pipeline IS the stock pipeline -- same components, same order, same
sigma schedule, same stage 2, same decode -- which is both the design constraint and the
parity test (see `parity_check.py`). With a `StemConfig` it denoises a joint
`[stem0 | stem1 | mix]` audio sequence in one pass, and returns the separated spans.

Written from scratch against the upstream packages only.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.diffusion_steps import EulerDiffusionStep, Res2sDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import AUTO_TILING, AutoTiling, TilingConfig
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tools import AudioLatentTools
from ltx_core.types import Audio, AudioLatentShape, VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    DurationPredictor,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
    require_num_frames_source,
    resolve_num_frames,
)
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    ensure_tiling_config,
    generated_keyframe_conditionings,
    get_device,
    has_generated_keyframes,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.media_io import HDRColorSpace
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.samplers import res2s_audio_video_denoising_loop
from ltx_pipelines.utils.types import DEFAULT_AUTO_DURATION, AutoDuration, ModalitySpec, OffloadMode

from .prepared_stage import PreparedDiffusionStage
from .stem_config import StemConfig

logger = logging.getLogger(__name__)


@dataclass
class StemResult:
    """What a stem render produces on top of the stock (video, audio) pair."""

    spans: list[Audio]                 # decoded per span, in layout order: stem0, stem1, mix
    joint_latent: torch.Tensor         # PATCHIFIED [1, num_spans*span_len, 128]; mix is the LAST span
    span_len: int
    mix_index: int
    attention: object | None = None    # capture.AttentionCapture, when capture was armed

    @property
    def mix(self) -> Audio:
        return self.spans[self.mix_index]

    @property
    def stems(self) -> list[Audio]:
        return [a for i, a in enumerate(self.spans) if i != self.mix_index]


class StemTwoStagesHQPipeline:
    """Stock two-stage HQ pipeline; `stem` turns the joint-stem method on."""

    def __init__(  # noqa: PLR0913
        self,
        model_paths: ModelPaths,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        distilled_lora_strength_stage_1: float,
        distilled_lora_strength_stage_2: float,
        spatial_upsampler_path: str,
        loras: tuple[LoraPathStrengthAndSDOps, ...],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        prompt_enhancer_gemma_root: str | None = None,
        diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
        stem_transformer=None,          # X0Model, prepared by model_build.build_stem_transformer
        stem_builder=None,              # the builder it came from (for checkpoint/model_config)
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self._scheduler = LTX2Scheduler()

        distilled_lora_stage_1 = LoraPathStrengthAndSDOps(
            path=distilled_lora[0].path,
            strength=distilled_lora_strength_stage_1,
            sd_ops=distilled_lora[0].sd_ops,
        )
        distilled_lora_stage_2 = LoraPathStrengthAndSDOps(
            path=distilled_lora[0].path,
            strength=distilled_lora_strength_stage_2,
            sd_ops=distilled_lora[0].sd_ops,
        )

        self.prompt_encoder = PromptEncoder(
            model_paths, self.dtype, self.device, registry=registry, offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
            prompt_enhancer_gemma_root=prompt_enhancer_gemma_root,
        )
        self.image_conditioner = ImageConditioner(
            model_paths.video_vae(), self.dtype, self.device, registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.upsampler = VideoUpsampler(
            model_paths.video_vae(), spatial_upsampler_path, self.dtype, self.device,
            registry=registry, alloc_trim_strategy=alloc_trim_strategy,
        )
        self.video_decoder = VideoDecoder(
            model_paths.video_vae(), self.dtype, self.device, registry=registry,
            alloc_trim_strategy=alloc_trim_strategy, diffvae_optimization=diffvae_optimization,
        )
        self.audio_decoder = AudioDecoder(
            model_paths.audio_vae(), self.dtype, self.device, registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.duration_predictor = DurationPredictor.from_checkpoint(
            model_paths.duration_head_path, self.dtype, self.device,
        )

        stage_kwargs = dict(
            quantization=quantization, registry=registry, compilation_config=compilation_config,
            offload_mode=offload_mode, alloc_trim_strategy=alloc_trim_strategy,
        )
        if stem_transformer is not None:
            # Stage 1 runs OUR prepared model (base + fused distilled + separable adapter +
            # gates); stage 2 stays a stock stage on the stock checkpoint.
            self.stage_1 = PreparedDiffusionStage(
                stem_transformer, stem_builder, self.dtype, self.device)
        else:
            self.stage_1 = DiffusionStage.from_checkpoint(
                model_paths.transformer(), self.dtype, self.device,
                loras=(*loras, distilled_lora_stage_1), **stage_kwargs,
            )
        self.stage_2 = DiffusionStage.from_checkpoint(
            model_paths.transformer(), self.dtype, self.device,
            loras=(*loras, distilled_lora_stage_2), **stage_kwargs,
        )

    # ------------------------------------------------------------------ helpers

    def _audio_tools(self, num_frames: int, frame_rate: float, spans: int) -> AudioLatentTools:
        """Audio tools sized for `spans` concatenated spans of one clip's length.

        `spans=1` reproduces exactly what `DiffusionStage` builds internally.
        """
        one = AudioLatentShape.from_duration(1, num_frames / frame_rate)
        joint = AudioLatentShape(
            batch=one.batch, channels=one.channels, frames=one.frames * spans,
            mel_bins=one.mel_bins,
        )
        return AudioLatentTools(AudioPatchifier(patch_size=1), joint)

    # ------------------------------------------------------------------ call

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        audio_guider_params: MultiModalGuiderParams,
        images: list[ImageConditioningInput],
        num_frames: int | AutoDuration = DEFAULT_AUTO_DURATION,
        vae_dtype: torch.dtype | None = None,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        max_batch_size: int = 1,
        stage_1_sigmas: torch.Tensor | None = None,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        color_space: HDRColorSpace | None = None,
        generated_keyframes: int | Sequence[int] = 0,
        stem: StemConfig | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio, int, TilingConfig | None, StemResult | None]:
        require_num_frames_source(num_frames, self.duration_predictor)
        images = self.image_conditioner.resolve_crf(images)
        assert_resolution(height=height, width=width, is_two_stage=True)
        if has_generated_keyframes(generated_keyframes):
            self.stage_1.assert_generated_keyframes_supported()
        if stem is not None:
            stem.validate()

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16
        if vae_dtype is None:
            vae_dtype = dtype

        # --- text ---------------------------------------------------------------
        # Stem captions and every per-channel negative are encoded in the SAME call as the
        # scene prompt, so all blocks are padded to one width and the block-diagonal text
        # bias is exact.
        prompts = [prompt, negative_prompt]
        extra: dict[str, int] = {}
        stem_neg_idx: list[int] = []
        if stem is not None:
            prompts += list(stem.stem_prompts)
            for field in ("video_negative_prompt", "mix_negative_prompt"):
                text = getattr(stem, field)
                if text:
                    extra[field] = len(prompts)
                    prompts.append(text)
            for text in stem.resolved_stem_negatives():
                stem_neg_idx.append(len(prompts))
                prompts.append(text)
        encs = self.prompt_encoder(
            prompts,
            enhance_first_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
        )
        ctx_p, ctx_n = encs[0], encs[1]
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        num_frames = resolve_num_frames(
            num_frames, self.duration_predictor, video_encoding=v_context_p,
            audio_encoding=a_context_p, frame_rate=frame_rate,
        )

        scale_factors = tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path)
        tiling_config = ensure_tiling_config(
            tiling_config, scale_factors=scale_factors,
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(batch=1, frames=num_frames, height=height, width=width,
                                        fps=frame_rate),
            diffvae_optimization=self.video_decoder.diffvae_optimization, device=self.device,
        )

        # --- stage 1 ------------------------------------------------------------
        stage_1_output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=frame_rate)
        stage_1_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images, height=stage_1_output_shape.height,
                width=stage_1_output_shape.width, video_encoder=enc, dtype=dtype,
                device=self.device, color_space=color_space,
            )
        )
        stage_1_conditionings.extend(generated_keyframe_conditionings(generated_keyframes, num_frames))

        stepper = Res2sDiffusionStep()
        if stage_1_sigmas is None:
            # The latent argument is load-bearing: without it the scheduler falls back to
            # 4096 tokens and picks the wrong resolution-dependent sigma shift.
            empty_latent = torch.empty(
                VideoLatentShape.from_pixel_shape(
                    stage_1_output_shape, scale_factors=self.stage_1.video_scale_factors,
                ).to_torch_shape()
            )
            stage_1_sigmas = self._scheduler.execute(latent=empty_latent, steps=num_inference_steps)
        sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)

        stem_result: StemResult | None = None

        if stem is None:
            video_state, audio_state = self.stage_1(
                denoiser=GuidedDenoiser(
                    v_context=v_context_p, a_context=a_context_p,
                    video_guider=MultiModalGuider(params=video_guider_params,
                                                  negative_context=v_context_n),
                    audio_guider=MultiModalGuider(params=audio_guider_params,
                                                  negative_context=a_context_n),
                ),
                sigmas=sigmas, noiser=noiser, stepper=stepper,
                width=stage_1_output_shape.width, height=stage_1_output_shape.height,
                frames=num_frames, fps=frame_rate,
                video=ModalitySpec(context=v_context_p, conditionings=stage_1_conditionings),
                audio=ModalitySpec(context=a_context_p),
                loop=res2s_audio_video_denoising_loop,
                max_batch_size=max_batch_size,
            )
            final_audio_latent = audio_state.latent
        else:
            from .denoiser import JointStemDenoiser          # local: keeps the stock path clean
            from .loops import JointEulerLoop, JointRes2sLoop

            audio_tools = self._audio_tools(num_frames, frame_rate, stem.num_spans)
            span_len = audio_tools.target_shape.frames // stem.num_spans
            gates = getattr(self.stage_1.prepared, "stem_gates", None)
            if gates is None:
                raise RuntimeError(
                    "a StemConfig was given but stage 1 does not carry an installed gate set; "
                    "build the pipeline with model_build.build_stem_transformer(...)")

            stem_ctx = [encs[2 + k] for k in range(stem.num_stems)]
            neg_enc = {name: encs[idx] for name, idx in extra.items()}
            denoiser = JointStemDenoiser(
                stem=stem, gates=gates, span_len=span_len,
                v_context_p=v_context_p,
                v_context_n=(neg_enc["video_negative_prompt"].video_encoding
                             if "video_negative_prompt" in neg_enc else v_context_n),
                stem_encodings=stem_ctx, scene_encoding=ctx_p, negative_encoding=ctx_n,
                mix_negative_encoding=neg_enc.get("mix_negative_prompt"),
                stem_negative_encodings=[encs[i] for i in stem_neg_idx],
                video_guider_params=video_guider_params,
                audio_guider_params=audio_guider_params,
                sigmas=sigmas, device=self.device, dtype=dtype,
            )
            loop_cls = JointRes2sLoop if stem.sampler == "res_2s" else JointEulerLoop
            loop = loop_cls(stem=stem, audio_tools=audio_tools, noiser=noiser,
                            span_len=span_len, denoiser_ctx=denoiser, device=self.device,
                            dtype=dtype)
            if stem.sampler == "euler":
                stepper = EulerDiffusionStep()

            # --- attention capture (observational; must not change a byte of the render) --
            capture_handles = None
            if stem.capture is not None and stem.capture.enabled:
                from .capture import CaptureHandles

                latent_shape = VideoLatentShape.from_pixel_shape(
                    stage_1_output_shape, scale_factors=self.stage_1.video_scale_factors)
                n_blocks = gates.counts.get("capture")
                if not n_blocks:
                    raise RuntimeError(
                        "capture is enabled but no v2a sites were wrapped -- the transformer "
                        "was built without it (build_stem_transformer saw capture disabled)")
                # t2v replays each source's caption through the VIDEO text cross-attention,
                # so it needs the VIDEO-side encoding of each caption -- the same encodings
                # the audio spans read, taken from the other side of the connector.
                capture_handles = CaptureHandles(
                    grid=(latent_shape.frames, latent_shape.height, latent_shape.width),
                    num_spans=stem.num_spans, span_len=span_len, n_blocks=n_blocks,
                    streams=stem.capture.streams, band=stem.capture.band,
                    save_extra=stem.capture.save_extra, channels=stem.capture.channels,
                    tee_blocks=stem.capture.tee_blocks,
                    scene_context=v_context_p,
                    source_contexts=[e.video_encoding for e in stem_ctx],
                    source_masks=[getattr(e, "attention_mask", None) for e in stem_ctx],
                    device=self.device,
                )
                gates.capture = capture_handles
                if "t2v" in stem.capture.channels:
                    from .capture import install_adaln_probe

                    install_adaln_probe(gates)

            video_state, joint_state = self.stage_1(
                denoiser=denoiser, sigmas=sigmas, noiser=noiser, stepper=stepper,
                width=stage_1_output_shape.width, height=stage_1_output_shape.height,
                frames=num_frames, fps=frame_rate,
                video=ModalitySpec(context=v_context_p, conditionings=stage_1_conditionings),
                audio=None,                                  # THE SEAM -- see loops.py
                loop=loop, max_batch_size=max_batch_size,
            )
            attention = None
            if capture_handles is not None:
                attention = capture_handles.finish()
                gates.capture = None                 # disarm before stage 2 / the decodes
                logger.info("attention captured: %s", attention.summary())

            joint_latent = joint_state.latent                # raw, patchified, 3 spans
            expected = (1, span_len * stem.num_spans, joint_latent.shape[-1])
            if tuple(joint_latent.shape) != expected:
                raise RuntimeError(
                    f"joint audio came back as {tuple(joint_latent.shape)}, expected {expected} "
                    "-- upstream truncated it, which happens when `audio` is not None")

            one_tools = self._audio_tools(num_frames, frame_rate, 1)
            spans = [self._decode_span(joint_latent, one_tools, k, span_len)
                     for k in range(stem.num_spans)]
            mix_index = stem.num_spans - 1
            stem_result = StemResult(spans=spans, joint_latent=joint_latent.detach().cpu(),
                                     span_len=span_len, mix_index=mix_index,
                                     attention=attention)
            # The delivered audio is the mix span; stage 2 refines video only.
            final_audio_latent = one_tools.unpatchify(
                _span_state(joint_latent, one_tools, mix_index, span_len)).latent

        # --- stage 2 ------------------------------------------------------------
        upscaled_video_latent = self.upsampler(video_state.latent[:1])
        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width,
                                               height=height, fps=frame_rate)
        stage_2_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images, height=stage_2_output_shape.height,
                width=stage_2_output_shape.width, video_encoder=enc, dtype=dtype,
                device=self.device, color_space=color_space,
            )
        )
        if isinstance(self.stage_1, PreparedDiffusionStage):
            self.stage_1.to_cpu()        # 42 GB must not sit on the card through stage 2 / decode

        video_state, _ = self.stage_2(
            denoiser=SimpleDenoiser(v_context=v_context_p, a_context=a_context_p),
            sigmas=stage_2_sigmas, noiser=noiser, stepper=Res2sDiffusionStep(),
            width=width, height=height, frames=num_frames, fps=frame_rate,
            video=ModalitySpec(context=v_context_p, conditionings=stage_2_conditionings,
                               noise_scale=stage_2_sigmas[0].item(),
                               initial_latent=upscaled_video_latent),
            audio=ModalitySpec(context=a_context_p, noise_scale=stage_2_sigmas[0].item(),
                               initial_latent=final_audio_latent),
            loop=res2s_audio_video_denoising_loop,
        )

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator,
                                           dtype=vae_dtype)
        decoded_audio = self.audio_decoder(final_audio_latent)
        return decoded_video, decoded_audio, num_frames, tiling_config, stem_result

    # ------------------------------------------------------------------ decode

    def _decode_span(self, joint_latent: torch.Tensor, one_tools: AudioLatentTools,
                     index: int, span_len: int) -> Audio:
        state = _span_state(joint_latent, one_tools, index, span_len)
        return self.audio_decoder(one_tools.unpatchify(state).latent)


def _span_state(joint_latent: torch.Tensor, one_tools: AudioLatentTools, index: int,
                span_len: int):
    """A single-span LatentState carved out of the joint sequence, for unpatchify/decode."""
    from ltx_core.types import LatentState

    lo, hi = index * span_len, (index + 1) * span_len
    span = joint_latent[:, lo:hi]
    return LatentState(
        latent=span,
        clean_latent=torch.zeros_like(span),
        denoise_mask=torch.ones(span.shape[0], span_len, 1, device=span.device,
                                dtype=torch.float32),
        positions=one_tools.patchifier.get_patch_grid_bounds(one_tools.target_shape).to(span.device),
    )
