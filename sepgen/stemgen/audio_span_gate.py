"""SepGen: ASYMMETRIC cross-span mask for the audio SELF-attention.

WHY. The joint audio stream is ``[stem0 | stem1 (| mix)]`` in ONE sequence, and today NOTHING
masks ``audio_attn1``: every token of every span attends to every token of every other span.
Measured consequence (``analysis/attnflow_probe.md``, arm C step-1500): **the mix span puts
0.617 of its attention mass on the two STEM spans** (uniform-chance line for two of three spans
= 0.667), i.e. only ~0.38 of the mix span's self-attention context is the mix itself. The frozen
base model was pretrained generating ONE audio stream whose self-attention context is 100 % that
stream. So the mix span -- the span the user has verified the base model renders well -- is
computed OFF-DISTRIBUTION by construction, and the tokens contaminating it are pure noise at
high sigma and collapsed/garbage in exactly our failure modes.

PRIOR CONFIRMATION IN THIS CODEBASE (phase 10_1, not a new hypothesis). The same bug was found
and fixed once already, on the same kind of concatenated audio sequence:
``nightrun/2026-07-20/p10_1ms_234415/ledger.md`` "13:05 -- user found a REAL design bug: the mix
slot was reading the stems". At LoRA step 0 (``lora_B = 0`` -> the model IS the base model, so
the output MUST be base quality) the validation audio was not base quality; root cause verified
as the hub mask's ``allow[0, :] = 1`` letting the mix slot read the noise stems. After forcing
the mix slot to attend only itself, the step-0 mix matched the base model's own single-stream
generation (spectral-flatness position +0.11 vs +0.13, energy -22.0 dB vs -22.0 dB). Phase 10_3's
``joint_stems.py`` never carried that mask over (its audio ``Modality`` leaves ``attention_mask``
unset, joint_stems.py:458-466), so the fix was silently regressed.

THE TOPOLOGY (rows = queries, cols = keys; mix is the LAST span here, unlike phase 10_1 where
it was slot 0):

        stem0 stem1  mix
  stem0 [  1    s     1  ]   s = ``allow_sibling`` (default 0)
  stem1 [  s    1     1  ]
  mix   [  0    0     1  ]   <- protected: exactly the single-stream task the base model knows

Combined with the block-diagonal TEXT mask (``text_block_gate.py``, a different module and a
different key axis), a protected mix span sees: its own 113 audio tokens + the scene caption +
the video keys -- token-for-token the base model's ordinary text+video-to-audio generation.

MECHANISM (verified against ltx_core, two interchangeable routes that produce IDENTICAL bias):

  * NATIVE: ``Modality.attention_mask`` ``[B, T, T]`` in [0, 1] ->
    ``TransformerArgsPreprocessor._prepare_self_attention_mask`` (transformer_args.py:151-180)
    converts it to the additive log-space bias ``[B, 1, T, T]`` (1 -> log 1 = 0, 0 ->
    ``finfo.min``) -> ``TransformerArgs.self_attention_mask`` (transformer_args.py:231, 244) ->
    ``audio_attn1(mask=audio.self_attention_mask)`` (transformer.py:304-310). Built ONCE per
    forward and shared by all 48 blocks (model.py:426). Reachable from repo-side code wherever
    WE construct the audio ``Modality`` (training: joint_stems.py:458; mix-leads validation:
    joint_stem_sampler.py:377).
  * WRAPPER (this module's ``install_audio_span_gate``): wraps every ``*audio_attn1`` forward and
    injects the pre-built additive bias when the call site passes ``mask=None`` -- which is every
    call site on the joint path today. Same proven pattern as ``text_block_gate`` (audio_attn2)
    and ``spatial_mask_gate`` (video_to_audio_attn). This is the route that also covers the
    FROZEN ``ValidationSampler._run_denoising``, whose audio ``Modality`` is built inside
    third_party and hardcodes ``attention_mask`` to the (absent) ``slot_adapter`` hook.

The two routes are numerically identical by construction: ``build_span_attention_bias`` emits
exactly 0.0 / ``torch.finfo(dtype).min``, which is what ``_prepare_self_attention_mask`` produces
from the {0, 1} allow matrix. The wrapper DEFERS whenever a mask is already present, so arming
both cannot double-apply.

CFG / STG. The bias is a property of the SEQUENCE LAYOUT, not of the prompt, so it must be
applied identically on the positive pass, the CFG negative pass and the STG-perturbed pass -- and
it is: the frozen loops build the audio ``Modality`` once and ``replace(...)`` only ``context``
for the negative pass (validation_sampler.py:594-596, joint_stem_sampler.py:415-418), and the
module-level wrapper is pass-agnostic. **There is NO length-routing hazard here**, unlike
``text_block_gate``: that gate must distinguish positive from negative by context length because
its KEY axis is the text axis and the negative prompt has a different token count. This gate's
key axis is the AUDIO token axis, which is identical on all three passes.

PER-TOKEN TIMESTEPS. Orthogonal. The staggered/independent sigma schedules (arms F/G) enter
through ``Modality.timesteps`` (per-token AdaLN) and never touch the attention mask; a protected
mix span at its own sigma is still protected.

BACKEND. A non-None mask routes ``Attention.forward`` to ``masked_attention_function``
(attention.py:544-547). ``LTX_MASKED_ATTENTION=sdpa`` is REQUIRED -- the xformers cutlass kernel
needs ``attn_bias.stride(-2) % 8 == 0`` and our audio length (3 x 113 = 339) violates it. Phase
10_3 already sets it unconditionally for the text gate (train_jointstem.py:24, and every probe
script), so this gate adds no new requirement. It must be set BEFORE the first masked-attention
resolution because ``automatic_masked_attention`` is ``functools.cache``d (attention.py:311-317).

LEGACY PARITY. ``topology="full"`` returns **None**, never an all-ones mask, so the unmasked
kernel path is taken and legacy runs stay byte-identical. (An all-ones mask is a mathematical
no-op and was measured bit-identical to the unmasked path on this host -- ledger.md "CONFOUND
RULED OUT", max abs diff 0.000e+00 -- but it still forces the masked kernel and costs speed.)
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

SpanAttentionTopology = Literal["full", "mix_protected"]


def build_span_allow_matrix(
    num_stems: int,
    include_mix: bool,
    topology: SpanAttentionTopology,
    allow_sibling: bool,
) -> Tensor | None:
    """Span-level allow matrix ``[S, S]`` (float32, 1 = attend, 0 = blocked), or None for "full".

    Row = querying span, column = key span. Span order is ``[stem0 .. stemK-1 (, mix)]`` -- the
    MIX IS LAST, matching ``joint_stems.prepare_training_inputs`` (joint_stems.py:385-388) and
    ``joint_stem_sampler._generate_joint_stems`` (joint_stem_sampler.py:188-198).

    "full" returns None deliberately (not an all-ones matrix): a mask forces the attention
    backend off the unmasked kernel, so the legacy topology must not pay for a no-op.

    Args:
        num_stems: K, the number of speaker stem spans.
        include_mix: whether a mix span trails the stems.
        topology: "full" = legacy unrestricted attention; "mix_protected" = the asymmetric
            "mix leads, stems follow" mask (mix attends only itself; stems attend own + mix).
        allow_sibling: whether stem queries may also attend the OTHER stem span(s). Only
            meaningful under "mix_protected".
    """
    if num_stems < 1:
        raise ValueError(f"num_stems must be >= 1, got {num_stems}")
    if topology == "full":
        return None
    if topology != "mix_protected":
        raise ValueError(f"unknown span attention topology: {topology!r}")
    if not include_mix:
        raise ValueError(
            "topology='mix_protected' requires include_mix=True: there is no mix span to "
            "protect, and without one the stems would attend only themselves (which is the "
            "'none' topology, not this method)")

    num_spans = num_stems + 1
    mix = num_spans - 1
    allow = torch.zeros(num_spans, num_spans, dtype=torch.float32)
    allow.fill_diagonal_(1.0)                                    # every span always sees itself
    allow[:mix, mix] = 1.0                                       # every stem may read the mix
    if allow_sibling:
        allow[:mix, :mix] = 1.0                                  # stems may also read each other
    # allow[mix, :mix] stays 0 -- THE point of this mask. The mix span's self-attention context
    # is then its own span alone, bit-for-bit the single-stream shape the base model was
    # pretrained on, so its generation is protected from noisy/collapsed stem tokens.
    return allow


def _expand_spans(allow: Tensor, span_lens: list[int]) -> Tensor:
    """Span-level ``[S, S]`` -> token-level ``[T, T]`` by repeating each span's rows/cols."""
    if allow.shape[0] != len(span_lens):
        raise ValueError(
            f"allow matrix is {allow.shape[0]}x{allow.shape[1]} but {len(span_lens)} span "
            "lengths were given -- one row/column per span is required")
    lens = torch.tensor(span_lens, dtype=torch.long)
    return allow.repeat_interleave(lens, dim=0).repeat_interleave(lens, dim=1)


def build_span_allow_mask(
    span_lens: list[int],
    allow: Tensor | None,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor | None:
    """Token-level ``Modality.attention_mask`` ``[B, T, T]`` in {0, 1}, or None when allow is None.

    This is the NATIVE route: hand the result to ``Modality(attention_mask=...)`` and
    ``TransformerArgsPreprocessor._prepare_self_attention_mask`` turns it into the additive
    log-space bias (0 / ``finfo.min``) that ``audio_attn1`` consumes.
    """
    if allow is None:
        return None
    mask = _expand_spans(allow, span_lens).to(device=device, dtype=dtype)
    _assert_no_dead_query_row(mask)
    return mask.unsqueeze(0).expand(batch, -1, -1)


def build_span_attention_bias(
    span_lens: list[int],
    allow: Tensor | None,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor | None:
    """Additive pre-softmax bias ``[B, 1, T, T]`` (0.0 allowed / ``finfo.min`` blocked), or None.

    This is the WRAPPER route (bypasses the preprocessor), and it is numerically IDENTICAL to
    what ``_prepare_self_attention_mask`` produces from ``build_span_allow_mask``: that function
    maps 1 -> log(1) = 0 and 0 -> ``torch.finfo(dtype).min``.

    Emitted at ``dtype`` (use the audio latent dtype). ``finfo(bfloat16).min`` is -3.39e38, the
    same exponent range as float32, so the blocked logits underflow to exactly zero probability.
    """
    if allow is None:
        return None
    dense = _expand_spans(allow, span_lens).to(device=device)
    _assert_no_dead_query_row(dense)
    neg = torch.finfo(dtype).min
    bias = torch.where(
        dense > 0,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), neg, dtype=dtype, device=device),
    )
    return bias.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1)


def _assert_no_dead_query_row(mask: Tensor) -> None:
    """Fail loud on a fully-masked query row -- softmax over an all-``-inf`` row is NaN."""
    dead = (mask > 0).sum(dim=-1) == 0
    if bool(dead.any()):
        raise ValueError(
            f"{int(dead.sum())} query rows attend to nothing; softmax would produce NaN. "
            "Every span must at least attend itself")


class _BiasHolder:
    """Module-global stash for the current forward's span bias (or None). One active forward at
    a time (single GPU, sequential steps), same assumption as ``text_block_gate`` /
    ``spatial_mask_gate``."""

    bias: Tensor | None = None


_HOLDER = _BiasHolder()


def set_span_attention_bias(bias: Tensor | None) -> None:
    _HOLDER.bias = bias


def clear_span_attention_bias() -> None:
    _HOLDER.bias = None


def current_span_attention_bias() -> Tensor | None:
    return _HOLDER.bias


def install_audio_span_gate(model: torch.nn.Module) -> list[str]:
    """Wrap every ``*audio_attn1`` forward to inject the stashed span bias.

    Injects ONLY when the call site passed ``mask=None`` (so the NATIVE
    ``Modality.attention_mask`` route wins if both are armed -- no double application) and only
    when the stashed bias matches the query length exactly; a mismatched length is a bug (a
    sequence that is not the joint span layout) and fails loud rather than masking the wrong
    tokens. With no bias stashed the wrapper is a pass-through, so installing it is safe on
    legacy arms.

    Idempotent; returns the wrapped module names. Fails loud if the transformer has no
    ``audio_attn1``. Video self-attention (``attn1``) is never touched.
    """
    wrapped: list[str] = []
    for name, module in model.named_modules():
        if not name.endswith("audio_attn1"):
            continue
        if getattr(module, "_audio_span_gated", False):
            continue
        orig_forward = module.forward

        def make(orig):
            def forward(x, context=None, mask=None, **kwargs):
                if mask is None:
                    bias = _HOLDER.bias
                    if bias is not None:
                        if bias.shape[-1] != x.shape[1] or bias.shape[-2] != x.shape[1]:
                            raise ValueError(
                                f"span bias is {tuple(bias.shape)} but audio_attn1 got "
                                f"{x.shape[1]} query tokens -- the stashed mask does not "
                                "describe this sequence")
                        if bias.shape[0] != x.shape[0]:
                            raise ValueError(
                                f"span bias batch {bias.shape[0]} != audio batch {x.shape[0]}")
                        mask = bias.to(dtype=x.dtype, device=x.device)
                return orig(x, context=context, mask=mask, **kwargs)
            return forward

        module.forward = make(orig_forward)
        module._audio_span_gated = True                           # noqa: SLF001
        module._audio_span_orig_forward = orig_forward            # noqa: SLF001
        wrapped.append(name)
    if not wrapped:
        raise RuntimeError(
            "install_audio_span_gate found no 'audio_attn1' modules -- the transformer has no "
            "audio self-attention to gate")
    return wrapped


def uninstall_audio_span_gate(model: torch.nn.Module) -> None:
    for _name, module in model.named_modules():
        if getattr(module, "_audio_span_gated", False):
            module.forward = module._audio_span_orig_forward      # noqa: SLF001
            del module._audio_span_gated
            del module._audio_span_orig_forward
    clear_span_attention_bias()
