"""Attention-site wrappers that turn a stock transformer into the stem transformer.

Five gates, installed in a fixed order on the module tree. They hold no module-level state:
everything per-call lives on a `GateHandles` instance that the denoiser owns and mutates,
so two pipelines could run in one process without stepping on each other.

  1. text     -- block-diagonal caption routing on `audio_attn2`
  2. span     -- cross-span topology on `audio_attn1` (this is also beta's carrier)
  3. lora     -- zero our adapter's delta on the mix span (the mix stays frozen-base)
  4. a2v      -- video reads only the mix span of the audio sequence
  5. nag      -- normalized attention guidance, LAST so it wraps the routed forward

Upstream's attention signature (attention.py:520) is
`forward(x, context=None, mask=None, pe=None, k_pe=None, perturbation_mask=None,
all_perturbed=False)`; a non-None `mask` routes to the masked kernel, `None` keeps the
unmasked one. Every gate below defers when the caller already supplied a mask.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class GateHandles:
    """Per-call state shared by the installed wrappers.

    PASS IDENTITY IS DECLARED, NOT INFERRED. An earlier design had the gates work out which
    guidance pass they were in by comparing the text context's width -- positive concat vs
    negative -- which is silent-failure bait: two passes that happen to share a width get
    each other's routing and the render is subtly wrong with nothing raised. Instead the
    denoiser announces every forward through `pass_context(...)`, and the wrappers act only
    on what was declared. The width is then only ever used as an ASSERTION that the declared
    bias matches the tensor it is about to be applied to.
    """

    num_spans: int
    span_len: int
    a2v_mode: str = "slice"

    # set per forward by the denoiser via pass_context(); None => this pass routes nothing
    text_bias: torch.Tensor | None = None      # [B,1,T,K_text] (B rows = batched passes)
    nag_active: bool = False                   # whether NAG participates in THIS pass
    nag_rows: torch.Tensor | None = None       # [B] 1/0 per batched pass, when batching

    span_bias: torch.Tensor | None = None      # [1,1,T,T] cross-span topology (+ beta)
    span_base: torch.Tensor | None = None      # the armed base, for beta restore/verify

    nag_scale: float = 2.0
    nag_tau: float = 2.5
    nag_alpha: float = 0.5
    nag_null_bias: torch.Tensor | None = None  # sibling-complement bias
    nag_configured: bool = False               # NAG is armed for this cell at all

    # Attention capture (stempipe.capture.CaptureHandles) when armed for this cell. It rides
    # the same declaration so a pass is announced ONCE, to everything that cares about which
    # pass it is -- there is no second place that could disagree.
    capture: object | None = None

    counts: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def pass_context(self, *, text_bias: torch.Tensor | None, nag: bool = False,
                     nag_rows: torch.Tensor | None = None, capture: str | None = None):
        """Declare what the next forward is, for its duration.

        `nag_rows` is for a batched forward: one flag per row, so NAG applies to the rows
        that asked for it and leaves the others exactly as they were. `capture` names which
        stream this forward's attention belongs to ("main", "ti"), or None to record nothing
        -- negative and modality passes are simply never named.
        """
        prev = (self.text_bias, self.nag_active, self.nag_rows)
        self.text_bias = text_bias
        armed = self.nag_configured and self.nag_null_bias is not None
        self.nag_active = bool((nag or nag_rows is not None) and armed)
        self.nag_rows = nag_rows if armed else None
        recording = self.capture is not None and self.capture.wants(capture)
        if recording:
            self.capture.begin_pass(capture)
        try:
            yield
        finally:
            if recording:
                self.capture.end_pass()
            self.text_bias, self.nag_active, self.nag_rows = prev

    @property
    def total_tokens(self) -> int:
        return self.num_spans * self.span_len

    @property
    def mix_lo(self) -> int:
        return (self.num_spans - 1) * self.span_len

    def restore_span_base(self) -> None:
        self.span_bias = None if self.span_base is None else self.span_base.clone()

    def assert_span_base_intact(self) -> None:
        """Catches beta compounding across cells: the base must never absorb a boost."""
        if self.span_base is None:
            return
        if self.span_bias is not None and self.span_bias.shape != self.span_base.shape:
            raise RuntimeError("span bias shape drifted from its armed base")


def _wrap(module, name: str, make):
    original = module.forward
    if getattr(module, f"_stem_{name}", False):
        raise RuntimeError(f"{name} gate is already installed on this module")
    module.forward = make(original)
    setattr(module, f"_stem_{name}", True)
    return original


# --------------------------------------------------------------------------- 1. text


def _install_text(model, h: GateHandles) -> int:
    n = 0
    for mod_name, module in model.named_modules():
        if not mod_name.endswith("audio_attn2"):
            continue

        def make(original):
            def forward(x, context=None, mask=None, pe=None, k_pe=None, **kw):
                if mask is None and h.text_bias is not None and context is not None:
                    bias = h.text_bias
                    if bias.shape[0] not in (1, x.shape[0]):
                        raise RuntimeError(
                            f"declared text bias has {bias.shape[0]} rows for a batch of "
                            f"{x.shape[0]}")
                    if bias.shape[-1] != context.shape[1] or bias.shape[-2] != x.shape[1]:
                        raise RuntimeError(
                            f"declared text bias {tuple(bias.shape)} does not fit this pass "
                            f"(queries {x.shape[1]}, text keys {context.shape[1]}) -- the "
                            "denoiser declared the wrong pass for this forward")
                    mask = bias.to(dtype=x.dtype, device=x.device)
                return original(x, context=context, mask=mask, pe=pe, k_pe=k_pe, **kw)
            return forward

        _wrap(module, "text", make)
        n += 1
    if n == 0:
        raise RuntimeError("no audio_attn2 modules found; wrong checkpoint?")
    h.counts["text"] = n
    return n


# --------------------------------------------------------------------------- 2. span


def _install_span(model, h: GateHandles) -> int:
    n = 0
    for mod_name, module in model.named_modules():
        if not mod_name.endswith("audio_attn1"):
            continue

        def make(original):
            def forward(x, context=None, mask=None, pe=None, k_pe=None, **kw):
                if (mask is None and h.span_bias is not None
                        and x.shape[1] == h.span_bias.shape[-1]):
                    mask = h.span_bias.to(dtype=x.dtype, device=x.device)
                return original(x, context=context, mask=mask, pe=pe, k_pe=k_pe, **kw)
            return forward

        _wrap(module, "span", make)
        n += 1
    if n == 0:
        raise RuntimeError("no audio_attn1 modules found; wrong checkpoint?")
    h.counts["span"] = n
    return n


# --------------------------------------------------------------------------- 3. lora


def _install_lora(model, h: GateHandles) -> int:
    """Zero our adapter's contribution on the mix span.

    The mix must be computed by the frozen base so that the mixture this method decomposes
    is the one the base model would have produced. Implemented by recomputing ONLY the mix
    rows through `base_layer` and overwriting them -- exact, and costs a third of one
    projection rather than a second full forward.
    """
    try:
        from peft.tuners.lora import LoraLayer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("peft is required for the adapter span gate") from exc

    n = 0
    for mod_name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        # A module whose KEY/VALUE axis is the audio sequence cannot be span-gated on its
        # output rows; refuse rather than silently mis-gate.
        if mod_name.endswith(("audio_attn2.to_k", "audio_attn2.to_v",
                              "video_to_audio_attn.to_k", "video_to_audio_attn.to_v")):
            raise RuntimeError(
                f"{mod_name} is adapted but its token axis is keys, not queries; the span "
                "gate cannot express that. Remove it from the adapter's target modules.")

        def make(original, base_layer=module.base_layer):
            def forward(x, *args, **kw):
                out = original(x, *args, **kw)
                if x.dim() == 3 and x.shape[1] == h.total_tokens:
                    mix = base_layer(x[:, h.mix_lo:])
                    out = torch.cat([out[:, :h.mix_lo], mix.to(out.dtype)], dim=1)
                return out
            return forward

        _wrap(module, "lora", make)
        n += 1
    if n == 0:
        raise RuntimeError(
            "no PEFT LoraLayer modules found -- the adapter was fused instead of attached, "
            "so its delta cannot be gated off the mix span")
    h.counts["lora"] = n
    return n


# --------------------------------------------------------------------------- 4. a2v


def _install_a2v(model, h: GateHandles) -> int:
    """Video attends only to the mix span.

    `slice` (default) cuts the keys/values and their positional embeddings to the mix span,
    which is the same softmax as masking the stems out but keeps the UNMASKED attention
    kernel -- masking forces every block onto the masked path, a measured picture-quality
    suspect. `mask` keeps the old additive-bias behaviour for A/B.
    """
    if h.a2v_mode == "off":
        h.counts["a2v"] = 0
        return 0
    n = 0
    for mod_name, module in model.named_modules():
        if not mod_name.endswith("audio_to_video_attn"):
            continue

        def make(original):
            def forward(x, context=None, mask=None, pe=None, k_pe=None, **kw):
                if context is not None and context.shape[1] == h.total_tokens:
                    if h.a2v_mode == "slice":
                        context = context[:, h.mix_lo:]
                        if k_pe is not None:
                            k_pe = tuple(t[:, :, h.mix_lo:] for t in k_pe)
                    elif mask is None:
                        bias = torch.zeros(1, 1, 1, h.total_tokens, dtype=x.dtype,
                                           device=x.device)
                        bias[..., :h.mix_lo] = torch.finfo(x.dtype).min
                        mask = bias
                return original(x, context=context, mask=mask, pe=pe, k_pe=k_pe, **kw)
            return forward

        _wrap(module, "a2v", make)
        n += 1
    if n == 0:
        raise RuntimeError("no audio_to_video_attn modules found; wrong checkpoint?")
    h.counts["a2v"] = n
    return n


# --------------------------------------------------------------------------- 5. nag


def _install_nag(model, h: GateHandles) -> int:
    """Normalized attention guidance, in attention-feature space.

    Z+ is the routed positive attention (each span reading its own caption); Z- is the same
    tokens reading the SIBLING captions. The guided feature is pushed away from the sibling
    and renormalised so its magnitude cannot run away:

        G = Z+ * s - Z- * (s - 1);  G *= min(||G||1/||Z+||1, tau) / ratio;
        out = G * alpha + Z+ * (1 - alpha)

    Installed LAST so `original` here is the text-gated forward.
    """
    n = 0
    for mod_name, module in model.named_modules():
        if not mod_name.endswith("audio_attn2"):
            continue

        def make(original):
            def forward(x, context=None, mask=None, pe=None, k_pe=None, **kw):
                # participates only in passes the denoiser declared it for
                fires = (h.nag_active and mask is None and context is not None
                         and x.shape[1] == h.total_tokens)
                z_pos = original(x, context=context, mask=mask, pe=pe, k_pe=k_pe, **kw)
                if not fires:
                    return z_pos
                if h.nag_null_bias.shape[-1] != context.shape[1]:
                    raise RuntimeError(
                        f"NAG null bias {tuple(h.nag_null_bias.shape)} does not fit this "
                        f"pass (text keys {context.shape[1]})")
                z_neg = original(x, context=context,
                                 mask=h.nag_null_bias.to(dtype=x.dtype, device=x.device),
                                 pe=pe, k_pe=k_pe, **kw)
                g = z_pos * h.nag_scale - z_neg * (h.nag_scale - 1.0)
                ratio = g.abs().sum(dim=-1, keepdim=True) / (
                    z_pos.abs().sum(dim=-1, keepdim=True) + 1e-6)
                g = g * (ratio.clamp(max=h.nag_tau) / (ratio + 1e-6))
                out = g * h.nag_alpha + z_pos * (1.0 - h.nag_alpha)
                if h.nag_rows is not None:
                    keep = h.nag_rows.to(device=out.device).view(-1, 1, 1).bool()
                    out = torch.where(keep, out, z_pos)
                return out
            return forward

        _wrap(module, "nag", make)
        n += 1
    h.counts["nag"] = n
    return n


# --------------------------------------------------------------------------- install


def install_gates(model, *, num_spans: int, span_len: int, a2v_mode: str = "slice",
                  with_lora_gate: bool = True) -> GateHandles:
    """Install all gates in the required order and report what was wrapped.

    Order matters: NAG must wrap the text-routed forward, or its positive branch would read
    every caption at once and the guidance would be computed against the wrong reference.
    """
    h = GateHandles(num_spans=num_spans, span_len=span_len, a2v_mode=a2v_mode)
    _install_text(model, h)
    _install_span(model, h)
    if with_lora_gate:
        _install_lora(model, h)
    _install_a2v(model, h)
    _install_nag(model, h)
    model.stem_gates = h
    return h
