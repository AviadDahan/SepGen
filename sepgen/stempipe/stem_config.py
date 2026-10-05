"""Configuration surface for the stem add-on.

`StemConfig is None` => the pipeline is stock LTX-2.5. Everything our method does is
expressed here; nothing in this package reads a training config, a trainer, or any other
inference code in this project (written from
scratch against the upstream LTX-2.5 packages only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# The audio sequence our method denoises is [stem0 | stem1 | ... | mix]; THE MIX IS LAST.
# Every slice, mask and gate in this package derives from that one fact.

SamplerKind = Literal["res_2s", "euler"]
A2VMode = Literal["slice", "mask", "off"]


@dataclass(frozen=True)
class BetaSchedule:
    """Mix-read amplification on the audio self-attention bias.

    `b0` decaying-uniform term on all stem-query x mix-key entries: b0 * (1 - frac).
    `bf` CONSTANT term on the |t_stem - t_mix| <= w temporal diagonal band.
    `frac` runs 0 -> 1 over the schedule; on res_2s it is interpolated from sigma on a
    30-step reference grid so the decay shape matches the Euler form despite 15 steps
    with midpoints.
    """

    b0: float = 1.0
    bf: float = 1.0
    w: int = 4

    @staticmethod
    def parse(spec: str | None) -> "BetaSchedule | None":
        """`diag:B0:BF:W` (the only kind this package implements)."""
        if not spec:
            return None
        parts = spec.split(":")
        if parts[0] != "diag" or len(parts) != 4:
            raise ValueError(f"beta schedule must be 'diag:B0:BF:W', got {spec!r}")
        return BetaSchedule(b0=float(parts[1]), bf=float(parts[2]), w=int(parts[3]))


@dataclass(frozen=True)
class NagParams:
    """Normalized attention guidance on the stems' text cross-attention."""

    scale: float = 2.0
    tau: float = 2.5
    alpha: float = 0.5


@dataclass(frozen=True)
class TrainedInterface:
    """The method's core: re-solve the stems against a CLEAN mix at per-token timestep 0.

    Gating is expressed in BOTH currencies because the two samplers count differently:
    res_2s calls the denoiser twice per step plus a terminal sigma-0 call, so a
    step-index gate would fire wrongly there -- res_2s keys on `sigma_gate`, Euler on
    `from_step`.
    """

    enabled: bool = True
    sigma_gate: float = 0.97          # res_2s: engage when sigma <= this (1.0 = from the start)
    from_step: int = 12               # euler: engage when step_index >= this (of 30)
    sibling_negative: bool = True     # negative branch = the sibling's caption


@dataclass(frozen=True)
class CaptureConfig:
    """Dump the attention that a source-localization readout is computed from.

    Purely OBSERVATIONAL: it records and changes nothing, which is a claim the
    observer-identity check has to keep honest (capture on must give byte-identical media to
    capture off). Reading the maps back out is a separate step and a separate script, so this
    switch is only about whether the render leaves evidence behind.
    """

    enabled: bool = True
    channels: tuple[str, ...] = ("v2a",)
    # The two worlds the denoiser runs the model in: the ordinary conditional pass, and the
    # trained-interface pass where the mix is clean at timestep 0 and the sources are
    # re-solved against it. Kept apart rather than averaged, so which to pool stays a
    # decision the readout makes later instead of one this render makes for it.
    streams: tuple[str, ...] = ("main", "ti")
    band: tuple[int, ...] = (26, 27, 28, 29, 30, 31, 32, 33, 36)
    # Which blocks the per-step and per-audio-token tees cover: "band" (the 9 inherited
    # localization blocks) or "all" (every block). The per-block maps are always all 48; these
    # two are the expensive axes. "all" is what lets "which layer localizes on THIS model" be
    # answered from the dump rather than by another render.
    tee_blocks: str = "band"
    save_extra: tuple[str, ...] = ()      # "heads" | "audio_time" | "all_steps"

    def validate(self) -> None:
        from .capture import CHANNELS, EXTRAS, STREAMS

        for name, got, known in (("channels", self.channels, CHANNELS),
                                 ("streams", self.streams, STREAMS),
                                 ("save_extra", self.save_extra, EXTRAS)):
            unknown = [v for v in got if v not in known]
            if unknown:
                raise ValueError(f"unknown capture {name}: {unknown}; known: {list(known)}")
        if self.enabled and not self.channels:
            raise ValueError("capture is enabled with no channels to capture")
        if self.enabled and not self.streams:
            raise ValueError("capture is enabled with no passes to capture")
        if len(set(self.band)) != len(self.band):
            raise ValueError(f"capture band repeats a block: {self.band}")
        if self.tee_blocks not in ("band", "all"):
            raise ValueError(f"capture tee_blocks must be 'band' or 'all', "
                             f"got {self.tee_blocks!r}")


@dataclass(frozen=True)
class StemConfig:
    """The whole add-on. Absent => stock LTX."""

    stem_prompts: tuple[str, ...] = ()          # one caption per source; len == num_stems
    include_mix: bool = True                    # the trailing mix span
    sampler: SamplerKind = "res_2s"

    # --- per-channel negative prompts ---------------------------------------------
    # `None` falls back to the call's single `negative_prompt`, which is what upstream
    # takes. Naming them separately matters because the channels want different things:
    # the video negative is about picture quality, while a source's negative is about
    # *content that belongs to something else*.
    video_negative_prompt: str | None = None
    mix_negative_prompt: str | None = None
    stem_negative_prompts: tuple[str, ...] = ()   # one per source, in span order

    # --- attention topology -------------------------------------------------------
    span_sibling: bool = True                   # stems may read each other
    a2v_mode: A2VMode = "slice"                 # how video reads the audio: mix-span only
    text_block_diagonal: bool = True            # each span sees only its own caption block

    # --- guidance -----------------------------------------------------------------
    stem_rescale: float = 0.7                   # stems' CFG-rescale (mix keeps the preset's)
    stem_modality_scale: float = 1.0            # unified 1.0 on both samplers (2026-08-26)
    per_span_stats: bool = True                 # guider statistics per span, never global
    # Draw the MIXTURE's initial noise where stock draws its audio -- immediately after the
    # video, before the sources. The mix span already uses frozen base weights and reads only
    # itself, so with stock's noise as well it is sampling the same mixture stock would have
    # sampled, instead of whatever the 4th draw happens to give.
    mix_noise_first: bool = True

    # --- mechanisms ---------------------------------------------------------------
    trained_interface: TrainedInterface = field(default_factory=TrainedInterface)
    nag: NagParams | None = field(default_factory=NagParams)
    beta: BetaSchedule | None = field(default_factory=BetaSchedule)

    # --- observation (changes nothing about the render) ---------------------------
    capture: CaptureConfig | None = None

    # --- staggered generation (Euler only) ----------------------------------------
    mix_lead_alpha: float = 1.0                 # mix sigma = alpha * stem sigma
    mix_sigma_track: tuple[float, ...] | None = None   # explicit per-step mix sigmas

    @classmethod
    def from_dict(cls, d: dict, *, stem_prompts: tuple[str, ...] = ()) -> "StemConfig":
        """Build from a config-file section; unknown keys are refused, not ignored."""
        # keys starting with "_" are config comments; "enabled" is the arm switch
        d = {k: v for k, v in d.items() if k != "enabled" and not k.startswith("_")}
        if isinstance(d.get("stem_negative_prompts"), list):
            d["stem_negative_prompts"] = tuple(d["stem_negative_prompts"])
        ti = d.pop("trained_interface", None)
        nag = d.pop("nag", None)
        beta = d.pop("beta", None)
        capture = d.pop("capture", None)
        known = {f for f in cls.__dataclass_fields__ if f != "stem_prompts"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown stem config keys: {sorted(unknown)}")
        if capture is not None:
            capture = {k: (tuple(v) if isinstance(v, list) else v)
                       for k, v in capture.items() if not k.startswith("_")}
        return cls(
            stem_prompts=stem_prompts,
            trained_interface=TrainedInterface(**ti) if ti is not None else TrainedInterface(),
            nag=(NagParams(**nag) if nag else None),
            beta=(BetaSchedule.parse(beta) if isinstance(beta, str) else
                  BetaSchedule(**beta) if beta else None),
            capture=(CaptureConfig(**capture) if capture else None),
            **d,
        )

    def resolved_stem_negatives(self) -> tuple[str, ...]:
        """One negative per source, in span order, or () to fall back to the shared one.

        Plain text, nothing interpreted. To push a source away from what the other one is
        saying, write that other caption here -- the strings being equal is what makes it
        the sibling negative; there is no channel-pointing syntax to learn or to go stale.
        """
        value = self.stem_negative_prompts
        if not value:
            return ()
        if len(value) != self.num_stems:
            raise ValueError(
                f"got {len(value)} stem negative prompts for {self.num_stems} sources; "
                "give exactly one per source, or none at all")
        return tuple(value)

    @property
    def num_stems(self) -> int:
        return len(self.stem_prompts)

    @property
    def num_spans(self) -> int:
        return self.num_stems + (1 if self.include_mix else 0)

    def validate(self) -> None:
        if self.num_stems < 2:
            raise ValueError(f"need at least 2 stem captions, got {self.num_stems}")
        if not self.include_mix:
            raise ValueError(
                "include_mix=False is not supported: the trained interface, the a2v gate "
                "and the span topology are all defined against a trailing mix span")
        if self.trained_interface.enabled and not (0.0 < self.trained_interface.sigma_gate <= 1.0):
            raise ValueError("trained-interface sigma gate must be in (0, 1]")
        if self.nag is not None and self.nag.scale <= 1.0:
            raise ValueError("NAG scale must be > 1.0 (scale 1.0 is a no-op by construction)")
        if self.sampler == "res_2s" and (self.mix_sigma_track or self.mix_lead_alpha != 1.0):
            raise ValueError(
                "staggered generation (mix_sigma_track / mix_lead_alpha) needs per-token "
                "sigmas, which the res_2s loop cannot express -- use sampler='euler'")
        if self.a2v_mode not in ("slice", "mask", "off"):
            raise ValueError(f"unknown a2v mode {self.a2v_mode!r}")
        if self.capture is not None:
            self.capture.validate()
            if self.capture.enabled and "ti" in self.capture.streams \
                    and not self.trained_interface.enabled:
                raise ValueError(
                    "capture asks for the 'ti' pass but the trained interface is disabled, "
                    "so that pass never runs -- drop it from streams")
        self.resolved_stem_negatives()          # raises on a wrong-length list
