"""Attention capture: the maps a source-localization readout is computed from.

The seam is `module.attention_function` — a DIFFERENT attribute from the `module.forward`
that this package's five gates wrap, so capture and gating compose without touching each
other.

By the time q/k reach that callable they are post-`to_q`/`to_k`, post-RMSNorm and post-RoPE,
shaped `(B, T, heads*dim_head)`. So nothing here replicates anything about the model: it
reshapes, computes `softmax(q kᵀ/√d)` in fp32 on the side, and hands the original call
through unchanged. Hooking any higher level would mean re-applying RMSNorm and RoPE by hand.

Which pass a forward belongs to is **declared** by the denoiser (`pass_context(capture=...)`),
never inferred from tensor widths — the same rule the gates follow, for the same reason: two
passes that happen to share a shape would otherwise silently get each other's treatment.

The channel captured here is **v2a**: the module named `video_to_audio_attn`, which despite
its name has AUDIO queries and VIDEO keys (`transformer.py:166` — "Q: Audio, K,V: Video").
It is named for the direction information flows, which is the opposite of its query/key
layout, so the installer asserts the layout rather than trusting the name. Our three spans
are its queries and the video grid is its keys, which makes the map per-source by
construction — no counterfactual needed.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import torch

# Bounds the fp32 logits temporary: 8 heads x 339 queries x 5760 keys x 4 B ~= 62 MB.
HEAD_CHUNK = 8

_V2A_RE = re.compile(r"\.(\d+)\.video_to_audio_attn$")
# The leading dot is load-bearing: "audio_attn2".endswith("attn2") is True, so a suffix match
# without it lands on the AUDIO text-attention instead of the video's, silently capturing the
# wrong site.
_T2V_RE = re.compile(r"\.(\d+)\.attn2$")

# The 9 blocks the phase-14 lineage pools over. Kept as a DEFAULT, never as the only thing
# saved: it was tuned on a different model, so all 48 blocks are stored and the band stays
# re-derivable offline instead of by re-rendering.
DEFAULT_BAND = (26, 27, 28, 29, 30, 31, 32, 33, 36)

STREAMS = ("main", "ti")
CHANNELS = ("v2a", "t2v")

# BOS and the token after it act as attention sinks -- they absorb mass that has nothing to do
# with the caption's content, so a concept vector built over them describes the sink, not the
# source. Dropped from every replayed caption.
_SINK_COLUMNS = 2
EXTRAS = ("heads", "audio_time", "all_steps")


@dataclass
class AttentionCapture:
    """One cell's finished maps, ready to write. All arrays are numpy."""

    arrays: dict[str, np.ndarray]
    grid: tuple[int, int, int]
    span_len: int
    num_spans: int
    band: tuple[int, ...]
    tee: tuple[int, ...]
    pass_counts: dict[str, int]

    def npz_payload(self) -> dict[str, np.ndarray]:
        """Everything that goes into `attention_maps.npz`.

        Numeric metadata rides along so the readout never has to be told the geometry it is
        reading -- a grid passed separately is a grid that can be wrong.
        """
        meta = {
            "grid_FHW": np.array(self.grid, dtype=np.int32),
            "span_len": np.array(self.span_len, dtype=np.int32),
            "num_spans": np.array(self.num_spans, dtype=np.int32),
            "band": np.array(self.band, dtype=np.int32),
            "tee_blocks": np.array(self.tee, dtype=np.int32),
            "pass_counts": np.array([self.pass_counts.get(s, 0) for s in STREAMS],
                                    dtype=np.int32),
            "pass_count_names": np.array(STREAMS),
        }
        return {**self.arrays, **meta}

    def summary(self) -> str:
        mb = sum(a.nbytes for a in self.arrays.values()) / 1e6
        shapes = ", ".join(f"{k}{tuple(v.shape)}" for k, v in sorted(self.arrays.items()))
        counts = ", ".join(f"{k} {v}" for k, v in self.pass_counts.items())
        return f"{mb:.1f} MB raw | passes: {counts} | {shapes}"


@dataclass
class _Accumulator:
    """Running sums for one (channel, stream). Every stored map is a MEAN over passes."""

    layers: torch.Tensor                                  # [n_blocks, S, Tk] fp32
    count: int = 0
    steps_band: list[torch.Tensor] = field(default_factory=list)     # per pass [9, S, Tk] f16
    all_steps: list[torch.Tensor] = field(default_factory=list)      # per pass [B, S, Tk] f16
    heads_band: torch.Tensor | None = None                # [9, heads, S, Tk] fp32
    audio_time_band: torch.Tensor | None = None           # [9, S, L, Tk] fp32


class CaptureHandles:
    """Per-cell capture state: armed by the pipeline, declared by the denoiser.

    Holds no module-level state, exactly like `GateHandles`, so two pipelines in one process
    could capture independently.
    """

    def __init__(self, *, grid: tuple[int, int, int], num_spans: int, span_len: int,
                 n_blocks: int, streams: tuple[str, ...] = STREAMS,
                 band: tuple[int, ...] = DEFAULT_BAND, save_extra: tuple[str, ...] = (),
                 channels: tuple[str, ...] = ("v2a",), tee_blocks: str = "band",
                 scene_context: torch.Tensor | None = None,
                 source_contexts: list[torch.Tensor] | None = None,
                 source_masks: list[torch.Tensor] | None = None,
                 device: torch.device | None = None):
        self.grid = tuple(int(g) for g in grid)
        self.video_tokens = int(np.prod(self.grid))
        self.num_spans = num_spans
        self.span_len = span_len
        self.n_blocks = n_blocks
        self.streams = tuple(streams)
        self.band = tuple(band)
        self.save_extra = tuple(save_extra)
        self.device = device or torch.device("cuda")

        bad = [b for b in self.band if not 0 <= b < n_blocks]
        if bad:
            raise ValueError(f"band blocks {bad} are outside 0..{n_blocks - 1}")
        # Which blocks the per-step and per-audio-token tees cover. The per-BLOCK maps are
        # always every block; these two are the expensive ones, so they are the ones with a
        # choice. "band" keeps the 9 localization blocks (small); "all" keeps every block, so
        # the question "which layer localizes on THIS model" can be answered from the dump
        # instead of by another render -- the band was tuned on a different model.
        if tee_blocks not in ("band", "all"):
            raise ValueError(f"tee_blocks must be 'band' or 'all', got {tee_blocks!r}")
        self.tee = tuple(self.band) if tee_blocks == "band" else tuple(range(n_blocks))
        self._tee_pos = {b: i for i, b in enumerate(self.tee)}
        self._head_pos = {b: i for i, b in enumerate(self.band)}

        self.channels = tuple(channels)
        # t2v replays each source's caption through the VIDEO text cross-attention, which the
        # video never actually reads -- it only ever sees the scene prompt. The scene context
        # is kept alongside so the replay can be checked against the model's own keys before
        # any of it is believed.
        self.scene_context = scene_context
        self.source_contexts = source_contexts or []
        self.source_masks = source_masks or []
        self.modulation: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._replay_checked: set[int] = set()

        self._acc: dict[tuple[str, str], _Accumulator] = {}
        self.active_stream: str | None = None
        self._pass_band: torch.Tensor | None = None
        self._pass_all: torch.Tensor | None = None
        self._seen_blocks: set[int] = set()
        self._t2v_pass_band: torch.Tensor | None = None
        self._t2v_seen: set[int] = set()

    # ------------------------------------------------------------------ declaration

    def wants(self, stream: str | None) -> bool:
        return stream is not None and stream in self.streams

    @property
    def num_sources(self) -> int:
        return self.num_spans - 1

    def begin_pass(self, stream: str) -> None:
        """Open a per-pass record. Anything a block reports now belongs to `stream`."""
        self.active_stream = stream
        self._seen_blocks = set()
        self._t2v_seen = set()
        tk = self.video_tokens
        self._pass_band = torch.zeros(len(self.tee), self.num_spans, tk,
                                      dtype=torch.float32, device=self.device)
        self._pass_all = (torch.zeros(self.n_blocks, self.num_spans, tk,
                                      dtype=torch.float32, device=self.device)
                          if "all_steps" in self.save_extra else None)
        self._t2v_pass_band = (torch.zeros(len(self.tee), self.num_sources, tk,
                                           dtype=torch.float32, device=self.device)
                               if "t2v" in self.channels else None)

    def end_pass(self) -> None:
        """Close it, and refuse a partial one.

        A pass that reported fewer blocks than the model has means a block was skipped or a
        wrapper was lost -- averaging that in would quietly bias every map toward whichever
        blocks did report.
        """
        if self.active_stream is None:
            return
        for channel, seen, pass_band, pass_all in (
                ("v2a", self._seen_blocks, self._pass_band, self._pass_all),
                ("t2v", self._t2v_seen, self._t2v_pass_band, None)):
            if not seen:
                continue
            if len(seen) != self.n_blocks:
                missing = sorted(set(range(self.n_blocks)) - seen)
                raise RuntimeError(
                    f"capture pass '{self.active_stream}' saw {len(seen)} of "
                    f"{self.n_blocks} {channel} blocks; missing {missing[:5]}...")
            acc = self._acc[(channel, self.active_stream)]
            acc.count += 1
            acc.steps_band.append(pass_band.to(torch.float16).cpu())
            if pass_all is not None:
                acc.all_steps.append(pass_all.to(torch.float16).cpu())
        self.active_stream = None
        self._pass_band = self._pass_all = self._t2v_pass_band = None
        self._seen_blocks = self._t2v_seen = set()

    # ------------------------------------------------------------------ recording

    def _accumulator(self, channel: str, stream: str, *, heads: int) -> _Accumulator:
        key = (channel, stream)
        acc = self._acc.get(key)
        if acc is None:
            # v2a has one row per SPAN (the mixture included); t2v replays only the source
            # captions, so it has one row per source and no mixture row to drop later.
            s = self.num_spans if channel == "v2a" else self.num_sources
            tk = self.video_tokens
            acc = _Accumulator(layers=torch.zeros(self.n_blocks, s, tk, dtype=torch.float32,
                                                  device=self.device))
            if "heads" in self.save_extra:
                acc.heads_band = torch.zeros(len(self.band), heads, s, tk,
                                             dtype=torch.float32, device=self.device)  # band
            if "audio_time" in self.save_extra and channel == "v2a":
                acc.audio_time_band = torch.zeros(len(self.tee), s, self.span_len, tk,
                                                  dtype=torch.float32, device=self.device)
            self._acc[key] = acc
        return acc

    @torch.no_grad()
    def record_v2a(self, block_idx: int, q: torch.Tensor, k: torch.Tensor, heads: int) -> None:
        """One block's audio-query x video-key attention, reduced and accumulated."""
        stream = self.active_stream
        if stream is None:
            return
        b, tq, hd = q.shape
        if b != 1:
            raise RuntimeError(f"capture expects batch 1, got {b}")
        want_q = self.num_spans * self.span_len
        if tq != want_q:
            raise RuntimeError(
                f"v2a block {block_idx}: {tq} queries, expected {want_q} "
                f"({self.num_spans} spans x {self.span_len}) -- the audio layout changed")
        tk = k.shape[1]
        if tk != self.video_tokens:
            raise RuntimeError(
                f"v2a block {block_idx}: {tk} keys, expected {self.video_tokens} = "
                f"{self.grid} -- this module's keys are not the video grid")
        if block_idx in self._seen_blocks:
            raise RuntimeError(f"v2a block {block_idx} reported twice in one declared pass")

        dh = hd // heads
        length = self.span_len
        acc = self._accumulator("v2a", stream, heads=heads)
        qf = q.view(1, tq, heads, dh).permute(0, 2, 1, 3).float()
        kf = k.view(1, tk, heads, dh).permute(0, 2, 1, 3).float()

        span_mean = torch.zeros(self.num_spans, tk, dtype=torch.float32, device=q.device)
        tee_pos = self._tee_pos.get(block_idx, -1)
        head_pos = self._head_pos.get(block_idx, -1)
        for h0 in range(0, heads, HEAD_CHUNK):
            h1 = min(h0 + HEAD_CHUNK, heads)
            logits = qf[:, h0:h1] @ kf[:, h0:h1].transpose(-2, -1) / math.sqrt(dh)
            w = logits.softmax(-1)[0]                       # [h, Tq, Tk]
            head_sum = w.sum(0)                             # [Tq, Tk]
            for s in range(self.num_spans):
                rows = head_sum[s * length:(s + 1) * length]
                span_mean[s] += rows.mean(0)
                if tee_pos >= 0 and acc.audio_time_band is not None:
                    acc.audio_time_band[tee_pos, s] += rows
                if head_pos >= 0 and acc.heads_band is not None:
                    acc.heads_band[head_pos, h0:h1, s] += (
                        w[:, s * length:(s + 1) * length].mean(1))
            del logits, w, head_sum
        span_mean /= heads

        acc.layers[block_idx] += span_mean
        if tee_pos >= 0:
            self._pass_band[tee_pos] = span_mean
        if self._pass_all is not None:
            self._pass_all[block_idx] = span_mean
        self._seen_blocks.add(block_idx)

    # ------------------------------------------------------------------ t2v

    def stage_modulation(self, module, shift_kv: torch.Tensor, scale_kv: torch.Tensor) -> None:
        """Record the AdaLN affine this block is about to apply to its text context.

        The model modulates the text before projecting it to keys and values
        (`encoder_hidden_states = context * (1 + scale_kv) + shift_kv`). A replayed caption
        has to go through the same affine or its keys live in a different space than the ones
        the model actually used, and the saliency would be measured against nothing.
        """
        self.modulation[id(module)] = (shift_kv, scale_kv)

    def _replay(self, module, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A caption's keys and values as THIS block would have computed them.

        A block that does not use cross-attention AdaLN never reaches the probe and has no
        modulation staged; there the context goes straight to the projections. Which case
        applies is not asserted here -- the replayed-key check against the model's own keys
        settles it, and settles it on evidence rather than on a flag.
        """
        weight = module.to_k.weight
        ctx = context.to(dtype=weight.dtype, device=weight.device)
        staged = self.modulation.get(id(module))
        if staged is not None:
            shift_kv, scale_kv = staged
            ctx = ctx * (1 + scale_kv.to(ctx.dtype)) + shift_kv.to(ctx.dtype)
        return module.k_norm(module.to_k(ctx)), module.to_v(ctx)

    @torch.no_grad()
    def record_t2v(self, block_idx: int, module, q: torch.Tensor, k: torch.Tensor,
                   v: torch.Tensor, heads: int) -> None:
        """Per-source saliency over the video grid, ConceptAttention style.

        The video's text cross-attention only ever reads the SCENE prompt, so asking "which
        video cells respond to source k's caption" means replaying that caption through this
        block's own key/value projections and re-running the attention against it. That is
        legal here only because these text keys carry no rotary embedding (`transformer.py`
        passes no `pe`/`k_pe` to this site) -- with RoPE the replay would need positions the
        caption never had.

        The score is taken in OUTPUT space, not attention space: how much each video token's
        attention output points along the caption's own content direction. Attention weight
        alone says a token looked at the caption; this says it came back carrying it.
        """
        stream = self.active_stream
        if stream is None or "t2v" not in self.channels:
            return
        if self.scene_context is None or not self.source_contexts:
            raise RuntimeError(
                "t2v capture is armed but no caption encodings were handed to it, so there is "
                "nothing to replay and nothing to check the replay against")
        b, tq, hd = q.shape
        if b != 1:
            raise RuntimeError(f"capture expects batch 1, got {b}")
        if tq != self.video_tokens:
            raise RuntimeError(
                f"t2v block {block_idx}: {tq} queries, expected {self.video_tokens} = "
                f"{self.grid} -- this site's queries are not the video grid")
        if block_idx in self._t2v_seen:
            raise RuntimeError(f"t2v block {block_idx} reported twice in one declared pass")

        # Before believing any counterfactual, reproduce a fact: replaying the context the
        # model DID read must give back the keys it actually used. If it does not, the key
        # path is not `modulate -> to_k -> k_norm` alone and every replay below is invalid.
        if block_idx not in self._replay_checked:
            ref, _ = self._replay(module, self.scene_context)
            if ref.shape != k.shape:
                raise RuntimeError(
                    f"t2v block {block_idx}: replayed scene keys are {tuple(ref.shape)}, the "
                    f"model's are {tuple(k.shape)}")
            err = ((ref - k).abs().mean() / (k.abs().mean() + 1e-8)).item()
            if err > 1e-2:
                raise RuntimeError(
                    f"t2v block {block_idx}: replayed scene keys differ from the model's own "
                    f"by {err:.2%} -- the key path is not modulate + to_k + k_norm alone, so "
                    "the per-source counterfactual would be measuring the wrong thing")
            self._replay_checked.add(block_idx)

        dh = hd // heads
        acc = self._accumulator("t2v", stream, heads=heads)
        qf = q.view(1, tq, heads, dh).permute(0, 2, 1, 3).float()
        tee_pos = self._tee_pos.get(block_idx, -1)

        for s, ctx in enumerate(self.source_contexts):
            k_s, v_s = self._replay(module, ctx)
            kf = k_s.view(1, -1, heads, dh).permute(0, 2, 1, 3).float()
            vf = v_s.view(1, -1, heads, dh).permute(0, 2, 1, 3).float()
            valid = self.source_masks[s] if s < len(self.source_masks) else None
            concept = self._concept_vector(vf, valid)          # [heads, dh]
            score = torch.zeros(tq, dtype=torch.float32, device=q.device)
            for h0 in range(0, heads, HEAD_CHUNK):
                h1 = min(h0 + HEAD_CHUNK, heads)
                logits = qf[:, h0:h1] @ kf[:, h0:h1].transpose(-2, -1) / math.sqrt(dh)
                out = logits.softmax(-1)[0] @ vf[0, h0:h1]      # [h, Tq, dh]
                score += (out * concept[h0:h1, None]).sum(-1).sum(0)
                del logits, out
            score /= heads
            acc.layers[block_idx, s] += score
            if tee_pos >= 0:
                self._t2v_pass_band[tee_pos, s] = score
            if tee_pos >= 0 and acc.heads_band is not None:
                pass          # per-head t2v is not stored: the score is already head-summed
        self._t2v_seen.add(block_idx)

    @staticmethod
    def _concept_vector(vf: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        """The caption's own direction in value space: the mean of its content columns."""
        length = vf.shape[2]
        lo = min(_SINK_COLUMNS, max(0, length - 1))
        if valid is not None:
            # This package reads text masks as BINARY, 1 = valid (see masks._apply_block). An
            # additive mask (0 for valid, -finfo.max for pad) would invert under .bool() and
            # the concept vector would be built from the padding -- silently, and only
            # visible as a readout that points at nothing. Refuse instead.
            uniq = torch.unique(valid)
            if bool(((uniq != 0) & (uniq != 1)).any()):
                raise RuntimeError(
                    f"text mask is not binary (values {uniq[:4].tolist()}); it looks additive, "
                    "and reading it as valid/pad would select exactly the wrong columns")
            keep = valid.reshape(-1).to(torch.bool)[:length].clone()
            keep[:lo] = False
            if not keep.any():
                keep = torch.ones(length, dtype=torch.bool, device=vf.device)
        else:
            keep = torch.ones(length, dtype=torch.bool, device=vf.device)
            keep[:lo] = False
        return vf[0, :, keep.to(vf.device)].mean(dim=1)

    # ------------------------------------------------------------------ finish

    def finish(self) -> AttentionCapture:
        """Reduce the accumulators to the arrays that go on disk."""
        f, h, w = self.grid
        arrays: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        for (channel, stream), acc in sorted(self._acc.items()):
            if acc.count == 0:
                continue
            counts[stream] = acc.count
            n = float(acc.count)
            # The row count comes from the accumulator itself: v2a has one row per SPAN
            # (mixture included), t2v one per SOURCE. Reshaping either with a hardcoded
            # count silently mismatches the other.
            rows = acc.layers.shape[1]
            # fp32, deliberately: these ARE the readout's input, and the reference chain's
            # fp16 round-trip created tie plateaus that changed which cells survived a
            # top-q gate. The bulk tees below stay fp16 -- nothing pools them yet.
            arrays[f"{channel}_{stream}_layers"] = (
                (acc.layers / n).reshape(self.n_blocks, rows, f, h, w)
                .cpu().numpy().astype(np.float32))
            if acc.steps_band:
                arrays[f"{channel}_{stream}_steps"] = (
                    torch.stack(acc.steps_band)
                    .reshape(acc.count, len(self.tee), rows, f, h, w)
                    .numpy().astype(np.float16))
            if acc.all_steps:
                arrays[f"{channel}_{stream}_all_steps"] = (
                    torch.stack(acc.all_steps)
                    .reshape(acc.count, self.n_blocks, rows, f, h, w)
                    .numpy().astype(np.float16))
            if acc.heads_band is not None:
                heads = acc.heads_band.shape[1]
                arrays[f"{channel}_{stream}_heads_band"] = (
                    (acc.heads_band / n)
                    .reshape(len(self.band), heads, rows, f, h, w)
                    .cpu().numpy().astype(np.float16))
            if acc.audio_time_band is not None:
                arrays[f"{channel}_{stream}_audio_time"] = (
                    (acc.audio_time_band / n)
                    .reshape(len(self.tee), rows, self.span_len, f, h, w)
                    .cpu().numpy().astype(np.float16))
        if not arrays:
            raise RuntimeError(
                "capture was armed but recorded nothing -- no pass was ever declared, so "
                "either the denoiser did not announce one or the wrappers were not installed")
        self._acc.clear()
        return AttentionCapture(arrays=arrays, grid=self.grid, span_len=self.span_len,
                                num_spans=self.num_spans, band=self.band, tee=self.tee,
                                pass_counts=counts)


# --------------------------------------------------------------------------- install


def _wrap(module, name: str, make):
    if getattr(module, f"_stem_capture_{name}", False):
        raise RuntimeError(f"capture is already installed on this module ({name})")
    setattr(module, name, make(getattr(module, name)))
    setattr(module, f"_stem_capture_{name}", True)


def install_capture(model, gates) -> int:
    """Wrap every v2a attention backend, once, at build time. Returns the block count.

    The wrappers read `gates.capture` on every call rather than closing over a fixed handles
    object, so a per-cell handles can be swapped in and out (and set to None to disarm)
    without ever reinstalling — the same lifecycle the gates themselves have.

    Both the unmasked and masked callables are wrapped: this site is called maskless
    (`transformer.py:387` passes no mask), so a masked call means the layout changed and a
    recompute would no longer match the distribution the model actually used — raise rather
    than record something subtly wrong.
    """
    n = 0
    for mod_name, module in model.named_modules():
        m = _V2A_RE.search(mod_name)
        if m is None:
            continue
        block_idx = int(m.group(1))

        def make(original, idx=block_idx):
            def wrapped(q, k, v, heads, *args, **kw):
                cap = gates.capture
                if cap is not None and cap.active_stream is not None:
                    cap.record_v2a(idx, q, k, heads)
                return original(q, k, v, heads, *args, **kw)
            return wrapped

        def make_masked(original, idx=block_idx):
            def wrapped(q, k, v, heads, mask, *args, **kw):
                cap = gates.capture
                if cap is not None and cap.active_stream is not None:
                    raise RuntimeError(
                        f"v2a block {idx} was called WITH a mask while capture was armed; "
                        "this site is maskless by construction, so the layout changed")
                return original(q, k, v, heads, mask, *args, **kw)
            return wrapped

        _wrap(module, "attention_function", make)
        _wrap(module, "masked_attention_function", make_masked)
        n += 1
    if n == 0:
        raise RuntimeError("no video_to_audio_attn modules found; wrong checkpoint?")

    t = 0
    for mod_name, module in model.named_modules():
        m = _T2V_RE.search(mod_name)
        if m is None:
            continue
        block_idx = int(m.group(1))

        def make_t2v(original, idx=block_idx, mod=module):
            def wrapped(q, k, v, heads, *args, **kw):
                cap = gates.capture
                if (cap is not None and cap.active_stream is not None
                        and "t2v" in cap.channels):
                    cap.record_t2v(idx, mod, q, k, v, heads)
                return original(q, k, v, heads, *args, **kw)
            return wrapped

        _wrap(module, "attention_function", make_t2v)
        t += 1
    if t and t != n:
        raise RuntimeError(f"wrapped {n} v2a sites but {t} t2v sites; the block count differs")
    gates.counts["capture"] = n
    gates.counts["capture_t2v"] = t
    return n


def install_adaln_probe(gates) -> None:
    """Record each block's text-context AdaLN affine, so a replay can match the model's keys.

    Patches the upstream module-level `apply_cross_attention_adaln`, which is where the affine
    is computed, immediately before the attention call it feeds. Patching upstream rather than
    reimplementing the affine is deliberate: the modulation is `prompt_scale_shift_table` plus
    an optional timestep-conditioned term, and which of those applies is a property of the
    checkpoint, not something to hardcode here.

    Idempotent, and a no-op for every site except the ones a capture is armed for.
    """
    from ltx_core.model.transformer import transformer as tmod

    if getattr(tmod, "_stem_adaln_probe", False):
        return
    original = tmod.apply_cross_attention_adaln

    def patched(x_normed, context, attn, q_shift, q_scale, q_gate,
                prompt_scale_shift_table, prompt_timestep, context_mask=None):
        cap = gates.capture
        if cap is not None and cap.active_stream is not None and "t2v" in cap.channels:
            kv = prompt_scale_shift_table[None, None].to(device=x_normed.device,
                                                         dtype=x_normed.dtype)
            if prompt_timestep is not None:
                kv = kv + prompt_timestep.reshape(x_normed.shape[0],
                                                  prompt_timestep.shape[1], 2, -1)
            shift_kv, scale_kv = kv.unbind(dim=2)
            cap.stage_modulation(attn, shift_kv, scale_kv)
        return original(x_normed, context, attn, q_shift, q_scale, q_gate,
                        prompt_scale_shift_table, prompt_timestep, context_mask)

    tmod.apply_cross_attention_adaln = patched
    tmod._stem_adaln_probe = True
