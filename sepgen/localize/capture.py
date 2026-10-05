"""Attention capture for localization, recorded during the separation render.

Two localization channels are measured on every conditional (positive) transformer pass, as a pure
observer: each wrapped attention callable recomputes an fp32 ``softmax(q k^T / sqrt(d))`` on the
side (head-chunked; q/k arrive post-RMSNorm+RoPE, so this is exactly the backend's distribution)
and then delegates unchanged, so the stems are bit-identical with or without capture.

- ``v2a``: ``video_to_audio_attn``, the audio-span queries reading video keys; the query-mean of
  each span ([stem0 | stem1 | mix]) over the video tokens.
- ``t2v``: the video ``attn2`` (video queries x scene text), read as a per-stem counterfactual.
  The video branch only sees the SCENE caption, so the keys and values are rebuilt from each
  stem's own caption with the same block's ``to_k`` + ``k_norm`` / ``to_v`` (after the same
  AdaLN text modulation) and scored in output space (ConceptAttention): saliency =
  <o_x, concept_k>, with o_x the model's own attn2 output per video token and concept_k the mean
  of stem k's rebuilt values over its content-word columns.

A transformer pre-hook classifies passes: perturbed and CFG-negative passes are skipped (the
negative pass is recognised by its audio text-context length).
"""
import math
import os
import re

os.environ.setdefault("LTX_MASKED_ATTENTION", "sdpa")

import numpy as np  # noqa: E402
import torch  # noqa: E402

N_SPANS = 3        # [stem0 | stem1 | mix]
HEAD_CHUNK = 8


# ------------------------------------------------------------------ capture state
class Cap:
    enabled = False
    pass_kind = None
    neg_len = None
    span_len = None
    video_tokens = None
    grid = None                    # (F, H, W)
    video_grid_from_positions = None   # grid read off video.positions (checked once)
    steps_v2a_maps: list = []      # per pos pass: {block: np [3, F*H*W]} query-mean
    # Per-stem counterfactual t2v: keep the real video queries of this pass, rebuild keys and
    # values from stem k's own context through the same block's to_k + k_norm / to_v.
    stem_ctx: list | None = None       # [2] CPU tensors [1, 1024, D] per-stem contexts
    scene_ctx = None                   # CPU tensor [1, 1024, D] -- the context the model USES
    stem_valid: list | None = None     # [2] index tensors of non-pad text positions
    ctx_dtype = None                   # dtype the model's text context is fed in
    kv_mod = None                      # (shift_kv, scale_kv) of the CURRENT cross-attn call:
                                       # the block AdaLN-modulates the text context before
                                       # to_k, so a replay must apply the same transform
    steps_t2v_stem: list = []          # per pos pass: {block: np [2, Tv]} head-mean
    stem_content_valid: list | None = None   # [2] index tensors: CONTENT word columns only
    out_space: bool = False                  # compute output-space saliency arrays
    steps_t2v_stem_content: list = []        # {block: np [2, Tv]} head-mean, content pooling
    steps_t2v_out: list = []                 # {block: np [2, Tv]} out-space, all-token concept
    steps_t2v_out_content: list = []         # {block: np [2, Tv]} out-space, content concept
    sigmas: list = []

    @classmethod
    def reset(cls):
        cls.steps_v2a_maps = []
        cls.steps_t2v_stem = []
        cls.steps_t2v_stem_content = []
        cls.steps_t2v_out, cls.steps_t2v_out_content = [], []
        cls.sigmas = []
        cls.pass_kind, cls.span_len = None, None


def transformer_pre_hook(_module, args, kwargs):
    """Classify the pass; open a per-step record on conditional (pos) passes only."""
    if not Cap.enabled:
        return
    audio = kwargs.get("audio", args[1] if len(args) > 1 else None)
    pert = kwargs.get("perturbations", args[2] if len(args) > 2 else None)
    if pert is not None:
        Cap.pass_kind = "perturbed"
        return
    if audio is None or audio.context is None:
        raise RuntimeError("capture pre-hook: no audio modality/context on this forward")
    if int(audio.context.shape[1]) == Cap.neg_len:
        Cap.pass_kind = "neg"          # CFG-negative pass
        return
    Cap.pass_kind = "pos"
    video = kwargs.get("video", args[0] if args else None)
    if Cap.video_grid_from_positions is None and video is not None:
        pos = video.positions          # [1, 3, T] per-token (t, h, w) grid coordinates
        t, h, w = (pos[0, i] for i in range(3))
        uniq = [int(x.unique().numel()) for x in (t, h, w)]
        Cap.video_grid_from_positions = tuple(uniq)
        print(f"[attnloc] video latent {tuple(video.latent.shape)} positions "
              f"{tuple(pos.shape)}; config grid was {Cap.grid} = {Cap.video_tokens}",
              flush=True)
        if int(video.latent.shape[1]) != Cap.video_tokens:
            raise RuntimeError(
                f"video latent has {int(video.latent.shape[1])} tokens but the config grid "
                f"{Cap.grid} says {Cap.video_tokens} -- the render is NOT at the configured "
                "resolution, so every spatial map would be mis-gridded")
    t_audio = int(audio.latent.shape[1])
    if t_audio % N_SPANS != 0:
        raise RuntimeError(f"audio tokens {t_audio} not divisible by {N_SPANS} spans")
    Cap.span_len = t_audio // N_SPANS
    Cap.steps_v2a_maps.append({})
    Cap.steps_t2v_stem.append({})
    Cap.steps_t2v_stem_content.append({})
    Cap.steps_t2v_out.append({})
    Cap.steps_t2v_out_content.append({})
    Cap.sigmas.append(float(audio.sigma[0].item()))


class CaptureAttn:
    """Wraps one backend callable: measures on pos passes, then delegates unchanged."""

    def __init__(self, orig, kind: str, block_idx: int, module=None):
        self.orig = orig
        self.kind = kind               # "v2a" | "t2v"
        self.block_idx = block_idx
        self.module = module           # needed by the per-stem counterfactual t2v mode

    def __call__(self, q, k, v, heads, mask=None):
        if Cap.enabled and Cap.pass_kind == "pos":
            self._measure(q, k, v, heads, mask)
        if mask is None:
            return self.orig(q, k, v, heads)
        return self.orig(q, k, v, heads, mask)

    @torch.no_grad()
    def _measure_t2v_per_stem(self, qf, kf, heads: int, dh: int, tq: int):
        """Video queries x EACH STEM'S OWN prompt keys, rebuilt with this block's weights.

        The scene context is rebuilt too and checked against the key tensor the model actually
        passed in: if they disagree, the key path is not the one replayed here and the
        counterfactual would be meaningless -- so that is a hard error.
        """
        if tq != Cap.video_tokens:
            raise RuntimeError(
                f"t2v blk{self.block_idx}: Tq {tq} != {Cap.video_tokens} "
                f"(Tk {kf.shape[2]}, heads {heads}, dh {dh}, pass {Cap.pass_kind}, "
                f"sigma {Cap.sigmas[-1]:.3f}, step {len(Cap.sigmas)}, "
                f"span_len {Cap.span_len})")
        # NOT cached across steps: the context modulation is per-step, so the keys are too.
        mod = self.module
        if mod is None:
            raise RuntimeError("per-stem t2v needs the attention module reference")
        dev = qf.device
        built, built_v = [], []
        for ctx in [*Cap.stem_ctx, Cap.scene_ctx]:
            c = ctx.to(dev, dtype=Cap.ctx_dtype)
            if Cap.kv_mod is not None:
                shift_kv, scale_kv = Cap.kv_mod
                c = c * (1 + scale_kv.to(c.dtype)) + shift_kv.to(c.dtype)
            kk = mod.k_norm(mod.to_k(c))
            built.append(kk.view(1, kk.shape[1], heads, dh).permute(0, 2, 1, 3).float())
            if Cap.out_space:
                vv = mod.to_v(c)
                built_v.append(
                    vv.view(1, vv.shape[1], heads, dh).permute(0, 2, 1, 3).float())
        ref = built[-1]
        if ref.shape != kf.shape:
            raise RuntimeError(f"t2v blk{self.block_idx}: replayed scene key shape "
                               f"{tuple(ref.shape)} != model's {tuple(kf.shape)}")
        denom = kf.abs().mean().item() + 1e-8
        err = (ref - kf).abs().mean().item() / denom
        if err > 1e-2:
            raise RuntimeError(
                f"t2v blk{self.block_idx}: replayed scene keys differ from the model's by "
                f"{err:.2%} (mean abs, relative) -- the key path is not to_k + k_norm "
                "alone (RoPE or a mask is involved); per-stem counterfactual invalid")
        if Cap.out_space:
            vref = built_v[-1]
            vdenom = self._vf.abs().mean().item() + 1e-8
            verr = (vref - self._vf).abs().mean().item() / vdenom
            if verr > 1e-2:
                raise RuntimeError(
                    f"t2v blk{self.block_idx}: replayed scene VALUES differ from the "
                    f"model's by {verr:.2%} -- the value path is not to_v alone; "
                    "output-space saliency invalid")
        cache = built[:len(Cap.stem_ctx)]
        cache_v = built_v[:len(Cap.stem_ctx)] if Cap.out_space else None
        n_src = len(cache)
        do_content = Cap.stem_content_valid is not None
        pool = torch.zeros(n_src, tq)
        pool_content = torch.zeros(n_src, tq) if do_content else None
        for si, k_alt in enumerate(cache):
            cols = Cap.stem_valid[si].to(qf.device)
            ccols = Cap.stem_content_valid[si].to(qf.device) if do_content else None
            for h0 in range(0, heads, HEAD_CHUNK):
                h1 = min(h0 + HEAD_CHUNK, heads)
                lg = qf[:, h0:h1] @ k_alt[:, h0:h1].transpose(-2, -1) / math.sqrt(dh)
                soft = lg.softmax(-1)[0]
                w = soft[:, :, cols].sum(-1)                   # [h, Tv] mass on real tokens
                pool[si] += w.sum(0).cpu()
                if do_content:
                    wc = soft[:, :, ccols].sum(-1)             # [h, Tv] mass on CONTENT words
                    pool_content[si] += wc.sum(0).cpu()
                    del wc
                del lg, soft, w
        pool = pool / heads
        if do_content:
            pool_content = pool_content / heads

        # OUTPUT-SPACE saliency (ConceptAttention): o_x = the model's own attn2 output per
        # video token (scene keys+values, pre-to_out); concept vector per stem = mean of its
        # replayed values over the pooled columns; saliency = <o_x, concept> per head.
        if Cap.out_space:
            out_all = torch.zeros(n_src, tq)
            out_content = torch.zeros(n_src, tq) if do_content else None
            oc_all, oc_content = [], []
            for si in range(n_src):
                cols = Cap.stem_valid[si].to(qf.device)
                oc_all.append(cache_v[si][0][:, cols].mean(1))          # [heads, dh]
                if do_content:
                    ccols = Cap.stem_content_valid[si].to(qf.device)
                    oc_content.append(cache_v[si][0][:, ccols].mean(1))
            for h0 in range(0, heads, HEAD_CHUNK):
                h1 = min(h0 + HEAD_CHUNK, heads)
                lg = qf[:, h0:h1] @ kf[:, h0:h1].transpose(-2, -1) / math.sqrt(dh)
                o_x = lg.softmax(-1)[0] @ self._vf[0, h0:h1]            # [h, Tv, dh]
                for si in range(n_src):
                    s = (o_x * oc_all[si][h0:h1, None]).sum(-1)         # [h, Tv]
                    out_all[si] += s.sum(0).cpu()
                    if do_content:
                        sc = (o_x * oc_content[si][h0:h1, None]).sum(-1)
                        out_content[si] += sc.sum(0).cpu()
                        del sc
                    del s
                del lg, o_x
            Cap.steps_t2v_out[-1][self.block_idx] = (
                (out_all / heads).numpy().astype(np.float16))
            if do_content:
                Cap.steps_t2v_out_content[-1][self.block_idx] = (
                    (out_content / heads).numpy().astype(np.float16))
        # Alignment sanity: how much softmax mass lands on the assumed real-token span? A
        # span holding only its area share (n_tok / key_len) would mean the columns are
        # mis-aligned (or the model ignores the prompt); printed once per clip.
        if not Cap.steps_t2v_stem[-1] and len(Cap.sigmas) == 1:
            for si in range(n_src):
                share = float(pool[si].mean())
                area = float(Cap.stem_valid[si].numel()) / float(kf.shape[2])
                print(f"[attnloc]   blk{self.block_idx} stem{si}: mass on real tokens "
                      f"{share:.3f} vs uniform-share {area:.3f} "
                      f"({share / max(area, 1e-9):.1f}x)", flush=True)
        # A pooled softmax that covers EVERY key is identically 1.0 at every video token --
        # a constant map that looks like a valid array. Catch it on the first block.
        if not Cap.steps_t2v_stem[-1] and float(pool.max() - pool.min()) < 1e-4:
            raise RuntimeError(
                f"t2v blk{self.block_idx}: per-stem map is CONSTANT "
                f"(range {float(pool.max() - pool.min()):.2e}, mean {float(pool.mean()):.3f}) "
                "-- the pooled columns cover the whole key axis, so there is no signal left")
        Cap.steps_t2v_stem[-1][self.block_idx] = pool.numpy().astype(np.float16)
        if do_content:
            Cap.steps_t2v_stem_content[-1][self.block_idx] = (
                pool_content.numpy().astype(np.float16))

    @torch.no_grad()
    def _measure(self, q, k, v, heads, mask):
        if mask is not None:
            raise RuntimeError(
                f"{self.kind} blk{self.block_idx}: unexpected attention mask -- the "
                "v2a/t2v paths are maskless; a masked call means the layout changed")
        b, tq, hd = q.shape
        if b != 1:
            raise RuntimeError(f"capture expects B=1, got {b}")
        dh = hd // heads
        tk = k.shape[1]
        L = Cap.span_len
        qf = q.view(1, tq, heads, dh).permute(0, 2, 1, 3).float()
        kf = k.view(1, tk, heads, dh).permute(0, 2, 1, 3).float()

        if self.kind == "t2v":                       # video queries x SCENE text keys
            if Cap.stem_ctx is None:
                raise RuntimeError("t2v capture needs Cap.stem_ctx (the per-stem contexts)")
            self._vf = (v.view(1, tk, heads, dh).permute(0, 2, 1, 3).float()
                        if Cap.out_space else None)
            self._measure_t2v_per_stem(qf, kf, heads, dh, tq)
            self._vf = None
            return

        # v2a: audio-span queries x video keys
        if tq != N_SPANS * L:
            raise RuntimeError(f"v2a blk{self.block_idx}: Tq {tq} != {N_SPANS}*{L}")
        if tk != Cap.video_tokens:
            raise RuntimeError(f"v2a blk{self.block_idx}: Tk {tk} != {Cap.video_tokens}")
        span_mean = torch.zeros(N_SPANS, tk)
        for h0 in range(0, heads, HEAD_CHUNK):
            h1 = min(h0 + HEAD_CHUNK, heads)
            lg = qf[:, h0:h1] @ kf[:, h0:h1].transpose(-2, -1) / math.sqrt(dh)
            w = lg.softmax(-1)[0]                    # [h, 3L, Tk]
            wm = w.sum(0)                            # [3L, Tk] head-sum
            for qs in range(N_SPANS):
                rows = wm[qs * L:(qs + 1) * L]
                span_mean[qs] += rows.mean(0).cpu()
            del lg, w, wm
        Cap.steps_v2a_maps[-1][self.block_idx] = (
            (span_mean / heads).numpy().astype(np.float16))


def install_context_modulation_probe() -> None:
    """Record the AdaLN (shift_kv, scale_kv) applied to the text context of each cross-attn.

    ``apply_cross_attention_adaln`` transforms the context as
    ``context * (1 + scale_kv) + shift_kv`` before the attention module ever sees it, with
    per-block, per-step values. The per-stem counterfactual has to apply the SAME transform
    to the stem contexts, so this wrapper stashes them for the wrapper that runs immediately
    afterwards (inside the very same call). Read-only: it delegates unchanged.
    """
    from ltx_core.model.transformer import transformer as ltx_tf

    orig = ltx_tf.apply_cross_attention_adaln
    if getattr(orig, "_attnloc_wrapped", False):
        return

    def wrapped(x, context, attn, q_shift, q_scale, q_gate, prompt_scale_shift_table,
                prompt_timestep, context_mask=None):
        if Cap.enabled and Cap.stem_ctx is not None:
            batch_size = x.shape[0]
            shift_kv, scale_kv = (
                prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
                + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
            ).unbind(dim=2)
            Cap.kv_mod = (shift_kv, scale_kv)
        return orig(x, context, attn, q_shift, q_scale, q_gate, prompt_scale_shift_table,
                    prompt_timestep, context_mask)

    wrapped._attnloc_wrapped = True
    ltx_tf.apply_cross_attention_adaln = wrapped


def install_capture(model) -> tuple[int, int]:
    """Wrap the v2a and t2v attention callables; returns (n_wrapped_modules, n_blocks), with
    n_blocks counted from the v2a modules (one per block)."""
    n = 0
    n_v2a = 0
    for name, module in model.named_modules():
        if name.endswith("video_to_audio_attn"):
            kind = "v2a"
        elif name.endswith(".attn2"):        # video text attn ("attn2"); audio_attn2 has "_"
            kind = "t2v"
        else:
            continue
        m = re.search(r"\.(\d+)\.(?:video_to_audio_attn|attn2)$", name)
        if m is None:
            raise RuntimeError(f"cannot parse block index from {name}")
        block_idx = int(m.group(1))
        module.attention_function = CaptureAttn(module.attention_function, kind, block_idx,
                                                module)
        module.masked_attention_function = CaptureAttn(
            module.masked_attention_function, kind, block_idx, module)
        n += 1
        if kind == "v2a":
            n_v2a += 1
    if n == 0 or n_v2a == 0:
        raise RuntimeError("no video_to_audio_attn / attn2 modules found")
    return n, n_v2a


def stack_blocks(steps: list, n_blocks: int, shape: tuple) -> np.ndarray:
    """[steps][block->arr] -> np [S, n_blocks, *shape]; missing block = hard failure."""
    out = np.empty((len(steps), n_blocks, *shape), dtype=np.float16)
    for si, rec in enumerate(steps):
        for bi in range(n_blocks):
            if bi not in rec:
                raise RuntimeError(f"step {si}: block {bi} missing from capture")
            out[si, bi] = rec[bi]
    return out
