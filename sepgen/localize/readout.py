"""Localization readout: captured attention -> per-stem maps -> GT-free channel choice -> 2-D track.

Pure numpy/torch post-processing, no LTX imports.

  build-time maps   both channels averaged over the 30-step sigma ladder and over the block band
                    26-33 + 36, turned into per-cell stem probabilities (t2v: softmax over the two
                    stems of the output-space saliency; v2a: pairwise share of attention mass),
                    then per-frame normalized and top-10% masked (`sharpen`).
  channel choice    per stem, a GT-free suspicion z-score from three stability signals (pointer
                    jitter, support spread, frame-to-frame consistency) against frozen statistics
                    (sampler_stats.json); the calmer channel wins, and a stem whose suspicion
                    exceeds the frozen threshold is self-flagged.
  track             per frame, a support-gated Weiszfeld geometric median of the map on the cell
                    grid with its support covariance, smoothed by a constant-velocity Kalman
                    filter + RTS with per-clip maximum-likelihood process noise.

`GRID` is the latent grid (F, H, W); the caller sets it once per run (`readout.GRID = ...`).
"""
import numpy as np
import torch

GRID = (15, 16, 24)
EPS = 1e-12
BAND = (26, 27, 28, 29, 30, 31, 32, 33, 36)
TOPQ = 0.10
GMED_TOPQ = 0.1        # support gate for the geomedian pointer
Q_GRID = np.logspace(-6, 2, 25)
N_CELLS = GRID[1] * GRID[2]

# Stop words inside a prompt act as local background attractors -- the t2v concept pools
# CONTENT words only.
STOPWORDS = frozenset("""a an the and or but of in on at to for with from by as is are was
were be been being am it its it's this that these those there here he she they them his her
their our your my i you we while during over under near very really quite just only also
too then than so such both each any some no not into out up down off about against between
through after before again further once doing does did do have has had having will would
can could shall should may might must same own other another more most all""".split())


def content_columns(prompt: str, n_tok: int, tokenizer) -> torch.Tensor:
    """Columns (post-connector, right-padded order) of CONTENT-word tokens, sink excluded."""
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    if len(enc["input_ids"]) != n_tok:
        raise RuntimeError(f"tokenizer offsets length {len(enc['input_ids'])} != n_tok "
                           f"{n_tok} for {prompt[:60]!r}")
    cols = []
    for i, (a, b) in enumerate(enc["offset_mapping"]):
        if i < 2 or b <= a:                      # BOS + first token = sink; specials
            continue
        word = prompt[a:b].strip().lower()
        if len(word) > 1 and any(ch.isalnum() for ch in word) and word not in STOPWORDS:
            cols.append(i)
    if not cols:                                 # degenerate prompt: fall back to all valid
        print(f"[probe] WARNING: no content tokens in {prompt[:60]!r}; using all tokens",
              flush=True)
        return torch.arange(2, n_tok)
    return torch.tensor(cols, dtype=torch.long)


# ------------------------------------------------------------------ maps
def sharpen(p2: np.ndarray, topq: float | None) -> np.ndarray:
    """[2, F, H, W] per-source maps -> per-frame normalized, optionally top-q masked."""
    out = p2 / (p2.sum(axis=(-2, -1), keepdims=True) + EPS)
    if topq:
        for k in range(out.shape[0]):
            for f in range(out.shape[1]):
                thr = np.quantile(out[k, f], 1.0 - topq)
                out[k, f] = np.where(out[k, f] >= thr, out[k, f], 0.0)
        out = out / (out.sum(axis=(-2, -1), keepdims=True) + EPS)
    return out


def to_prob(maps: np.ndarray, family: str) -> np.ndarray:
    """[..., 2, F, H, W] raw family values -> per-token stem probabilities."""
    m = maps.astype(np.float64)
    if family.startswith("out"):
        m = m - m.max(axis=-4, keepdims=True)          # softmax over the stem axis
        e = np.exp(m)
        return e / (e.sum(axis=-4, keepdims=True) + EPS)
    m = m + EPS
    return m / m.sum(axis=-4, keepdims=True)           # pairwise share of nonneg mass


# ------------------------------------------------------------------ pointer + tracking
def weighted_geomedian_2d(ys, xs, wts, iters=64):
    """Weiszfeld weighted geometric median on grid coords."""
    py, px = float((ys * wts).sum() / wts.sum()), float((xs * wts).sum() / wts.sum())
    for _ in range(iters):
        d = np.sqrt((ys - py) ** 2 + (xs - px) ** 2) + 1e-9
        w = wts / d
        ny, nx = float((ys * w).sum() / w.sum()), float((xs * w).sum() / w.sum())
        if abs(ny - py) + abs(nx - px) < 1e-6:
            break
        py, px = ny, nx
    return py, px


def cv_matrices(dt: float, q: float):
    """State [y x vy vx]: transition F and white-noise-acceleration Q."""
    F = np.eye(4)
    F[:2, 2:] = dt * np.eye(2)
    Q = np.zeros((4, 4))
    Q[:2, :2] = q * dt**3 / 3 * np.eye(2)
    Q[:2, 2:] = Q[2:, :2] = q * dt**2 / 2 * np.eye(2)
    Q[2:, 2:] = q * dt * np.eye(2)
    return F, Q


def kalman_forward(z, R, dt, q):
    n = len(z)
    F, Q = cv_matrices(dt, q)
    H = np.zeros((2, 4))
    H[:, :2] = np.eye(2)
    m = np.zeros(4)
    m[:2] = z[0]
    P = np.eye(4) * 1e4
    ms = np.empty((n, 4)); Ps = np.empty((n, 4, 4))
    mp = np.empty((n, 4)); Pp = np.empty((n, 4, 4))
    ll = 0.0
    for k in range(n):
        if k > 0:
            m = F @ m
            P = F @ P @ F.T + Q
        mp[k], Pp[k] = m, P
        S = H @ P @ H.T + R[k]
        Sinv = np.linalg.inv(S)
        v = z[k] - H @ m
        ll += -0.5 * (v @ Sinv @ v + np.linalg.slogdet(2 * np.pi * S)[1])
        K = P @ H.T @ Sinv
        m = m + K @ v
        P = (np.eye(4) - K @ H) @ P
        ms[k], Ps[k] = m, P
    return ms, Ps, mp, Pp, ll


def rts_backward(ms, Ps, mp, Pp, dt, q):
    F, _ = cv_matrices(dt, q)
    sm = ms.copy(); sP = Ps.copy()
    for k in range(len(ms) - 2, -1, -1):
        G = Ps[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        sm[k] = ms[k] + G @ (sm[k + 1] - mp[k + 1])
        sP[k] = Ps[k] + G @ (sP[k + 1] - Pp[k + 1]) @ G.T
    return sm, sP


def measurements(m_stem: np.ndarray):
    """[F, H, W] per-frame normalized map -> (z [F,2], R [F,2,2]) geomedian + support cov."""
    yy, xx = np.meshgrid(np.arange(GRID[1]), np.arange(GRID[2]), indexing="ij")
    z = np.zeros((GRID[0], 2))
    R = np.zeros((GRID[0], 2, 2))
    for f in range(GRID[0]):
        m = m_stem[f].astype(np.float64)
        support = m if (m > 0).mean() <= GMED_TOPQ + 1e-6 else np.where(
            m >= np.quantile(m, 1.0 - GMED_TOPQ), m, 0.0)
        sel = support > 0
        sy, sx, sw = yy[sel], xx[sel], support[sel]
        py, px = weighted_geomedian_2d(sy, sx, sw)
        z[f] = (py, px)
        d = np.stack([sy - py, sx - px])
        R[f] = (d * sw) @ d.T / (sw.sum() + EPS)
    floor = max(1e-8, 1e-6 * float(np.trace(R.mean(axis=0)) / 2))
    return z, R + floor * np.eye(2)


# ------------------------------------------------------------------ GT-free confidence
def stem_signals(maps2k: np.ndarray, sib: np.ndarray) -> dict:
    """GT-free signals for one stem's [F,H,W] map (sib = sibling's map, same shape)."""
    z, R = measurements(maps2k)
    lls = [kalman_forward(z, R, 1.0, q)[4] for q in Q_GRID]
    q_ml = float(Q_GRID[int(np.argmax(lls))])
    jitter = float(np.linalg.norm(np.diff(z, axis=0), axis=1).mean())
    meas_sigma = float(np.mean(np.sqrt(np.trace(R, axis1=1, axis2=2) / 2)))
    ent, tcos, marg = [], [], []
    for f in range(GRID[0]):
        p = maps2k[f].astype(np.float64)
        tot = p.sum()
        if tot <= 0:
            continue
        p = p / tot
        nz = p[p > 0]
        ent.append(float(-(nz * np.log(nz)).sum() / np.log(N_CELLS)))
        sup = p > 0
        s_own = p[sup].sum()
        s_sib = sib[f].astype(np.float64)[sup].sum() / (sib[f].sum() + EPS)
        marg.append(float(abs(s_own / (s_own + s_sib + EPS) - 0.5) * 2))
        if f > 0 and maps2k[f - 1].sum() > 0:
            a = maps2k[f].astype(np.float64).ravel()
            b = maps2k[f - 1].astype(np.float64).ravel()
            tcos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + EPS)))
    return {"jitter": jitter, "meas_sigma": meas_sigma, "q_ml_log10": float(np.log10(q_ml)),
            "entropy": float(np.mean(ent)), "tconsist": float(np.mean(tcos)),
            "margin": float(np.mean(marg))}
