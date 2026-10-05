#!/usr/bin/env python
"""Lift localize.py's per-stem 2-D readout to 3-D tracks on MoGe-2 geometry (demo stage 4).

Per stem, per latent frame (anchor):
  chosen-channel attention map (maps.npz)  -> pre-gated to its top-2 % cells (--support-topq)
  -> support_gate + weighted geometric median over the cells' metric 3-D points (direction)
  -> range = the NEAREST significant mode of the metric depth in a 128 px window at the 2-D
     pointer (--range-mode depth-mode): the sound source is the foreground surface at the
     attention location, while background straight behind a subject shares its ray and would
     pull a pooled range toward it
  -> measurement covariance from that evidence: radial sigma = the mode's own MAD, lateral sigma
     = one cell's angular size at range, inflated by window coverage and by the stem's own
     silence (--evidence-gate: power ratio to the stem's p90 RMS around the anchor)
  -> silent anchors bridged by interpolation between voiced ones (--evidence-bridge)
  -> 3-D constant-velocity Kalman + RTS with per-clip ML process noise, one 3-sigma innovation-
     gated refit (a measurement inconsistent with the track's own motion, e.g. a subject that left
     the frame while its sound continues, is coasted over)
  -> cubic-Hermite resample of the posterior to every video frame.

The lifted clip is the original, static-camera clip, so the tracks are in its camera frame
(OpenCV: x right, y down, z forward); the audio stage re-expresses them relative to the authored
moving camera.

CPU, seconds.
  python demo/lift_tracks.py --localize-dir out/localize --depth-npz out/depth/depth.npz \
      --out out/tracks_3d.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.io import wavfile

VAE_TEMPORAL_STRIDE = 8
GMED_TOPQ = 0.1       # the localizer's own per-frame support gate
EPS = 1e-12


# ------------------------------------------------------------------ pooling
def weighted_mean(points: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = w / (w.sum() + EPS)
    return (w[:, None] * points).sum(axis=0)


def weighted_geometric_median(points: np.ndarray, w: np.ndarray,
                              n_iter: int = 200, tol: float = 1e-7) -> np.ndarray:
    """Weiszfeld iteration for the weighted geometric median of (N, 3) points."""
    w = w / (w.sum() + EPS)
    y = weighted_mean(points, w)
    for _ in range(n_iter):
        d = np.linalg.norm(points - y, axis=1)
        if (d < tol).any():
            # median coincides with a data point: it is the minimizer iff its
            # weight dominates; standard Weiszfeld treatment — return it.
            return points[int(np.argmin(d))].copy()
        inv = w / d
        y_new = (inv[:, None] * points).sum(axis=0) / inv.sum()
        if np.linalg.norm(y_new - y) < tol:
            return y_new
        y = y_new
    return y


def weighted_cov(points: np.ndarray, w: np.ndarray, mu: np.ndarray) -> np.ndarray:
    w = w / (w.sum() + EPS)
    d = points - mu
    return (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(axis=0)


def support_gate(m: np.ndarray) -> np.ndarray:
    """The localizer's own per-frame support gate (adaptive top-10 %): a map already
    sparser than the gate is used as-is; otherwise it is re-gated."""
    m = m.astype(np.float64)
    if (m > 0).mean() <= GMED_TOPQ + 1e-6:
        return m
    return np.where(m >= np.quantile(m, 1.0 - GMED_TOPQ), m, 0.0)


def pool_world(p_map: np.ndarray, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """(F,16,24) map + (F,16,24,3) metric points -> per-frame 3-D pointer and covariance."""
    n_frames = p_map.shape[0]
    track = np.empty((n_frames, 3))
    covs = np.empty((n_frames, 3, 3))
    regated = 0
    for f in range(n_frames):
        support = support_gate(p_map[f])
        regated += int(not np.array_equal(support, p_map[f].astype(np.float64)))
        sel = support > 0
        if not sel.any():
            raise ValueError(f"frame {f}: empty support -- a per-stem map cannot be empty; "
                             "the map upstream is degenerate, not the gate")
        pts = cells[f][sel].reshape(-1, 3).astype(np.float64)
        w = support[sel].reshape(-1)
        mu = weighted_geometric_median(pts, w)
        track[f] = mu
        covs[f] = weighted_cov(pts, w, mu)
    # Numerical floor only, scaled to the clip's own spread: a map concentrated in one cell
    # gives a singular R.
    floor = max(1e-8, 1e-6 * float(np.trace(covs.mean(axis=0)) / 3))
    return track, covs + floor * np.eye(3), {"frames_regated": regated}


# ------------------------------------------------------------------ smoothing
def _cv_matrices(dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """State [x y z vx vy vz]: transition F and white-noise-acceleration Q."""
    F = np.eye(6)
    F[:3, 3:] = dt * np.eye(3)
    Q = np.zeros((6, 6))
    Q[:3, :3] = q * dt**3 / 3 * np.eye(3)
    Q[:3, 3:] = Q[3:, :3] = q * dt**2 / 2 * np.eye(3)
    Q[3:, 3:] = q * dt * np.eye(3)
    return F, Q


def _kalman_forward(z: np.ndarray, R: np.ndarray, dt: float, q: float):
    """Forward pass. Returns filtered (means, covs), predicted (means, covs),
    and the total innovation log-likelihood."""
    n = len(z)
    F, Q = _cv_matrices(dt, q)
    H = np.zeros((3, 6))
    H[:, :3] = np.eye(3)
    m = np.zeros(6)
    m[:3] = z[0]
    P = np.eye(6) * 1e4  # diffuse init: first measurement dominates
    ms = np.empty((n, 6)); Ps = np.empty((n, 6, 6))
    mp = np.empty((n, 6)); Pp = np.empty((n, 6, 6))
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
        P = (np.eye(6) - K @ H) @ P
        ms[k], Ps[k] = m, P
    return ms, Ps, mp, Pp, ll


def _rts_backward(ms, Ps, mp, Pp, dt: float, q: float):
    n = len(ms)
    F, _ = _cv_matrices(dt, q)
    sm = ms.copy(); sP = Ps.copy()
    for k in range(n - 2, -1, -1):
        G = Ps[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        sm[k] = ms[k] + G @ (sm[k + 1] - mp[k + 1])
        sP[k] = Ps[k] + G @ (sP[k + 1] - Pp[k + 1]) @ G.T
    return sm, sP


def smooth_track(
    track: np.ndarray,          # (n, 3) pooled measurements
    covs: np.ndarray,           # (n, 3, 3) measurement covariances
    fps: float = 10.0,
    q_grid: np.ndarray | None = None,
) -> dict:
    """CV-Kalman + RTS with per-clip ML process noise. Returns posterior
    position track, position/velocity sigmas, and the fitted q."""
    n = len(track)
    dt = 1.0 / fps
    # Covariance floor: numerical only (singular R when attention sits in one
    # cell); scaled to the clip's measurement spread, not a tuned constant.
    floor = max(1e-8, 1e-6 * float(np.trace(covs.mean(axis=0)) / 3))
    R = covs + floor * np.eye(3)
    if q_grid is None:
        q_grid = np.logspace(-6, 2, 25)
    lls = [_kalman_forward(track, R, dt, q)[4] for q in q_grid]
    q_ml = float(q_grid[int(np.argmax(lls))])
    ms, Ps, mp, Pp, ll = _kalman_forward(track, R, dt, q_ml)
    sm, sP = _rts_backward(ms, Ps, mp, Pp, dt, q_ml)
    pos = sm[:, :3]
    vel = sm[:, 3:]
    pos_sigma = np.sqrt(np.trace(sP[:, :3, :3], axis1=1, axis2=2) / 3)
    vel_sigma = np.sqrt(np.trace(sP[:, 3:, 3:], axis1=1, axis2=2) / 3)
    speed = np.linalg.norm(vel, axis=1)
    return {
        "pos": pos, "vel": vel,
        "pos_sigma": pos_sigma, "vel_sigma": vel_sigma,
        "q_ml": q_ml, "loglik": float(ll),
        # inferred static-ness: is the velocity posterior consistent with 0?
        "static_zmax": float(np.max(speed / (vel_sigma + EPS))),
        "meas_sigma": np.sqrt(np.trace(R, axis1=1, axis2=2) / 3),
    }


def hermite_resample(pos: np.ndarray, vel: np.ndarray, n_out: int, fps: float) -> np.ndarray:
    """Cubic-Hermite resample of a CV posterior from n_in to n_out samples over the same span.

    Position AND velocity come from the RTS smoother, so this evaluates the continuous
    trajectory of the constant-velocity model that was already fitted; at the knots it
    reproduces the posterior exactly. `vel` is m/s and the Hermite tangent scale is the knot
    spacing dt.
    """
    n_in = len(pos)
    if n_in < 2:
        raise ValueError(f"cannot resample a {n_in}-sample track")
    dt = 1.0 / fps
    t_in = np.arange(n_in) * dt
    t_out = np.linspace(0.0, t_in[-1], n_out)
    idx = np.clip(np.searchsorted(t_in, t_out, side="right") - 1, 0, n_in - 2)
    s = (t_out - t_in[idx]) / dt
    s2, s3 = s ** 2, s ** 3
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    return (h00[:, None] * pos[idx] + h10[:, None] * vel[idx] * dt
            + h01[:, None] * pos[idx + 1] + h11[:, None] * vel[idx + 1] * dt)


# ------------------------------------------------------------------ geometry
def masked_area_cells(pts_hw3: np.ndarray, h: int, w: int,
                      stat: str = "mean") -> np.ndarray:
    """Per-cell area reduction of a camera-frame point map with NaN-masked pixels onto the
    localizer's cell grid. stat="mean" is the area mean (NaNs excluded by weighting);
    stat="median" is robust to subject/background depth mixing in boundary cells."""
    H, W = pts_hw3.shape[:2]
    assert H % h == 0 and W % w == 0, f"grid {h}x{w} must tile {H}x{W}"
    bh, bw = H // h, W // w
    blocks = pts_hw3.reshape(h, bh, w, bw, 3)
    if stat == "median":
        import warnings
        flat = blocks.transpose(0, 2, 1, 3, 4).reshape(h, w, bh * bw, 3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells
            # single-axis nanmedian = numpy's fast path
            return np.nanmedian(flat, axis=2)
    assert stat == "mean", stat
    fin = np.isfinite(blocks).all(-1, keepdims=True)
    num = np.where(fin, blocks, 0.0).sum(axis=(1, 3))
    den = fin.sum(axis=(1, 3)).astype(np.float64)
    with np.errstate(invalid="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def nearest_depth_mode(z_win: np.ndarray, min_mass: float
                       ) -> tuple[float, float, float] | None:
    """Nearest significant mode of a local depth sample -- the RGB-D-tracking classic for
    'object depth at a 2-D point': the depth distribution around the pointer is multi-modal
    (subject surface vs background); a mean or median mixes the modes, the NEAREST mode with
    non-trivial mass is the foreground surface. Returns (range_m, mass fraction of the chosen
    mode, radial sigma of the mode) or None if no finite depth.
    """
    from scipy.signal import find_peaks
    z = z_win[np.isfinite(z_win)].ravel()
    if z.size == 0:
        return None
    lo, hi = 0.5, 200.0
    logz = np.log(np.clip(z, lo, hi))
    edges = np.linspace(np.log(lo), np.log(hi), 61)
    hist, _ = np.histogram(logz, bins=edges)
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
    sm = np.convolve(hist.astype(np.float64), kernel / kernel.sum(),
                     mode="same")
    peaks, _ = find_peaks(np.concatenate([[0.0], sm, [0.0]]))
    peaks -= 1
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(sm))])
    # basin boundaries = minima between consecutive peaks
    bounds = [0]
    for a, b in zip(peaks[:-1], peaks[1:]):
        bounds.append(a + int(np.argmin(sm[a:b + 1])))
    bounds.append(len(sm))
    total = hist.sum()
    for j, p in enumerate(peaks):          # nearest first (bins are sorted)
        b0, b1 = bounds[j], bounds[j + 1]
        mass = hist[b0:b1].sum() / total
        if mass < min_mass:
            continue
        sel = logz[(logz >= edges[b0]) & (logz < edges[min(b1, 60)])]
        assert sel.size, "basin with mass but no samples"
        r = float(np.exp(np.median(sel)))
        # radial sigma of the MODE ITSELF (robust MAD of its basin) -- the measurement noise;
        # the pooled support's depth variance is background-contaminated
        sig = float(max(0.25, 1.4826 * np.median(np.abs(np.exp(sel) - r))))
        return r, float(mass), sig
    return None


def anchor_energy_inflation(wav_path: Path, n_anchors: int, n_frames: int,
                            frame_rate: float, max_inflation: float
                            ) -> np.ndarray:
    """Per-anchor measurement-covariance inflation from the stem's OWN energy: a silent source
    gives no evidence that its attention pointer is on it. inflation = (rms_ref / rms)^2
    (power ratio; rms_ref = p90 of anchor RMS), clipped to [1, max_inflation] -- continuous, no
    on/off threshold; at -20 dB below active level the measurement counts 100x less."""
    sr, x = wavfile.read(wav_path)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(1)
    if np.abs(x).max() > 2.0:            # integer PCM -> float
        x /= np.abs(np.iinfo(np.int16).max)
    half = 0.5 * VAE_TEMPORAL_STRIDE / frame_rate
    rms = np.empty(n_anchors)
    for lf in range(n_anchors):
        t = min(lf * VAE_TEMPORAL_STRIDE, n_frames - 1) / frame_rate
        a = max(0, int((t - half) * sr))
        b = min(len(x), max(a + 1, int((t + half) * sr)))
        rms[lf] = np.sqrt(np.mean(x[a:b] ** 2))
    ref = np.percentile(rms, 90)
    assert ref > 0, f"{wav_path}: dead stem (p90 RMS == 0)"
    return np.clip((ref / np.maximum(rms, 1e-12)) ** 2, 1.0, max_inflation)


def fill_nan_cells(cells: np.ndarray) -> np.ndarray:
    """Fill fully-masked cells (e.g. sky) from the nearest finite cell; the pooling assumes
    dense geometry."""
    bad = ~np.isfinite(cells).all(-1)
    if not bad.any():
        return cells
    assert not bad.all(), "every cell masked — no geometry at all"
    idx = ndimage.distance_transform_edt(bad, return_distances=False,
                                         return_indices=True)
    return cells[tuple(idx)]


def spherical(pts: np.ndarray) -> dict:
    """OpenCV camera frame (x right, y down, z forward) -> az (+RIGHT of the
    optical axis), el (+UP), range in metres."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.linalg.norm(pts, axis=1)
    return {"az_deg": np.degrees(np.arctan2(x, z)).tolist(),
            "el_deg": np.degrees(np.arcsin(np.clip(-y / np.maximum(r, 1e-9),
                                                   -1, 1))).tolist(),
            "range_m": r.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--localize-dir", type=Path, required=True,
                    help="localize.py output (track.json, maps.npz, stem_{0,1}.wav)")
    ap.add_argument("--depth-npz", type=Path, required=True,
                    help="moge_depth.py output for the same (original) clip")
    ap.add_argument("--out", type=Path, required=True, help="tracks json")
    ap.add_argument("--frame-rate", type=float, default=24.0)
    ap.add_argument("--evidence-gate", action=argparse.BooleanOptionalAction, default=True,
                    help="inflate measurement covariance on low-energy anchors from the "
                         "stem's own separated audio (default: on)")
    ap.add_argument("--evidence-max-inflation", type=float, default=1e4,
                    help="clip ceiling for the energy inflation factor "
                         "(default 1e4 = silence coasted over)")
    ap.add_argument("--cell-stat", choices=("mean", "median"), default="mean",
                    help="per-cell reduction of the MoGe point map (default: mean)")
    ap.add_argument("--range-mode", choices=("pooled", "depth-mode"),
                    default="depth-mode",
                    help="'depth-mode' (default) = range from the NEAREST significant "
                         "depth mode in a window at the 2-D pointer; 'pooled' = range "
                         "from the pooled geometric median")
    ap.add_argument("--mode-window-px", type=int, default=128,
                    help="half-size of the pointer window for depth-mode "
                         "range (video px, default 128)")
    ap.add_argument("--mode-min-mass", type=float, default=0.15,
                    help="minimum pixel-mass fraction for a depth mode to "
                         "count (default 0.15)")
    ap.add_argument("--evidence-bridge", action=argparse.BooleanOptionalAction, default=True,
                    help="zero-evidence anchors are EXCLUDED from the fit: their "
                         "measurements are replaced by interpolation between voiced "
                         "anchors (constant hold at the ends). Interpolate, never "
                         "extrapolate (default: on; needs --evidence-gate)")
    ap.add_argument("--support-topq", type=float, default=0.02,
                    help="pre-gate each attention map to its top-q cells BEFORE pooling "
                         "(default 0.02): the attention PEAK sits on the subject while a "
                         "top-10 %% support can blanket background")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    track = json.loads((args.localize_dir / "track.json").read_text())
    maps = np.load(args.localize_dir / "maps.npz")
    npz = np.load(args.depth_npz)
    dep16, msk, intr = npz["depth"], npz["mask"], npz["intrinsics"]
    F, H, W = dep16.shape

    def depth_frame(f: int) -> np.ndarray:
        return np.where(msk[f], np.asarray(dep16[f], dtype=np.float64), np.nan)
    n_anchors, gh, gw = track["grid_FHW"]
    tw, th = track["frame_wh"]
    assert abs(tw / th - W / H) < 1e-6, \
        f"aspect mismatch localize {tw}x{th} vs video {W}x{H}"
    # the lifted clip IS the fixed frame (original static-camera clip): C_tgt = I
    fx = float(intr[0][0, 0]) * W
    poses = [{"frame": f + 1, "az": 0.0, "el": 0.0, "dist": 1.0,
              "vs": 0.0, "pivot": [0.0, 0.0, 1.0],
              "C_tgt": np.eye(4).tolist()} for f in range(F)]
    cx, cy = W / 2.0, H / 2.0

    # camera-frame point cells per anchor frame
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    cells = np.empty((n_anchors, gh, gw, 3))
    for lf in range(n_anchors):
        f = min(lf * VAE_TEMPORAL_STRIDE, F - 1)
        z = depth_frame(f)
        pts = np.stack([(uu - cx) / fx * z, (vv - cy) / fx * z, z], -1)
        cells[lf] = fill_nan_cells(
            masked_area_cells(pts, gh, gw, stat=args.cell_stat))

    fps_latent = args.frame_rate / VAE_TEMPORAL_STRIDE
    variant = (f"evidence_gate={args.evidence_gate} cell_stat={args.cell_stat} "
               f"range_mode={args.range_mode} support_topq={args.support_topq} "
               f"bridge={args.evidence_bridge}")
    out = {"grid_FHW": track["grid_FHW"],
           "render_wh": [W, H], "fx_pixels": fx,
           "readout": f"{variant}: support_gate -> weighted_geometric_"
                      "median/weighted_cov -> smooth_track (3D CV-Kalman"
                      "+RTS) -> hermite_resample; MoGe cells",
           "evidence_gate": args.evidence_gate,
           "cell_stat": args.cell_stat,
           "range_mode": args.range_mode,
           "pchip_frames": F, "stems": {}}
    for si in (0, 1):
        stem = track[f"stem{si}"]
        p_map = np.asarray(maps[stem["channel"]][si], dtype=np.float64)
        assert p_map.shape == (n_anchors, gh, gw), \
            f"stem{si}: map {p_map.shape} != {(n_anchors, gh, gw)}"
        if args.support_topq is not None:
            # mirror support_gate's adaptive rule at a tighter q; the
            # module's own gate then passes the sparser map through
            tq = args.support_topq
            for lf in range(n_anchors):
                m = p_map[lf]
                if (m > 0).mean() > tq + 1e-6:
                    p_map[lf] = np.where(
                        m >= np.quantile(m, 1.0 - tq), m, 0.0)
        meas, covs, pool_diag = pool_world(p_map, cells)
        n_gated = 0
        if args.evidence_gate:
            infl = anchor_energy_inflation(
                args.localize_dir / f"stem_{si}.wav", n_anchors,
                F, args.frame_rate, args.evidence_max_inflation)
            covs = covs * infl[:, None, None]
            n_gated = int((infl > 10).sum())
        mode_diag = []
        if args.range_mode == "depth-mode":
            rw = args.mode_window_px
            for lf in range(n_anchors):
                f = min(lf * VAE_TEMPORAL_STRIDE, F - 1)
                u, v = (np.asarray(stem["pixels_xy"][lf]) * (W / tw))
                u, v = int(round(u)), int(round(v))
                y0, x0 = max(0, v - rw), max(0, u - rw)
                zw = np.where(
                    msk[f, y0:v + rw + 1, x0:u + rw + 1],
                    np.asarray(dep16[f, y0:v + rw + 1, x0:u + rw + 1],
                               np.float64), np.nan)
                res = nearest_depth_mode(zw, args.mode_min_mass)
                if res is None:      # fully masked window: coast anchor
                    covs[lf] = covs[lf] * args.evidence_max_inflation
                    mode_diag.append(None)
                    continue
                r_m, mass, sig_r = res
                # window-evidence guard: a truncated (frame-edge) or mostly-masked window
                # may not contain the subject at all -- its mode is confident but unfounded.
                # Trust scales with coverage of the NOMINAL window.
                cover = (np.isfinite(zw).sum()
                         / float((2 * rw + 1) ** 2))
                guard = float(np.clip((1.0 / max(cover, 1e-3)) ** 2,
                                      1.0, args.evidence_max_inflation))
                s = r_m / max(np.linalg.norm(meas[lf]), 1e-9)
                meas[lf] = meas[lf] * s
                # covariance FROM THE EVIDENCE, not from the pooled support's depth variance
                # (background-contaminated, which lets the smoother free-run and invent
                # motion). Radial = the mode's own spread; lateral = one cell's angular size
                # at range.
                sig_lat = max(0.30, (64.0 / fx) * 2.0 * r_m)
                covs[lf] = (np.diag([sig_lat ** 2, sig_lat ** 2,
                                     sig_r ** 2])
                            * (infl[lf] if args.evidence_gate else 1.0)
                            * guard)
                mode_diag.append({"anchor": lf, "range_m": r_m,
                                  "mass": round(mass, 3),
                                  "sigma_r": round(sig_r, 2),
                                  "coverage": round(cover, 3),
                                  "radial_rescale": round(s, 3)})
        n_bridged = 0
        if (args.evidence_bridge and args.evidence_gate
                and 0 < n_gated <= n_anchors - 2):
            # interpolate, never extrapolate: silent anchors carry NO information about
            # position -- bridge them between voiced neighbours (constant hold past the
            # ends) so the line fit is not tilted by muted junk
            voiced = infl <= 10.0
            idx = np.arange(n_anchors, dtype=np.float64)
            for d in range(3):
                meas[~voiced, d] = np.interp(
                    idx[~voiced], idx[voiced], meas[voiced, d])
            covs[~voiced] = covs[voiced].mean(0) * 4.0
            n_bridged = int((~voiced).sum())
        sm = smooth_track(meas, covs, fps=fps_latent)
        # innovation gate (robust-Kalman second pass): a measurement wildly inconsistent with
        # the track's own motion -- e.g. the depth mode jumping 12 -> 49 m in one anchor because
        # the subject LEFT THE FRAME while its sound continues -- is rejected (cov -> ceiling)
        # and the track coasts instead of chasing pixels the subject no longer occupies
        n_inno = 0
        resid = np.linalg.norm(np.asarray(sm["pos"]) - meas, axis=1)
        sig = np.sqrt(np.array([np.trace(c) / 3 for c in covs]))
        bad = resid > 3.0 * np.maximum(sig, 0.3)
        if bad.any() and not bad.all():
            covs[bad] = covs[bad] * args.evidence_max_inflation
            n_inno = int(bad.sum())
            sm = smooth_track(meas, covs, fps=fps_latent)
        pos_cam = hermite_resample(sm["pos"], sm["vel"], F, fps_latent)
        pos_src = np.empty_like(pos_cam)
        for f in range(F):
            C = np.asarray(poses[f]["C_tgt"])
            pos_src[f] = C[:3, :3] @ pos_cam[f] + C[:3, 3]
        rng = np.linalg.norm(pos_cam, axis=1)
        print(f"[lift] stem{si}: range p10/p50/p90 "
              f"{np.percentile(rng, 10):.1f}/{np.percentile(rng, 50):.1f}/"
              f"{np.percentile(rng, 90):.1f} m, q_ml_3d {sm['q_ml']:.3g}, "
              f"regated {pool_diag['frames_regated']}/{n_anchors}, "
              f"energy-gated {n_gated}/{n_anchors}, "
              f"bridged {n_bridged}/{n_anchors}, "
              f"innovation-gated {n_inno}/{n_anchors}")
        med = np.median(pos_src, 0)
        out["stems"][f"stem{si}"] = {
            "flagged": stem["flagged"], "channel": stem["channel"],
            "q_ml_3d": float(sm["q_ml"]),
            "anchors_energy_gated": n_gated,
            "anchors_voiced": ([bool(v) for v in (infl <= 10.0)]
                               if args.evidence_gate
                               else [True] * n_anchors),
            # display visibility with hysteresis: close 1-2 anchor gaps (drum hits must not
            # blink) and dilate 1 anchor at edges; only SUSTAINED silence hides a marker
            "anchors_visible": (
                [bool(v) for v in ndimage.binary_dilation(
                    ndimage.binary_closing(
                        infl <= 10.0, structure=np.ones(3, bool)),
                    structure=np.ones(3, bool))]
                if args.evidence_gate else [True] * n_anchors),
            "depth_modes": mode_diag,
            "cam_moving": spherical(pos_cam),
            "cam_moving_xyz": pos_cam.tolist(),
            "src_fixed_xyz": pos_src.tolist(),
            "src_fixed": spherical(pos_src),
            "src_fixed_median_xyz": med.tolist(),
            "src_fixed_p90_drift_m": float(np.percentile(
                np.linalg.norm(pos_src - med, axis=1), 90)),
        }
    out["camera_src_xyz"] = [
        (np.asarray(p["C_tgt"])[:3, 3]).tolist() for p in poses]
    args.out.write_text(json.dumps(out))
    print(f"[lift] -> {args.out}")


if __name__ == "__main__":
    main()
