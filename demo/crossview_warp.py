# Derived from ComfyUI-CrossViewWarp by cseti007
# (https://github.com/cseti007/ComfyUI-CrossViewWarp, commit 3266a83).
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the
# License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing permissions
# and limitations under the License.
#
# Modified by the SepGen authors: ComfyUI plumbing removed, sides_pivot() and the per-frame
# pose output of build_warp() added (see the module docstring).
"""CrossView-Warp conditioning builder — port of the ComfyUI node's warp core.

Builds the depth-warp reference guide the CrossView-Warp v2 IC-LoRA expects:
each frame reprojected into a (possibly keyframed) target camera, MAGENTA
disocclusion holes, painter z-buffer splat. Only the metric-depth path is used
here (MoGe-2 metres + validity mask); the brightness-depth normaliser is kept
for completeness.

PROVENANCE: ported from crossview_warp_node.py of
  https://github.com/cseti007/ComfyUI-CrossViewWarp
  commit 3266a83a84eaa6833a9ebc38ac7cdd8789e44f5b (2026-08-17), Apache-2.0.
The warp/keyframe/pose math is kept VERBATIM (the node docstring: it mirrors
the LoRA's training-time builder exactly — same magenta fill, painter
z-buffer, splat, intrinsics). Only the ComfyUI plumbing (node class, sockets,
preview sink, torch tensor I/O) is stripped; NumPy in, NumPy out.
Two deliberate behavioural choices, both documented upstream:
  * keep_source_aim defaults to TRUE here (upstream keeps False only for
    saved-workflow compatibility; True matches the training data and measures
    28.9 vs 42.3 against the training warps).
  * the auto pivot is upstream's own estimator (median 3-D point of the
    trimmed central region of frame 0) — not a nearest-cluster point, which
    upstream measured worse.
Additions: sides_pivot() (a pivot estimator for scenes with one subject on
each side and an empty centre), and build_warp() also returns the per-frame
resolved pose (az/el/dist/pivot AND the 4x4 camera-to-source extrinsic C_tgt)
that the audio stage uses to express the localized sources relative to the
moving camera.

Module only — no CLI. Driver: build_warp_guide.py (guide + previews).
"""
from __future__ import annotations

import os

import numpy as np

MAGENTA = np.array([255, 0, 255], dtype=np.uint8)

# Measured on a 96-core host under load: the numba kernel ran 17.9 s/frame
# (plus ~90 s JIT), while upstream's byte-identical numpy fallback ran
# ~1-2 s/frame. Numpy is therefore the DEFAULT here; set
# CROSSVIEW_USE_NUMBA=1 to re-enable the kernel (kept verbatim).
_WANT_NUMBA = os.environ.get("CROSSVIEW_USE_NUMBA") == "1"
try:
    if not _WANT_NUMBA:
        raise ImportError("numba disabled by default; see note above")
    from numba import njit

    @njit(cache=False)   # JIT-once per process; disk cache pickles module paths
    def _splat_kernel(tu0, tv0, cols, warp, splat, H, W):
        # z-sorted painter order, offset passes in dy/dx order - must stay
        # identical to the numpy fallback below (training warp format)
        n = tu0.shape[0]
        for dy in range(-splat, splat + 1):
            for dx in range(-splat, splat + 1):
                for i in range(n):
                    x = tu0[i] + dx
                    y = tv0[i] + dy
                    if 0 <= x < W and 0 <= y < H:
                        warp[y, x, 0] = cols[i, 0]
                        warp[y, x, 1] = cols[i, 1]
                        warp[y, x, 2] = cols[i, 2]
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


# --- warp math (verbatim from upstream, which mirrors monocular_warp.py) -----

def _warp_frame(rgb_ref, depth_ref, C_ref, C_tgt, fx_pix, splat, cx, cy):
    H, W = depth_ref.shape
    fy_pix = fx_pix
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_ref
    fin = np.isfinite(z) & (z > 0)
    thr = np.percentile(z[fin], 99.5) if fin.any() else 0
    fin = fin & (z < thr)
    Xc = np.stack([(u - W / 2.0) / fx_pix * z, (v - H / 2.0) / fy_pix * z, z], -1).reshape(-1, 3)
    Xw = (C_ref[:3, :3] @ Xc.T).T + C_ref[:3, 3]
    Ci = np.linalg.inv(C_tgt)
    Xd = (Ci[:3, :3] @ Xw.T).T + Ci[:3, 3]
    zt = Xd[:, 2]
    # Metric geometry carries NaN where the model marked sky/invalid; park the
    # resulting non-finite projections far outside the frame (dropped by `val`)
    # instead of letting numpy warn per frame.
    with np.errstate(invalid="ignore", divide="ignore"):
        uf = Xd[:, 0] / zt * fx_pix + cx
        vf = Xd[:, 1] / zt * fy_pix + cy
    ui = np.round(np.nan_to_num(uf, nan=-1e6, posinf=1e6, neginf=-1e6)).astype(int)
    vi = np.round(np.nan_to_num(vf, nan=-1e6, posinf=1e6, neginf=-1e6)).astype(int)
    val = fin.ravel() & (zt > 0)
    order = np.argsort(-zt)
    sel = order[val[order]]
    # gather indices/colors once; the splat passes only offset them
    tu0 = ui[sel]
    tv0 = vi[sel]
    cols = np.ascontiguousarray(rgb_ref.reshape(-1, 3)[sel])
    warp = np.tile(MAGENTA, (H, W, 1))
    if _HAVE_NUMBA:
        _splat_kernel(tu0.astype(np.int64), tv0.astype(np.int64), cols,
                      warp, int(splat), H, W)
    else:
        flat = warp.reshape(-1, 3)
        for dy in range(-splat, splat + 1):
            for dx in range(-splat, splat + 1):
                tu = tu0 + dx
                tv = tv0 + dy
                ok = (tu >= 0) & (tu < W) & (tv >= 0) & (tv < H)
                flat[tv[ok] * W + tu[ok]] = cols[ok]
    return warp.astype(np.uint8)


def _look_at(eye, target, world_down=np.array([0.0, 1.0, 0.0])):
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-9)
    right = np.cross(world_down, f)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        # Looking straight up/down collapses the basis; any perpendicular axis
        # works — the world forward axis is the natural pick.
        right = np.cross(np.array([0.0, 0.0, 1.0]), f)
        rn = np.linalg.norm(right)
    right = right / (rn + 1e-9)
    down = np.cross(f, right)
    C = np.eye(4)
    C[:3, 0], C[:3, 1], C[:3, 2], C[:3, 3] = right, down, f, eye
    return C


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _depth_to_z(depth_bhw, invert, ratio=4.0):
    """Normalise a brightness-like depth stack globally -> z (near small, far
    large). Unused on the metric path; kept verbatim for fidelity/fallback."""
    d = depth_bhw.astype(np.float64)
    if invert:
        d = -d
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    dn = np.clip((d - lo) / (hi - lo + 1e-9), 0, 1)
    r = max(float(ratio), 1.01)
    return 1.0 / (1.0 / r + (1.0 - 1.0 / r) * dn)


# --- keyframe interpolation (verbatim) ---------------------------------------

def _wrap_deg(a):
    """Wrap degrees to (-180, 180]."""
    return ((a + 180.0) % 360.0) - 180.0


def _ease(t, mode):
    if mode == "ease_in_out":
        return 0.5 - 0.5 * np.cos(np.pi * t)
    if mode == "ease_in":
        return t * t
    if mode == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    return t   # "linear" / "smooth" / unknown -> identity


def _unwrap_seq(degs):
    """Unwrap angles so each step takes the short way round (seam-safe lerp)."""
    out = [float(degs[0])]
    for d in degs[1:]:
        out.append(out[-1] + _wrap_deg(float(d) - out[-1]))
    return out


def _catmull(p0, p1, p2, p3, u):
    """Uniform Catmull-Rom segment: hits p1 at u=0 and p2 at u=1."""
    return 0.5 * ((2.0 * p1)
                  + (-p0 + p2) * u
                  + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u * u
                  + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u * u * u)


def _seg_value(vals, seg, u, smooth):
    """Value between vals[seg] and vals[seg+1] at local u; lerp or Catmull-Rom
    with end-point reflection (identical to lerp at 2 keyframes)."""
    p1, p2 = vals[seg], vals[seg + 1]
    if not smooth or len(vals) < 3:
        return p1 + (p2 - p1) * u
    p0 = vals[seg - 1] if seg > 0 else p1 + (p1 - p2)
    p3 = vals[seg + 2] if seg + 2 < len(vals) else p2 + (p2 - p1)
    return _catmull(p0, p1, p2, p3, u)


def parse_keyframes(data, frame_count, default_vs=0.0,
                    default_pivot=(0.0, 0.0, 1.05)):
    """Validate a keyframe list (already-parsed JSON) into sorted keyframe
    dicts. `f` is 1-based. Missing vs/px/py/pz inherit the defaults. Malformed
    input raises — a silently-wrong camera move is worse than a crash."""
    if not data:
        return []
    if not isinstance(data, list):
        raise ValueError("keyframes must be a list of keyframe objects")
    out = []
    for i, kf in enumerate(data):
        if not isinstance(kf, dict):
            raise ValueError(f"keyframe #{i} is not an object")
        f = int(round(float(kf["f"])))
        az, el, dist = float(kf["az"]), float(kf["el"]), float(kf["dist"])
        vs = float(kf.get("vs", default_vs))
        px = float(kf.get("px", default_pivot[0]))
        py = float(kf.get("py", default_pivot[1]))
        pz = float(kf.get("pz", default_pivot[2]))
        if f < 1:
            raise ValueError(f"keyframe #{i} sits at frame {f}; frames are 1-based")
        if f > frame_count:
            raise ValueError(f"keyframe #{i} sits at frame {f}, clip has "
                             f"{frame_count} frames")
        out.append({"f": f, "az": az, "el": el, "dist": dist, "vs": vs,
                    "px": px, "py": py, "pz": pz})
    out.sort(key=lambda k: k["f"])
    seen = [k["f"] for k in out]
    if len(set(seen)) != len(seen):
        raise ValueError("two keyframes share the same frame number")
    return out


def _prepare_path(kfs):
    """Pre-split keyframes into per-channel arrays (azimuth unwrapped once)."""
    out = {"f": [k["f"] for k in kfs], "az": _unwrap_seq([k["az"] for k in kfs])}
    for key in ("el", "dist", "vs", "px", "py", "pz"):
        out[key] = [k[key] for k in kfs]
    return out


def _sample_path(path, frame, easing, smooth):
    """Camera pose dict at a 1-based frame. Held (not extrapolated) outside the
    keyframed span. Catmull-Rom overshoot is clamped where physical."""
    fs = path["f"]
    KEYS = ("az", "el", "dist", "vs", "px", "py", "pz")
    if frame <= fs[0]:
        return {k: path[k][0] for k in KEYS}
    if frame >= fs[-1]:
        return {k: path[k][-1] for k in KEYS}
    seg = 0
    for i in range(len(fs) - 1):
        if fs[i] <= frame <= fs[i + 1]:
            seg = i
            break
    u = _ease((frame - fs[seg]) / float(fs[seg + 1] - fs[seg]), easing)
    az = _wrap_deg(_seg_value(path["az"], seg, u, smooth))
    return {
        "az": az,
        "el": float(np.clip(_seg_value(path["el"], seg, u, smooth), -90.0, 90.0)),
        "dist": float(np.clip(_seg_value(path["dist"], seg, u, smooth), 0.1, 3.0)),
        "vs": float(np.clip(_seg_value(path["vs"], seg, u, smooth), -1.0, 1.0)),
        "px": float(_seg_value(path["px"], seg, u, smooth)),
        "py": float(_seg_value(path["py"], seg, u, smooth)),
        # a pivot at or behind the camera has no orbit to define
        "pz": float(max(_seg_value(path["pz"], seg, u, smooth), 0.01)),
    }


def _orbit_C_tgt(az_deg, el_deg, dist, pivot, aim=None):
    """Camera pose orbiting `pivot`, looking at `aim` (default: the pivot).
    Angle signs negated so +azimuth orbits RIGHT, +elevation RISES (OpenCV
    frame). Upstream measured keeping the SOURCE aim (see build_warp) closer
    to the training warps than aiming at the pivot."""
    R_orbit = _rot_y(np.radians(-az_deg)) @ _rot_x(np.radians(-el_deg))
    eye = pivot + dist * (R_orbit @ (-pivot))
    return _look_at(eye, pivot if aim is None else aim)


# --- pivot estimation (extracted verbatim from upstream build()) -------------

def auto_pivot(z0, fx, cx, cy):
    """Upstream's pivot estimator: median 3-D point of the trimmed central
    region of frame 0 (scene-level aim; a nearest-cluster pivot would zoom the
    orbit into the subject)."""
    H, W = z0.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    central = np.zeros_like(z0, bool)
    central[H // 8: 4 * H // 5, W // 5: 4 * W // 5] = True
    fin = np.isfinite(z0) & (z0 > 0) & central
    fin &= z0 < np.percentile(z0[fin], 95.0)
    Xw0 = np.stack([(uu - cx) / fx * z0, (vv - cy) / fx * z0, z0], -1)
    return np.median(Xw0[fin], axis=0)


def sides_pivot(z0, fx, cx, cy):
    """Pivot estimator for two-flanking-subject scenes (sources
    left+right, CENTER EMPTY — upstream's central median lands 28-77 m deep
    in the background there, blowing up orbit radius and hole fraction).
    Median 3-D point over the left+right thirds (sky/ground rows trimmed,
    far tail cut at that region's P70) — i.e. the subjects' own depth, which
    is what the LoRA's training orbits circle. Returned pivot sits ON the
    optical axis at that depth (the orbit should circle the acoustic pair,
    not one of them)."""
    H, W = z0.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    region = np.zeros_like(z0, bool)
    region[H // 8: 4 * H // 5, : W // 3] = True
    region[H // 8: 4 * H // 5, 2 * W // 3:] = True
    fin = np.isfinite(z0) & (z0 > 0) & region
    fin &= z0 < np.percentile(z0[fin], 70.0)
    Xw0 = np.stack([(uu - cx) / fx * z0, (vv - cy) / fx * z0, z0], -1)
    med = np.median(Xw0[fin], axis=0)
    return np.array([0.0, med[1], med[2]])


# --- driver ------------------------------------------------------------------

def fx_from_hfov(width, hfov_deg):
    """Focal in pixels from a horizontal field of view (upstream default 50)."""
    return width / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))


def build_warp(rgb, metric_z, keyframes, *, fx, easing="linear",
               keep_source_aim=True, splat=2, progress=None):
    """The full keyframed warp pass.

    Args:
      rgb: [B,H,W,3] uint8 source frames.
      metric_z: [B,H,W] float metric depth in metres, NaN where invalid
        (MoGe-2 mask applied by the caller). NaNs become magenta holes.
      keyframes: validated list from parse_keyframes() — every keyframe
        carries its pivot (px/py/pz in metres, source camera frame). At least
        one keyframe required; one keyframe = static pose.
      fx: focal length in pixels (fx_from_hfov, or MoGe intrinsics * W).
      easing: linear | ease_in | ease_out | ease_in_out | smooth
        ("smooth" = Catmull-Rom through the keyframes, easing identity).
      keep_source_aim: True (default) keeps the source camera's aim point so
        the original framing carries over — what the LoRA's training data does
        (upstream: 28.9 vs 42.3 against the training warps). False re-aims at
        the pivot (subject snaps to centre).
      splat: painter splat radius (upstream call site uses 2).
      progress: optional callable(i) per frame (tqdm hook).

    Returns (warp [B,H,W,3] uint8, poses) where poses[i] is the resolved pose
    for frame i: az/el/dist/vs, pivot [3], and C_tgt — the 4x4 camera-to-
    source-frame extrinsic, saved for the 3-D lift's inverse transform.
    """
    B, H, W = rgb.shape[:3]
    if not keyframes:
        raise ValueError("build_warp needs at least one keyframe")
    path = _prepare_path(keyframes)
    smooth_path = easing == "smooth"
    cx = W / 2.0
    cy = H / 2.0
    C_ref = np.eye(4)

    def aim_of(pivot):
        # Source camera sits at the origin looking down +Z; keeping its aim
        # means a target on that axis at the pivot's own distance (upstream
        # measured this against centre-depth aiming; pivot distance wins).
        if not keep_source_aim:
            return None
        return np.array([0.0, 0.0, max(float(np.linalg.norm(pivot)), 1e-3)])

    warp_frames = []
    poses = []
    for i in range(B):
        fr = _sample_path(path, i + 1, easing, smooth_path)
        pivot = np.array([fr["px"], fr["py"], fr["pz"]], dtype=np.float64)
        C_tgt = _orbit_C_tgt(fr["az"], fr["el"], fr["dist"], pivot, aim_of(pivot))
        cy_i = cy + fr["vs"] * H   # keyframable vertical lens shift
        warp_frames.append(_warp_frame(rgb[i], metric_z[i], C_ref, C_tgt,
                                       fx, splat, cx, cy_i))
        poses.append({"frame": i + 1, "az": fr["az"], "el": fr["el"],
                      "dist": fr["dist"], "vs": fr["vs"],
                      "pivot": pivot.tolist(), "C_tgt": C_tgt.tolist()})
        if progress is not None:
            progress(i)
    return np.stack(warp_frames, 0), poses


# --- orbit-view diagram (verbatim port; documentation artifact) --------------

def _dist_scale(d):
    """Source-distance ratio (0.1..3.0) -> canvas radius multiplier (piecewise
    so dist=1.0 lands exactly on the shell)."""
    d = float(d)
    if d <= 1.0:
        return 0.45 + 0.55 * d
    return 1.0 + 0.25 * (d - 1.0)


def orbit_view_image(azimuth, elevation, distance, size=512, kfs=None, smooth=False):
    """Front view of the orbit globe documenting the camera setup (1:1 port of
    the upstream widget render; keyframed runs draw the whole green path)."""
    from PIL import Image, ImageDraw, ImageFont
    W = H = size
    k = size / 300.0
    S = min(W - 16 * k, H - 12 * k)
    cx, cy = W / 2.0, H / 2.0
    R = (S / 2.0 - 4 * k) * 0.62
    yaw, tilt = 0.24, 0.20

    ZONE_GREEN, ZONE_YELLOW = (45, 30, 20), (90, 45, 35)
    C_GREEN, C_YELLOW = (80, 200, 120), (230, 200, 90)

    def in_ellipse(az, el, zone):
        A, Eup, Edn = zone
        an = az / A
        en = el / Eup if el >= 0 else el / Edn
        return an * an + en * en <= 1

    def zone_color(az, el):
        if in_ellipse(az, el, ZONE_GREEN):
            return C_GREEN
        if in_ellipse(az, el, ZONE_YELLOW):
            return C_YELLOW
        return None

    def rot(p):
        x, y, z = p
        cyw, syw = np.cos(yaw), np.sin(yaw)
        ct, st = np.cos(tilt), np.sin(tilt)
        xr = x * cyw + y * syw
        yr = -x * syw + y * cyw
        yv = yr * ct - z * st
        zv = yr * st + z * ct
        return xr, yv, zv

    def pt(a_deg, e_deg):
        a, e = np.radians(a_deg), np.radians(e_deg)
        xr, yv, zv = rot((np.cos(e) * np.sin(a), -np.cos(e) * np.cos(a), np.sin(e)))
        return cx + R * xr, cy - R * zv, yv

    img = Image.new("RGBA", (W, H), (27, 27, 31, 255))

    def layer():
        return Image.new("RGBA", (W, H), (0, 0, 0, 0))

    lay = layer(); d = ImageDraw.Draw(lay)
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(70, 74, 86, 64))
    img = Image.alpha_composite(img, lay)

    lay = layer(); d = ImageDraw.Draw(lay)
    STEP = 15
    for a0 in range(-90, 90, STEP):
        for e0 in range(-45, 45, STEP):
            col = zone_color(a0 + STEP / 2, e0 + STEP / 2)
            if col is None:
                continue
            quad, dep = [], 0
            for aa, ee in ((a0, e0), (a0 + STEP, e0), (a0 + STEP, e0 + STEP), (a0, e0 + STEP)):
                x, y, f = pt(aa, ee)
                quad.append((x, y)); dep = f
            if dep >= 0:
                continue
            d.polygon(quad, fill=col + (33,))
    img = Image.alpha_composite(img, lay)

    d = ImageDraw.Draw(img)
    lw = max(1, round(1.5 * k))
    for ring in ([(a, 0) for a in range(-180, 181, 8)],
                 [(0, e) for e in range(-90, 91, 8)]):
        for (a1, e1), (a2, e2) in zip(ring[:-1], ring[1:]):
            x1, y1, f1 = pt(a1, e1)
            x2, y2, f2 = pt(a2, e2)
            col = (200, 204, 216, 128) if (f1 < 0 and f2 < 0) else (120, 124, 138, 38)
            d.line([x1, y1, x2, y2], fill=col, width=lw)
    d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(190, 190, 205, 140), width=lw)

    d.line([cx, cy + 12 * k, cx, cy - 2 * k], fill=(216, 216, 224, 255), width=max(2, round(4 * k)))
    r5 = 5 * k
    d.ellipse([cx - r5, cy - 8 * k - r5, cx + r5, cy - 8 * k + r5], fill=(216, 216, 224, 255))

    try:
        f_dot = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", round(9 * k))
        f_txt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", round(11 * k))
    except OSError:
        f_dot = f_txt = ImageFont.load_default()
    SNAPS = [(0, 0, "F"), (-45, 0, "L"), (45, 0, "R"), (-90, 0, ""), (90, 0, ""), (0, 30, "H"), (0, -15, "Lo")]
    for a, e, lab in SNAPS:
        sxp, syp, f = pt(a, e)
        rr = (9 if lab else 6) * k
        al = 255 if f < 0 else 90
        fill = (255, 255, 255, al) if (a == 0 and e == 0) else (58, 65, 80, al)
        d.ellipse([sxp - rr, syp - rr, sxp + rr, syp + rr], fill=fill,
                  outline=(154, 162, 181, al), width=max(1, round(1.5 * k)))
        if lab:
            tcol = (20, 22, 27, al) if (a == 0 and e == 0) else (217, 220, 227, al)
            bb = d.textbbox((0, 0), lab, font=f_dot)
            d.text((sxp - (bb[2] - bb[0]) / 2, syp - (bb[3] - bb[1]) / 2 - bb[1]), lab, fill=tcol, font=f_dot)

    def dashed(x1, y1, x2, y2, col, w):
        seg = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 + 1e-9
        dash, gap = 4 * k, 4 * k
        t = 0.0
        while t < seg:
            t2 = min(t + dash, seg)
            d.line([x1 + (x2 - x1) * t / seg, y1 + (y2 - y1) * t / seg,
                    x1 + (x2 - x1) * t2 / seg, y1 + (y2 - y1) * t2 / seg], fill=col, width=w)
            t = t2 + gap

    def dim(col, front):
        # mix toward the background for far-side markers (image is opaque, so
        # an alpha fill would be dropped by the final RGB convert)
        if front < 0:
            return col + (255,)
        return tuple(round(c + (b - c) * 0.55) for c, b in zip(col, (27, 27, 31))) + (255,)

    if not kfs:
        for i in range(14):
            x1, y1, _ = pt(azimuth * i / 14.0, elevation * i / 14.0)
            x2, y2, _ = pt(azimuth * (i + 1) / 14.0, elevation * (i + 1) / 14.0)
            d.line([x1, y1, x2, y2], fill=(120, 190, 255, 255), width=max(2, round(2.5 * k)))

        sx, sy, front = pt(azimuth, elevation)
        vx, vy = sx - cx, sy - cy
        L = (vx * vx + vy * vy) ** 0.5 + 1e-9
        dashed(cx, cy, cx + vx / L * 1.32 * L, cy + vy / L * 1.32 * L,
               (120, 190, 255, 90), max(1, round(k)))
        r3 = 3 * k
        d.ellipse([sx - r3, sy - r3, sx + r3, sy + r3], outline=(255, 255, 255, 153),
                  width=max(1, round(k)))

        distF = _dist_scale(distance)
        px = cx + vx * distF
        py = cy + vy * distF
        al = 255 if front < 0 else 115
        for dx, dy in ((-8, -6), (8, -6), (-8, 6), (8, 6)):
            d.line([px, py, cx + dx * k, cy + (dy - 6) * k], fill=(120, 190, 255, al),
                   width=max(1, round(k)))
        d.rounded_rectangle([px - 13 * k, py - 9 * k, px + 13 * k, py + 9 * k], radius=4 * k,
                            fill=(120, 190, 255, al), outline=(255, 255, 255, al),
                            width=max(2, round(2 * k)))
        r35 = 3.5 * k
        d.ellipse([px + 5 * k - r35, py - r35, px + 5 * k + r35, py + r35], fill=(32, 36, 44, al))

        label = f"az {azimuth:+.0f}  el {elevation:+.0f}  dist {distance:.2f}x"
    else:
        if len(kfs) >= 2:
            az_un = _unwrap_seq([kf["az"] for kf in kfs])
            els = [kf["el"] for kf in kfs]
            SUB = 24
            prev = None
            for seg in range(len(kfs) - 1):
                for s in range(SUB + 1):
                    u = s / SUB
                    a = _wrap_deg(_seg_value(az_un, seg, u, smooth))
                    e = _seg_value(els, seg, u, smooth)
                    x1, y1, _ = pt(a, e)
                    if prev is not None:
                        d.line([prev[0], prev[1], x1, y1], fill=(95, 206, 128, 255),
                               width=max(2, round(2.5 * k)))
                    prev = (x1, y1)

        for kf in kfs:
            f_no, kaz, kel, kdist = kf["f"], kf["az"], kf["el"], kf["dist"]
            kx, ky, kfront = pt(kaz, kel)
            distFk = _dist_scale(kdist)
            kpx, kpy = cx + (kx - cx) * distFk, cy + (ky - cy) * distFk
            dashed(kpx, kpy, kx, ky, dim((95, 206, 128), kfront), max(1, round(k)))
            r3 = 3 * k
            d.ellipse([kx - r3, ky - r3, kx + r3, ky + r3],
                      outline=dim((190, 190, 200), kfront), width=max(1, round(k)))
            rr = 10 * k
            d.ellipse([kpx - rr, kpy - rr, kpx + rr, kpy + rr], fill=dim((95, 206, 128), kfront),
                      outline=dim((255, 255, 255), kfront), width=max(2, round(2 * k)))
            lab = str(f_no)
            bb = d.textbbox((0, 0), lab, font=f_dot)
            d.text((kpx - (bb[2] - bb[0]) / 2, kpy - (bb[3] - bb[1]) / 2 - bb[1]), lab,
                   fill=(14, 20, 16, 255), font=f_dot)

        label = (f"{len(kfs)} keyframes  f{kfs[0]['f']}-{kfs[-1]['f']}  "
                 f"{'smooth' if smooth else 'linear'}")

    d.text((8 * k, 6 * k), label, fill=(255, 255, 100, 255), font=f_txt)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)
