#!/usr/bin/env python
"""Author the camera move and build the CrossView-Warp depth-warp guide for one clip (demo stage 3).

Samples a keyframed camera trajectory (within the IC-LoRA's reliable range: |az| <= 45 deg,
el in [-20, 30] deg), warps every source frame through the MoGe-2 metric depth (magenta
disocclusion holes; crossview_warp.py), and writes into --out-dir:

  warp.mp4           the IC-LoRA warp reference guide
  trajectory.json    resolved keyframes + PER-FRAME poses (incl. the 4x4 camera-to-source
                     extrinsic C_tgt the audio stage uses) + pivot + focal length
  orbit_view.png     camera-move diagram
  guide_preview.mp4  original | warp side by side, with the clip's audio

The orbit pivot defaults to the 'sides' estimator: the median 3-D point of the left and right
thirds of frame 0 (scenes with one subject on each side and an empty centre); 'auto' is the
upstream central-region median. Check guide_preview.mp4 before rendering: if a subject leaves the
warp at the trajectory extremes, pass --pivot or a smaller --traj-scale.

Env: MoGe (setup_demo.sh), CPU.
  python demo/build_warp_guide.py --video clip.mp4 --depth-npz out/depth/depth.npz \
      --trajectory orbit_rise --out-dir out/guide
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossview_warp import (auto_pivot, build_warp, fx_from_hfov,  # noqa: E402
                            orbit_view_image, parse_keyframes, sides_pivot)

AZ_CAP, EL_LO, EL_HI = 45.0, -20.0, 30.0   # the LoRA's reliable camera range

# Trajectory templates (121f @ 24 fps; az/el scaled by --traj-scale, capped).
# Easing per upstream guidance: 'smooth' = Catmull-Rom flowing move; the arc
# uses ease_in_out on a single leg (settle into the end pose).
TEMPLATES = {
    "arc": {"easing": "ease_in_out", "keyframes": [
        {"f": 1, "az": 0.0, "el": 0.0, "dist": 1.0},
        {"f": 121, "az": 40.0, "el": 8.0, "dist": 1.0}]},
    "swing": {"easing": "smooth", "keyframes": [
        {"f": 1, "az": -30.0, "el": 5.0, "dist": 1.0},
        {"f": 61, "az": 0.0, "el": 5.0, "dist": 1.0},
        {"f": 121, "az": 30.0, "el": 5.0, "dist": 1.0}]},
    "orbit_rise": {"easing": "smooth", "keyframes": [
        {"f": 1, "az": 0.0, "el": 0.0, "dist": 1.0},
        {"f": 61, "az": -25.0, "el": 12.0, "dist": 1.05},
        {"f": 121, "az": -45.0, "el": 20.0, "dist": 1.1}]},
}


def write_video(frames: np.ndarray, out: Path, fps: float, crf: int,
                audio_src: Path | None = None) -> None:
    """Encode RGB uint8 [F,H,W,3] via an ffmpeg rawvideo pipe; optionally mux
    the audio track of `audio_src` (no -shortest: never drop frames)."""
    F, H, W = frames.shape[:3]
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", f"{fps}", "-i", "-"]
    if audio_src is not None:
        cmd += ["-i", str(audio_src), "-map", "0:v", "-map", "1:a?",
                "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.stdin.write(frames.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0, f"ffmpeg failed for {out}"


def read_frames(video: Path, num_frames: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened(), f"cannot open {video}"
    frames = []
    while len(frames) < num_frames:
        ok, bgr = cap.read()
        assert ok, f"{video}: only {len(frames)} frames < {num_frames}"
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, 0)


def scaled_keyframes(template: dict, scale: float) -> list[dict]:
    kfs = []
    for kf in template["keyframes"]:
        az, el = kf["az"] * scale, kf["el"] * scale
        assert abs(az) <= AZ_CAP + 1e-6, \
            f"az {az:.1f} exceeds the reliable cap {AZ_CAP}"
        assert EL_LO - 1e-6 <= el <= EL_HI + 1e-6, \
            f"el {el:.1f} outside reliable range [{EL_LO}, {EL_HI}]"
        kfs.append({**kf, "az": az, "el": el})
    return kfs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--depth-npz", type=Path, required=True,
                    help="moge_depth.py output for the same clip")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--trajectory", choices=tuple(TEMPLATES), default="orbit_rise",
                    help="camera-move template (default: orbit_rise)")
    ap.add_argument("--traj-scale", type=float, default=1.0,
                    help="multiplies template az/el; capped at the reliable "
                         "range (default: 1.0)")
    ap.add_argument("--keyframes-json", type=Path, default=None,
                    help="custom move instead of a template: {keyframes: [{f, az, el, "
                         "dist}, ...], easing?}")
    ap.add_argument("--pivot", type=float, nargs=3, default=None,
                    metavar=("PX", "PY", "PZ"),
                    help="orbit pivot in METRES (source camera frame), overriding "
                         "--pivot-mode")
    ap.add_argument("--pivot-mode", choices=("sides", "auto"), default="sides",
                    help="pivot estimator when no --pivot: 'sides' = median "
                         "3-D point of the left+right thirds (scenes with the subjects "
                         "at the sides and an empty center, where the upstream central "
                         "median lands far in the background); 'auto' = upstream's "
                         "central-region median (default: sides)")
    ap.add_argument("--hfov", type=float, default=50.0,
                    help="assumed horizontal FoV in degrees; 0 = read focal "
                         "from MoGe intrinsics (default: 50)")
    ap.add_argument("--num-frames", type=int, default=121)
    ap.add_argument("--frame-rate", type=float, default=24.0)
    ap.add_argument("--crf", type=int, default=10,
                    help="x264 crf for warp.mp4; keep <=10 so magenta hole "
                         "edges survive encoding (default: 10)")
    args = ap.parse_args()

    odir = args.out_dir
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "args.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    npz = np.load(args.depth_npz)
    depth = npz["depth"].astype(np.float64)[:args.num_frames]
    mask = npz["mask"][:args.num_frames]
    z = np.where(mask, depth, np.nan)
    frames = read_frames(args.video, args.num_frames)
    assert frames.shape[:3] == z.shape, f"frames {frames.shape} vs depth {z.shape}"
    H, W = z.shape[1:]

    if args.hfov > 0:
        fx = fx_from_hfov(W, args.hfov)
    else:
        fx = float(npz["intrinsics"][0][0, 0]) * W

    scale = args.traj_scale
    if args.keyframes_json is not None:
        ov = json.loads(args.keyframes_json.read_text())
        raw_kfs, easing = ov["keyframes"], ov.get("easing", "smooth")
        template_name = "custom"
    else:
        template_name = args.trajectory
        tpl = TEMPLATES[template_name]
        raw_kfs, easing = scaled_keyframes(tpl, scale), tpl["easing"]

    if args.pivot is not None:
        pivot = np.array(args.pivot)
        pivot_source = "--pivot"
    elif args.pivot_mode == "sides":
        pivot = sides_pivot(z[0], fx, W / 2.0, H / 2.0)
        pivot_source = "sides (left+right thirds median, frame 0)"
    else:
        pivot = auto_pivot(z[0], fx, W / 2.0, H / 2.0)
        pivot_source = "auto (upstream central-region median, frame 0)"

    kfs = parse_keyframes(raw_kfs, args.num_frames,
                          default_pivot=tuple(float(c) for c in pivot))
    bar = tqdm(total=len(frames), desc=f"warp [{template_name}]")
    warp, poses = build_warp(frames, z, kfs, fx=fx, easing=easing,
                             progress=lambda i: bar.update(1))
    bar.close()

    write_video(warp, odir / "warp.mp4", args.frame_rate, args.crf)
    smooth_path = easing == "smooth"
    cv2.imwrite(str(odir / "orbit_view.png"),
                cv2.cvtColor(orbit_view_image(
                    kfs[0]["az"], kfs[0]["el"], kfs[0]["dist"],
                    kfs=kfs if len(kfs) >= 2 else None,
                    smooth=smooth_path), cv2.COLOR_RGB2BGR))
    half = np.concatenate([frames[:, ::2, ::2], warp[:, ::2, ::2]], axis=2)
    write_video(half, odir / "guide_preview.mp4", args.frame_rate, 18,
                audio_src=args.video)
    hole_frac = float((warp == [255, 0, 255]).all(-1).mean())
    (odir / "trajectory.json").write_text(json.dumps({
        "template": template_name, "scale": scale, "easing": easing,
        "keyframes": kfs, "pivot": pivot.tolist(),
        "pivot_source": pivot_source, "fx_pixels": fx,
        "hfov_deg": args.hfov, "num_frames": args.num_frames,
        "frame_rate": args.frame_rate, "hole_frac": hole_frac,
        "poses": poses}, indent=1))
    print(f"[guide] {template_name} x{scale:g} pivot "
          f"[{pivot[0]:+.2f} {pivot[1]:+.2f} {pivot[2]:.2f}] m "
          f"({pivot_source}), holes {hole_frac:.1%}")


if __name__ == "__main__":
    main()
