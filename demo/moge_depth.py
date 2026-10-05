#!/usr/bin/env python
"""MoGe-2 metric depth for one clip (demo stage 2).

Per-frame metric depth (metres) + validity mask + normalized intrinsics, saved as one
uncompressed npz, plus a colormapped inverse-depth preview mp4 muxed with the clip's own audio.
The horizontal field of view is fixed (--fov-x, 50 deg) so depth and intrinsics are temporally
consistent and match the lens the warp guide assumes (build_warp_guide.py --hfov).

Consumed by build_warp_guide.py (warp conditioning) and lift_tracks.py (3-D lift).

Env: MoGe (setup_demo.sh), one GPU, about 1-2 min for a 121-frame clip on an A6000.
  python demo/moge_depth.py --video clip.mp4 --out-dir out/depth
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from moge.model.v2 import MoGeModel

DEFAULT_MODEL = "Ruicheng/moge-2-vitl-normal"


def read_frames(video: Path, num_frames: int | None) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened(), f"cannot open {video}"
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if num_frames is not None and len(frames) >= num_frames:
            break
    cap.release()
    assert frames, f"no frames read from {video}"
    if num_frames is not None:
        assert len(frames) >= num_frames, \
            f"{video} has {len(frames)} frames < {num_frames}"
    return np.stack(frames, 0)


def depth_preview(depth: np.ndarray, mask: np.ndarray, out_mp4: Path,
                  src_video: Path, fps: float) -> None:
    """Colormapped inverse-depth preview (near = warm), source audio muxed."""
    F, H, W = depth.shape
    inv = np.where(mask, 1.0 / np.clip(depth, 1e-3, None), np.nan)
    lo, hi = np.nanpercentile(inv, 2), np.nanpercentile(inv, 98)
    tmp = out_mp4.with_suffix(".silent.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(F):
        n = np.clip((inv[i] - lo) / (hi - lo + 1e-9), 0, 1)
        n = np.nan_to_num(n, nan=0.0)
        frame = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        frame[~mask[i]] = (128, 0, 128)   # invalid/sky: magenta, like the warp
        vw.write(frame)
    vw.release()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-i", str(src_video),
         "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "18",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         "-movflags", "+faststart", str(out_mp4)], check=True)
    tmp.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="writes depth.npz + depth_preview.mp4")
    ap.add_argument("--model-id", default=DEFAULT_MODEL,
                    help=f"MoGe-2 checkpoint id (default: {DEFAULT_MODEL})")
    ap.add_argument("--num-frames", type=int, default=121,
                    help="frames to read; fail-fast if fewer (default: 121)")
    ap.add_argument("--frame-rate", type=float, default=24.0,
                    help="preview fps (default: 24)")
    ap.add_argument("--device", default="cuda",
                    help="inference device (default: cuda)")
    ap.add_argument("--num-tokens", type=int, default=2048,
                    help="MoGe base ViT tokens; suggested 1200-2500, more = "
                         "finer + slower (default: 2048)")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="frames per inference batch (default: 4)")
    ap.add_argument("--fov-x", type=float, default=50.0,
                    help="force this horizontal FoV (deg) so depth/intrinsics "
                         "are temporally consistent AND match the warp's "
                         "assumed lens (build_warp_guide.py --hfov). 0 = let "
                         "MoGe infer FoV per frame (default: 50)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    model = MoGeModel.from_pretrained(args.model_id).to(device).eval()
    (args.out_dir / "args.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()},
         "torch": torch.__version__, "device": device}, indent=2))

    video = args.video
    out_npz = args.out_dir / "depth.npz"
    frames = read_frames(video, args.num_frames)
    F, H, W = frames.shape[:3]
    depth = np.empty((F, H, W), np.float16)
    mask = np.empty((F, H, W), bool)
    intr = np.empty((F, 3, 3), np.float32)
    fov_x = args.fov_x if args.fov_x > 0 else None
    t_conv = t_infer = t_copy = 0.0
    for i in tqdm(range(0, F, args.batch_size), desc=f"depth {video.name}"):
        j = min(i + args.batch_size, F)
        t0 = time.time()
        imgs = (torch.from_numpy(np.ascontiguousarray(frames[i:j]))
                .to(device).permute(0, 3, 1, 2).float().div_(255.0))
        torch.cuda.synchronize() if device != "cpu" else None
        t1 = time.time()
        with torch.no_grad():
            out = model.infer(imgs, num_tokens=args.num_tokens, fov_x=fov_x)
        torch.cuda.synchronize() if device != "cpu" else None
        t2 = time.time()
        # fp16 conversion on the GPU (numpy's fp32->fp16 astype is far slower)
        depth[i:j] = out["depth"].half().cpu().numpy()
        mask[i:j] = out["mask"].bool().cpu().numpy()
        intr[i:j] = out["intrinsics"].float().cpu().numpy()
        t3 = time.time()
        t_conv, t_infer, t_copy = (t_conv + t1 - t0, t_infer + t2 - t1,
                                   t_copy + t3 - t2)
    print(f"[depth] timing: to-gpu {t_conv:.1f}s, infer {t_infer:.1f}s, "
          f"copy-back {t_copy:.1f}s", flush=True)
    np.savez(out_npz,  # uncompressed: decompressing large members is slow
             depth=depth, mask=mask, intrinsics=intr,
             video=str(video), width=W, height=H,
             num_tokens=args.num_tokens, fov_x=args.fov_x)
    depth_preview(depth.astype(np.float32), mask,
                  args.out_dir / "depth_preview.mp4", video, args.frame_rate)
    valid = mask.mean()
    med = float(np.nanmedian(np.where(mask, depth.astype(np.float32), np.nan)))
    print(f"[depth] {F}x{H}x{W}, valid {valid:.1%}, "
          f"median depth {med:.2f} m -> {out_npz}")


if __name__ == "__main__":
    main()
