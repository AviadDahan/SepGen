#!/usr/bin/env python
"""Mux the demo videos (demo stage 6), with the project page's encode settings.

  original.mp4       the original clip with its own audio-mix
  moving_mix.mp4     the moving-camera render with that same, unchanged audio-mix
  moving_ours.mp4    the same render with the re-spatialized stereo mix (mix_spatial.wav)
  moving_stem0.mp4,  the render with one re-spatialized stem each, played at the gain it has
  moving_stem1.mp4   inside mix_spatial.wav, so a source sounds where it sits in that mix

All are scaled to --width (768 on the page; 0 keeps the native size), x264 at --crf, AAC stereo,
faststart.

  python demo/mux.py --video clip.mp4 --mix-wav mix.wav --render out/render/render.mp4 \
      --spatial-dir out/audio_widened --out-dir out/final
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def encode(video: Path, audio: Path | None, dst: Path, width: int, crf: int, bitrate: str,
           volume_db: float | None = None) -> None:
    """Video from `video`, audio from `audio` (None keeps the video's own track)."""
    inputs = ["-i", str(video)] + (["-i", str(audio)] if audio else [])
    maps = ["-map", "0:v:0", "-map", "1:a:0" if audio else "0:a:0"]
    vf = ["-vf", f"scale={width}:-2"] if width else []
    af = ["-af", f"volume={volume_db:.3f}dB"] if volume_db is not None else []
    subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, *maps, "-shortest", *vf, *af,
                    "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", bitrate, "-ac", "2",
                    "-movflags", "+faststart", str(dst)], check=True)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", type=Path, required=True, help="the original clip")
    ap.add_argument("--mix-wav", type=Path, required=True, help="the original audio-mix")
    ap.add_argument("--render", type=Path, required=True, help="render_crossview.py render.mp4")
    ap.add_argument("--spatial-dir", type=Path, required=True,
                    help="mix_spatial.wav + stem_{0,1}_spatial.wav (spatialize_stems.py or "
                         "exaggerate_spatial.py output)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--width", type=int, default=768, help="output width; 0 = native")
    ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--audio-bitrate", default="128k")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    enc = dict(width=args.width, crf=args.crf, bitrate=args.audio_bitrate)

    encode(args.video, args.mix_wav, out / "original.mp4", **enc)
    encode(args.render, args.mix_wav, out / "moving_mix.mp4", **enc)
    encode(args.render, args.spatial_dir / "mix_spatial.wav", out / "moving_ours.mp4", **enc)
    # mix_spatial = g * (stem_0 + stem_1); play each stem at that same g
    stems = [sf.read(args.spatial_dir / f"stem_{k}_spatial.wav", dtype="float64")[0]
             for k in (0, 1)]
    mix, _ = sf.read(args.spatial_dir / "mix_spatial.wav", dtype="float64")
    g_db = 20 * np.log10(rms(mix) / rms(stems[0] + stems[1]))
    for k in (0, 1):
        encode(args.render, args.spatial_dir / f"stem_{k}_spatial.wav",
               out / f"moving_stem{k}.mp4", volume_db=g_db, **enc)
    print(f"[mux] stems at {g_db:+.2f} dB -> {out}", flush=True)


if __name__ == "__main__":
    main()
