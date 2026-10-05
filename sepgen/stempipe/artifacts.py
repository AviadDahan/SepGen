"""Writing a render to disk: wavs, the joint latent, and per-track muxed videos."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch
from ltx_pipelines.utils.media_io import encode_audio

from .stem_two_stages_hq import StemResult


def span_names(num_stems: int) -> list[str]:
    return [f"stem{k}" for k in range(num_stems)] + ["mix_generated"]


def save_stem_result(result: StemResult, out_dir: Path, *, fps: float,
                     video_path: Path | None = None, prompts: dict | None = None) -> list[Path]:
    """Write one cell's audio artifacts; returns the wav paths in layout order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = span_names(len(result.spans) - 1)
    wavs = []
    for name, audio in zip(names, result.spans):
        path = out_dir / f"{name}.wav"
        encode_audio(audio, str(path))
        wavs.append(path)

    torch.save({"joint_latent": result.joint_latent, "span_len": result.span_len,
                "mix_index": result.mix_index, "layout": names},
               out_dir / "joint_audio_latent.pt")
    if prompts is not None:
        (out_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))
    if result.attention is not None:
        save_attention(result.attention, out_dir)
    if video_path is not None:
        # The SOURCES get a muxed video each; the mixture does NOT. `video.mp4` already is
        # this video with the mixture on it, so a `mix_generated_muxed.mp4` was a second copy
        # of the same thing -- and a lossy one, since `-shortest` trims it to the audio and
        # drops the final frame. One name for one artifact (2026-08-26). Runs made before this
        # still carry the old file; boards read `video.mp4`, which every run has.
        for index, (name, wav) in enumerate(zip(names, wavs)):
            if index == result.mix_index:
                continue
            mux(video_path, wav, out_dir / f"{name}_muxed.mp4")
        if len(result.spans) - 1 == 2:      # two sources -> one per ear
            mux_split_ears(video_path, wavs[0], wavs[1],
                           out_dir / "sources_split_ears.mp4")
    return wavs


def save_attention(capture, out_dir: Path) -> Path:
    """Write `attention_maps.npz` -- everything a localization readout needs, and nothing
    it has to be told separately.

    Geometry travels WITH the maps: a grid passed to the readout by some other route is a
    grid that can disagree with the array it describes, which is the failure the reference
    lineage hit (a module-level GRID constant left a readout looping over 15 latent frames
    while the maps carried 16, emitting a track one frame short of its own evidence).
    """
    import numpy as np

    # UNCOMPRESSED, deliberately. Measured on a 541 MB dump: savez_compressed takes 29.6 s to
    # save 123 MB (23%), savez takes 1.9 s. The compression happens in the render process, so
    # those 28 seconds are GPU idle time -- ~1.6 GPU-hours across a 198-clip roster, to save
    # 24 GB out of 12 TB free. It also costs 3 s of decompression on every analysis read, and
    # the whole point of the dump is to be read many times. np.load reads either kind, so
    # dumps already written stay usable.
    path = out_dir / "attention_maps.npz"
    np.savez(path, **capture.npz_payload())
    return path


def ensure_faststart(video: Path) -> None:
    """Move the mp4's `moov` atom to the front, in place, so it streams.

    The pipeline's own encoder leaves `moov` after `mdat`, which means a browser with
    `preload="none"` must fetch the whole file before it plays. The review server can remux a
    faststart copy into a cache, but doing it once here costs a stream copy and spares every
    consumer -- and every board rebuild -- from discovering it separately.
    """
    tmp = video.with_suffix(".faststart.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-c", "copy",
                    "-movflags", "+faststart", str(tmp)], check=True)
    tmp.replace(video)


def mux(video: Path, wav: Path, out: Path) -> None:
    """Video + one audio track, faststart so a review page can stream it.

    The `-map` flags are load-bearing, not tidiness: our video file ALREADY carries the
    mixture, and without explicit mapping ffmpeg's default stream selection picks one audio
    stream across all inputs -- it took the video's embedded mixture and every "stem" mp4
    silently played the mix (caught by ear 2026-08-26, after the wavs themselves proved
    perfectly separated). Say exactly which streams to take.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav),
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-shortest",
         "-movflags", "+faststart", str(out)],
        check=True)


def mux_split_ears(video: Path, left: Path, right: Path, out: Path) -> None:
    """One video, source 0 in the LEFT ear and source 1 in the RIGHT.

    The most direct way to hear whether separation actually happened: if the two sources
    really carry different content, each ear gets its own and the scene splits apart; if the
    method merely copied the mixture twice, both ears carry the same thing and it collapses
    to mono. No metric makes that as obvious as listening to it.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(left),
         "-i", str(right),
         "-filter_complex",
         # Downmix each source to MONO first, then place one per ear with an EXPLICIT map.
         # Without this, `join` simply takes the first two channels it can find -- and since
         # our sources are stereo, that was source 0's own left and right in both ears, with
         # source 1 dropped entirely (caught by ear, then confirmed: L and R correlated 0.998
         # with each other and 0.999 with source 0).
         "[1:a]aformat=channel_layouts=mono[l];[2:a]aformat=channel_layouts=mono[r];"
         "[l][r]join=inputs=2:channel_layout=stereo:map=0.0-FL|1.0-FR[a]",
         "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-shortest", "-movflags", "+faststart", str(out)],
        check=True)


def dbfs(path: Path) -> float:
    """RMS level of a wav, for the dead-stem check. Absolute, never a ratio."""
    import soundfile as sf

    x, _ = sf.read(str(path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    import numpy as np

    return float(20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-9)))
