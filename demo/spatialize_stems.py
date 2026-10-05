#!/usr/bin/env python
"""Re-spatialize each separated stem for the moving camera (demo stage 5).

Per stem, the lifted 3-D track (in the original camera's frame) is re-expressed relative to the
AUTHORED moving camera, frame by frame (X_rel = inv(C_tgt) @ X_src, poses from the warp guide's
trajectory.json), and then:

1. VOLUME follows distance: gain g(t) = (r(t0) / r(t))^a, the free-field 1/r amplitude law for
   a = 1 (so t0 is unity gain and levels change only as the camera path changes the distance),
   clamped to +-12 dB, linearly interpolated from the per-frame values to every sample.
2. STEREO follows direction: the depth-scaled mono stem goes through a free-field two-cardioid
   renderer (two "ears" at +-9 cm: per-ear cardioid gain + inter-aural delay, no room, no 1/r
   of its own), driven by the per-frame azimuth/elevation resampled to 49 blocks and rendered
   as a crossfaded overlap-add (50 ms ramps).

Normalization: the depth-scaled remix (stem0 + stem1) is RMS-matched to the ORIGINAL audio-mix
and the same scalar is applied to both stems, so distance redistributes level over time and
between the sources without changing the overall level; the stereo mix is RMS-matched to the
original mix as well (RMS-match, never peak-normalize).

Outputs (--out-dir): stem_{0,1}_depth.wav (mono, 1/r), stem_{0,1}_spatial.wav (stereo),
mix_depth.wav, mix_spatial.wav, gains.json. Absolute dBFS per stem are printed; a stem below
-45 dBFS is flagged.

CPU, seconds.
  python demo/spatialize_stems.py --tracks out/tracks_3d.json --localize-dir out/localize \
      --mix-wav mix.wav --trajectory-json out/guide/trajectory.json --out-dir out/audio
"""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

DEAD_DBFS = -45.0
GAIN_CLAMP = (0.25, 4.0)   # +-12 dB safety clamp on the 1/r trajectory gain
CONTRACT_BLOCKS = 49       # block rate of the cardioid renderer; 121 blocks would be
                           # shorter than the crossfade ramp

# ------------------------------------------------------------------ free-field cardioid renderer
C = 343.0
EAR_Y = 0.09                       # +-9 cm ear offset
_TAPS = 64                         # short per-ear FIR (fractional delay lives here)
FADE_S_DEFAULT = 0.05              # 50 ms crossfade between blocks


def _xfade_render(mono, fs, traj, rirs_per_block, n_ch, fade_s):
    """Crossfaded overlap-add: block i's windowed signal is convolved with
    ITS OWN per-channel RIRs; adjacent windows sum to exactly 1."""
    sig_len = len(mono)
    n = len(traj)
    idx = np.linspace(0, sig_len, n + 1).astype(int)
    F = int(fade_s * fs)
    out = np.zeros((sig_len, n_ch))
    for i in range(n):
        a, b = idx[i], idx[i + 1]
        lo, hi = max(0, a - F), min(sig_len, b + F)
        w = np.ones(hi - lo)
        if F and a > 0:
            ramp = np.linspace(0.0, 1.0, 2 * F, endpoint=False)
            s = (a - F) - lo
            ln = min(2 * F - s, len(w))
            w[:ln] = ramp[s:s + ln]
        if F and b < sig_len:
            ramp = np.linspace(1.0, 0.0, 2 * F, endpoint=False)
            e = hi - (b - F)
            ln = min(e, len(w))
            w[len(w) - ln:] = np.minimum(
                w[len(w) - ln:], ramp[:ln][-ln:] if ln <= 2 * F else ramp)
        block = mono[lo:hi] * w
        if len(block) == 0:
            continue
        for ch, rir in enumerate(rirs_per_block[i]):
            cv = np.convolve(block, rir)[:sig_len - lo]
            out[lo:lo + len(cv), ch] += cv
    return out


def _ear_ir(gain: float, delay_s: float, fs: int) -> np.ndarray:
    """A single fractional-delayed tap of amplitude `gain`, in a length-_TAPS FIR.
    Delay is centred at _TAPS//2 so both + and - inter-aural delays fit."""
    ir = np.zeros(_TAPS)
    center = _TAPS // 2
    d = center + delay_s * fs
    i = int(np.floor(d))
    frac = d - i
    if 0 <= i < _TAPS - 1:
        ir[i] = gain * (1 - frac)
        ir[i + 1] = gain * frac
    elif 0 <= i < _TAPS:
        ir[i] = gain
    return ir


def render_cardioid_direct(mono, fs, az_left, el, r, *, fade_s=FADE_S_DEFAULT):
    """Free-field cardioid binaural render of a moving source. (len(mono), 2).

    az_left is +LEFT, el is +up. r is accepted for a common signature but only the
    DIRECTION is used (level is handled by the 1/r gain upstream).

      per ear e in {L=+y, R=-y}:
        gain_e  = 0.5 * (1 + cos<axis_e, dir_to_source>)     # cardioid pattern
        delay_e = -dot(ear_offset_e, dir) / c                 # relative ITD only
    """
    az_left = np.asarray(az_left, float)
    el = np.asarray(el, float)
    r = np.asarray(r, float)
    n = az_left.size
    if not (el.size == r.size == n):
        raise ValueError(f"az/el/r length mismatch: {n}/{el.size}/{r.size}")
    # unit source direction in (fwd, left, up)
    u = np.stack([np.cos(el) * np.cos(az_left),
                  np.cos(el) * np.sin(az_left),
                  np.sin(el)], axis=1)
    traj = (r[:, None] * u)                       # only for _xfade_render's block grid
    rirs = []
    for i in range(n):
        uy = float(u[i, 1])                       # +left component
        # cardioid gains: L axis +y, R axis -y
        gL = 0.5 * (1.0 + uy)
        gR = 0.5 * (1.0 - uy)
        # inter-aural delay: ear closer to source is earlier (negative delay)
        dL = -(EAR_Y * uy) / C
        dR = -(-EAR_Y * uy) / C
        rirs.append([_ear_ir(gL, dL, fs), _ear_ir(gR, dR, fs)])
    return _xfade_render(np.asarray(mono, float), fs, traj, rirs, 2, fade_s)


# ------------------------------------------------------------------ io
def to_blocks(arr: np.ndarray, n: int = CONTRACT_BLOCKS) -> np.ndarray:
    src = np.linspace(0.0, 1.0, len(arr))
    return np.interp(np.linspace(0.0, 1.0, n), src, np.asarray(arr, float))


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        raw = np.frombuffer(w.readframes(n), dtype=np.int16)
    x = raw.astype(np.float64).reshape(-1, ch).mean(1) / 32768.0
    return x, sr


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    y = np.clip(x, -1.0, 1.0)
    ch = 1 if y.ndim == 1 else y.shape[1]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((y * 32767.0).astype(np.int16).tobytes())


def dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x ** 2)))
    return 20.0 * np.log10(max(rms, 1e-9))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracks", type=Path, required=True,
                    help="lift_tracks.py output")
    ap.add_argument("--localize-dir", type=Path, required=True,
                    help="localize.py output (stem_{0,1}.wav)")
    ap.add_argument("--mix-wav", type=Path, required=True,
                    help="the original audio-mix (level reference)")
    ap.add_argument("--trajectory-json", type=Path, required=True,
                    help="build_warp_guide.py output: the authored per-frame camera poses")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--frame-rate", type=float, default=24.0)
    ap.add_argument("--gain-exponent", type=float, default=1.0,
                    help="exponent a in gain = (r0/r)^a; 1.0 = free-field "
                         "amplitude (default), 0.5 = gentler")
    args = ap.parse_args()

    tracks = json.loads(args.tracks.read_text())
    odir = args.out_dir
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "args.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    # Per-frame analytic transform of the lifted track into the moving camera's frame.
    traj = json.loads(args.trajectory_json.read_text())
    rel = {}
    for si in (0, 1):
        X = np.asarray(tracks["stems"][f"stem{si}"]["src_fixed_xyz"])
        out = np.empty((len(traj["poses"]), 3))
        for f, p in enumerate(traj["poses"]):
            Cm = np.asarray(p["C_tgt"])
            out[f] = Cm[:3, :3].T @ (X[min(f, len(X) - 1)] - Cm[:3, 3])
        rr = np.linalg.norm(out, axis=1)
        rel[si] = {
            "az_deg": np.degrees(np.arctan2(out[:, 0], out[:, 2])),
            "el_deg": np.degrees(np.arcsin(
                np.clip(-out[:, 1] / np.maximum(rr, 1e-9), -1, 1))),
            "range_m": rr}

    mix, sr = read_wav(args.mix_wav)
    stems, gains_rec = [], {}
    for si in (0, 1):
        x, sr_s = read_wav(args.localize_dir / f"stem_{si}.wav")
        assert sr_s == sr, f"stem sr {sr_s} != mix sr {sr}"
        rng = np.asarray(rel[si]["range_m"])
        g_frames = np.clip((rng[0] / np.maximum(rng, 1e-6))
                           ** args.gain_exponent, *GAIN_CLAMP)
        t_frames = np.arange(len(g_frames)) / args.frame_rate
        t_samples = np.arange(len(x)) / sr
        env = np.interp(t_samples, t_frames, g_frames)
        before = dbfs(x)
        y = x * env
        stems.append(y)
        gains_rec[f"stem{si}"] = {
            "range_m": rng.tolist(),
            "gain_frames": g_frames.tolist(),
            "gain_db_minmax": [float(20 * np.log10(g_frames.min())),
                               float(20 * np.log10(g_frames.max()))],
            "dbfs_before": before}

    remix = stems[0] + stems[1]
    norm = 10 ** ((dbfs(mix) - dbfs(remix)) / 20.0)
    remix *= norm
    spatial = []
    for si in (0, 1):
        stems[si] *= norm
        after = dbfs(stems[si])
        gains_rec[f"stem{si}"]["dbfs_after"] = after
        gains_rec[f"stem{si}"]["rms_norm_scalar"] = norm
        dead = " DEAD-STEM?" if after < DEAD_DBFS else ""
        print(f"[spatialize] stem{si}: "
              f"{gains_rec[f'stem{si}']['dbfs_before']:+.1f} -> "
              f"{after:+.1f} dBFS, trajectory gain "
              f"{gains_rec[f'stem{si}']['gain_db_minmax'][0]:+.1f}.."
              f"{gains_rec[f'stem{si}']['gain_db_minmax'][1]:+.1f} dB"
              f"{dead}")
        write_wav(odir / f"stem_{si}_depth.wav", stems[si], sr)
        # STEREO: pan + ITD by THIS stem's per-frame direction (the renderer's az is +LEFT,
        # the track's is +RIGHT -> negate)
        cm = rel[si]
        az_left = to_blocks(-np.radians(np.asarray(cm["az_deg"])))
        el = to_blocks(np.radians(np.asarray(cm["el_deg"])))
        rr = to_blocks(np.asarray(cm["range_m"]))
        st = render_cardioid_direct(stems[si], sr, az_left, el, rr)
        spatial.append(st)
        write_wav(odir / f"stem_{si}_spatial.wav", st, sr)
    mix_spatial = spatial[0] + spatial[1]
    norm2 = 10 ** ((dbfs(mix) - dbfs(mix_spatial.mean(1))) / 20.0)
    mix_spatial *= norm2
    write_wav(odir / "mix_spatial.wav", mix_spatial, sr)
    write_wav(odir / "mix_depth.wav", remix, sr)
    gains_rec["mix"] = {"dbfs_original": dbfs(mix),
                        "dbfs_depth_remix": dbfs(remix),
                        "dbfs_spatial_mix": dbfs(mix_spatial.mean(1)),
                        "spatial_norm_scalar": norm2}
    (odir / "gains.json").write_text(json.dumps(gains_rec))
    print(f"[spatialize] spatial mix {gains_rec['mix']['dbfs_spatial_mix']:+.1f} dBFS "
          f"(stereo, cardioid by the re-expressed az/el) -> {odir}")


if __name__ == "__main__":
    main()
