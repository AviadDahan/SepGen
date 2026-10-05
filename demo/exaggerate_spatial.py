"""Optional: widen the left-right placement of each spatial stem (the project-page demo uses x2.5).

Each stem's interaural level difference (ILD) is measured in short frames (50 ms, 10 ms hop),
smoothed over time (200 ms), and scaled by --factor with a per-sample left/right gain pair that
keeps the stem's total energy (L^2 + R^2) unchanged. The two widened stems are summed and scaled
to the RMS of the input mix_spatial.wav, so the overall level does not change. This exaggerates
the effect for illustration; it is not part of the method.

  python demo/exaggerate_spatial.py --src out/audio --out out/audio_widened --factor 2.5
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def frame_ild_db(x, hop, win):
    """Per-sample ILD (dB, left minus right) from framed energies, linearly interpolated."""
    n = len(x)
    centers, ild = [], []
    for s in range(0, max(1, n - win + 1), hop):
        seg = x[s:s + win]
        el, er = (seg[:, 0] ** 2).mean() + 1e-12, (seg[:, 1] ** 2).mean() + 1e-12
        centers.append(s + win / 2)
        ild.append(10 * np.log10(el / er))
    return np.interp(np.arange(n), centers, ild)


def smooth(v, width):
    k = np.hanning(width)
    k /= k.sum()
    pad = np.pad(v, (width // 2, width - width // 2 - 1), mode="edge")
    return np.convolve(pad, k, mode="valid")


def widen(x, factor, sr):
    """Scale the ILD of a stereo stem by `factor`, preserving per-sample L^2+R^2 energy weights."""
    ild = smooth(frame_ild_db(x, hop=sr // 100, win=sr // 20), width=sr // 5)   # 50 ms frames, 200 ms smoothing
    target = factor * ild
    # gains that move the ILD from `ild` to `target` while keeping gl^2*El + gr^2*Er = El + Er
    d = 10 ** ((target - ild) / 20)          # desired ratio gl/gr
    r = 10 ** (ild / 10)                     # El / Er
    gr = np.sqrt((r + 1) / (d ** 2 * r + 1))
    gl = d * gr
    return np.stack([x[:, 0] * gl, x[:, 1] * gr], axis=1), ild


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="spatialize_stems.py output (stem_{0,1}_spatial.wav, mix_spatial.wav)")
    ap.add_argument("--out", type=Path, required=True, help="new output directory")
    ap.add_argument("--factor", type=float, default=2.5)
    a = ap.parse_args()
    assert not a.out.exists(), f"{a.out} exists; every run writes a new directory"
    a.out.mkdir(parents=True)
    stems = []
    rec = {}
    for k in (0, 1):
        x, sr = sf.read(a.src / f"stem_{k}_spatial.wav")
        w, ild = widen(x, a.factor, sr)
        stems.append(w)
        ild_after = smooth(frame_ild_db(w, sr // 100, sr // 20), sr // 5)
        rec[f"stem{k}_ild_db_before"] = [round(float(np.percentile(ild, p)), 1) for p in (5, 50, 95)]
        rec[f"stem{k}_ild_db_after"] = [round(float(np.percentile(ild_after, p)), 1) for p in (5, 50, 95)]
    m_ref, sr = sf.read(a.src / "mix_spatial.wav")
    mix = stems[0] + stems[1]
    mix *= np.sqrt((m_ref ** 2).mean() / (mix ** 2).mean())
    peak = float(np.abs(mix).max())
    assert peak < 1.0, f"clipped (peak {peak:.3f})"
    sf.write(a.out / "mix_spatial.wav", mix, sr, subtype="PCM_16")
    for k in (0, 1):
        sf.write(a.out / f"stem_{k}_spatial.wav", stems[k], sr, subtype="PCM_16")
    rec["mix_rms_dbfs"] = round(float(20 * np.log10(np.sqrt((mix ** 2).mean()))), 1)
    rec["mix_peak"] = round(peak, 3)
    print(rec, flush=True)
    json.dump(dict(src=str(a.src), factor=a.factor, report=rec,
                   note="ILD scaled per stem for illustration"),
              open(a.out / "args.json", "w"), indent=1)


if __name__ == "__main__":
    main()
