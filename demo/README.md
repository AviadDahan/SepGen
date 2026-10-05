# Demo: localization and new-camera re-rendering

This pipeline produces the "moving camera" demo videos on the project page. It runs on one clip
with a static camera and two sound sources. SepGen separates the two sources and localizes them on
the original clip. The clip is then re-rendered from an authored moving camera, and each separated
source is re-spatialized for that camera.

| # | Stage | Script | Environment | Device |
|---|---|---|---|---|
| 1 | Separation + 2-D localization on the original clip | `localize.py` | SepGen (LTX-2.5) | GPU |
| 2 | Metric depth (MoGe-2, 50° horizontal field of view) | `moge_depth.py` | MoGe | GPU |
| 3a | Camera move + depth-warp guide | `build_warp_guide.py`, `crossview_warp.py` | MoGe | CPU |
| 3b | New view: LTX-2.3 + CrossView-Warp IC-LoRA, 1536×1024 | `render_crossview.py` | LTX-2.3 | GPU |
| 4 | Per-stem 3-D tracks from the 2-D tracks and the depth | `lift_tracks.py` | CPU | CPU |
| 5 | Per stem: 1/r gain and free-field cardioid stereo for the moving camera; optional ×2.5 widening of the left-right level difference | `spatialize_stems.py`, `exaggerate_spatial.py` | CPU | CPU |
| 6 | Mux | `mux.py` | CPU | CPU |

## How it works

1. **Localization.** `localize.py` runs the same render as `separate.py`, using `sep-12k`, 30 Euler
   steps and Cross-Stem Attention Guidance 2.0. Hooks (`sepgen/localize/capture.py`) record two
   channels during the render without changing the stems:
   - **t2v**, the video's text cross-attention, replayed with each stem's own caption and read in
     output space;
   - **v2a**, the stem spans' attention onto the video.

   For each stem, `sepgen/localize/readout.py` picks the steadier channel using a fixed,
   ground-truth-free score (`sepgen/localize/sampler_stats.json`). It then tracks a
   support-weighted geometric-median pointer with a constant-velocity Kalman smoother. A stem whose
   score passes the fixed threshold is flagged as low-confidence in `track.json`.
2. **Depth.** MoGe-2 gives metric depth per frame. The field of view is fixed at 50°, so the depth
   matches the lens the warp assumes.
3. **New view.**
   - The camera orbits a pivot at the subjects' depth: the median 3-D point of the left and right
     thirds of the first frame. The default move is `orbit_rise`, which reaches -45° azimuth and 20°
     elevation. The other moves are `swing` and `arc`. All stay within the LoRA's reliable range:
     |azimuth| ≤ 45°, elevation between -20° and 30°.
   - Each frame is reprojected through the depth into the moving camera. Disoccluded areas are
     filled magenta, as the CrossView-Warp LoRA expects.
   - LTX-2.3 renders the new view, conditioned on the original clip and the warp guide.
   - Base and LoRAs: the 22B dev base, with the distilled LoRA fused at strength 1.0 into both
     stages; the CrossView LoRA is used in stage 1 only.
   - Stage 2 starts at σ = 0.421875. The upstream start, σ = 0.909375, invents objects when the only
     prompt is the trigger word.
4. **3-D lift.**
   - Direction: per latent frame, the geometric median of the top 2% attention cells' 3-D points.
   - Range: the nearest significant mode of the depth in a ±128-pixel window at the pointer. A
     pooled range would slide toward background that lies on the same ray as the subject.
   - Measurement confidence comes from that depth mode and from the stem's own loudness. Silent
     anchors are interpolated between voiced ones instead of being fitted.
   - The track is smoothed with a 3-D constant-velocity Kalman filter and RTS smoother, with one
     innovation-gated refit, then resampled to every frame.
5. **Audio.** Each track is re-expressed relative to the moving camera, frame by frame.
   - Gain follows the distance as r(0)/r(t), clamped to ±12 dB.
   - Direction drives two cardioid "ears" 18 cm apart, which set the per-ear gain and the
     inter-aural delay. There is no room model.
   - The stems are RMS-matched to the original audio-mix, so the overall level does not change.
   - The project page also widens each stem's left-right level difference by 2.5×, keeping its
     energy. This is for illustration and is not part of the method; turn it off with `--widen 0`.

## Setup

Requirements:
- the main SepGen setup (`setup.sh`, `download_weights.sh`);
- a 48 GB GPU, as for separation;
- about 80 GB more disk for the LTX-2.3 weights;
- ffmpeg and ffprobe.

```bash
bash demo/setup_demo.sh            # LTX-2.3 env (LTX-2 @ 7dc613f), MoGe env, CPU env; writes demo/demo_env.sh
hf auth login                      # accept the Gemma-3 terms on google/gemma-3-12b-it-qat-q4_0-unquantized first
bash demo/download_demo_weights.sh # LTX-2.3 dev, distilled LoRA, upscaler, CrossView LoRA, Gemma-3, MoGe-2
```

Use `LTX23_MODELS=/path` to keep the LTX-2.3 weights elsewhere, and `SEPGEN_MODELS` for the
LTX-2.5 weights, as in the main README.

## Run

```bash
bash demo/run_demo.sh <clip.mp4> <mix.wav | -> <scene> <stem0> <stem1> <render_seed> <out_dir> \
    [--trajectory orbit_rise|swing|arc] [--widen 2.5|0] [--checkpoint sep-12k|PATH] [--stages ...]
```

The input clip needs at least 121 frames at 24 fps; the demo renders 121 frames, about 5 s. Pass
`-` as the mix to separate the clip's own soundtrack.

The page's lawn-mower clip is included:

```bash
E=examples/demo/lawn_mower.json
j() { python -c "import json,sys; print(json.load(open('$E'))[sys.argv[1]])" "$1"; }
bash demo/run_demo.sh "$(j video)" "$(j mix_wav)" "$(j scene_prompt)" "$(j stem0_prompt)" \
    "$(j stem1_prompt)" "$(j render_seed)" outputs/demo/lawn_mower --trajectory "$(j trajectory)"
```

`--stages localize,depth,...` runs a subset of the stages. Each stage reads the outputs of the
earlier ones from `<out_dir>`. The render stage runs in two processes: the first encodes the
trigger prompt and saves `render/prompt_ctx.pt`, and the second renders. This keeps Gemma's memory
out of the render process.

## Outputs

| Path | Content |
|---|---|
| `localize/` | `stem_0.wav`, `stem_1.wav`, `track.json` (2-D tracks, chosen channel, low-confidence flag), `maps.npz`, `overlay.png`, `repro.mp4` |
| `depth/` | `depth.npz` (depth, mask, intrinsics), `depth_preview.mp4` |
| `guide/` | `warp.mp4`, `trajectory.json` (per-frame camera poses), `orbit_view.png`, `guide_preview.mp4` |
| `render/` | `render.mp4` (with LTX-2.3's own audio), `render_mixaudio.mp4` (with the original mix) |
| `tracks_3d.json` | per-stem 3-D tracks in the original camera frame |
| `audio/` | `stem_{0,1}_depth.wav`, `stem_{0,1}_spatial.wav`, `mix_spatial.wav`, `gains.json` |
| `audio_widened/` | the ×2.5 widened `stem_{0,1}_spatial.wav` and `mix_spatial.wav` |
| `final/` | `original.mp4`, `moving_mix.mp4` (new view, original mix), `moving_ours.mp4` (new view, re-spatialized mix), `moving_stem{0,1}.mp4` |

Approximate times on one RTX A6000: localization 7 min; depth 2 min; warp guide 3-5 min on CPU;
render 1 min for the prompt plus 5 min; lift, audio and mux under 1 min.

## Reproducibility

Every stage was checked on the lawn-mower clip against the run that produced the page's video.
Stems, 2-D tracks, depth, warp guide, camera poses, 3-D tracks and audio are bit-identical.
The new-view render and the page's three videos are byte-identical as well.

## Limitations

- **Static input camera, two sources, 121 frames at 24 fps.** The 3-D lift treats the original
  camera as fixed.
- **The new view is generated.** The render follows the warp guide's geometry, but it can
  re-imagine identity. It can duplicate a subject, keeping a copy near its original position, or
  invent objects in large disoccluded regions such as the sky. Inspect `guide/guide_preview.mp4`
  first. If a subject leaves the warp at the end of the move, use `--trajectory swing`, a smaller
  scale (`build_warp_guide.py --traj-scale`) or an explicit `--pivot`.
- **Localization reads generation attention.** The attention gives semantic relevance, not
  source identity. A source can bind to a similar distractor, such as an engine sound bound to
  distant traffic, and no ground-truth-free signal reliably detects this. Distant or receding
  objects get noisier depth-mode ranges.
- The 3-D lift and the audio are an analytic re-rendering for the demo: free-field, no room,
  no head-related transfer function. They were not scored on the benchmark.

## Licenses

- **MoGe-2:** code and the `Ruicheng/moge-2-vitl-normal` weights are MIT.
- **CrossView-Warp v2 IC-LoRA** (`Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp_v2`, by Cseti): Apache-2.0.
  Its training renders used CC-BY assets, credited in the repository's ATTRIBUTION.md.
- **`crossview_warp.py`:** ported from [ComfyUI-CrossViewWarp](https://github.com/cseti007/ComfyUI-CrossViewWarp)
  (commit 3266a83), Apache-2.0. The file header lists the changes.
- **LTX-2.3:** the LTX-2 Community License.
- **Gemma-3:** the Gemma terms of use.
