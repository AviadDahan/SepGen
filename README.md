<h2 align="center">SepGen: Multi-Stem Audio-Video Separation and Generation in a Single Model</h2>

<p align="center">
  <b>Aviad Dahan</b><sup>1*</sup>, <b>Rajaei Khatib</b><sup>1*</sup>, <b>Yonatan Bitton</b><sup>2</sup>,
  <b>Idan Szpektor</b><sup>2</sup>, <b>Lior Wolf</b><sup>1</sup>, <b>Raja Giryes</b><sup>1</sup><br>
  <sup>1</sup>Tel Aviv University &nbsp; <sup>2</sup>Google &nbsp; <sup>*</sup>Equal contribution
</p>

<p align="center">
  <!-- ARXIV: replace with <a href="https://arxiv.org/abs/ID"><img src="https://img.shields.io/badge/arXiv-ID-b31b1b.svg" alt="arXiv"></a> -->
  <img src="https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg" alt="arXiv">
  <a href="https://sepgen.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://huggingface.co/collections/AviadDahan/sepgen-6ac3e516dd6528007ad32352"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow.svg" alt="Models"></a>
  <a href="https://huggingface.co/datasets/AviadDahan/SepGen-Dataset"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-orange.svg" alt="Dataset"></a>
</p>

SepGen extends a pretrained audio-video generator (LTX-2.5) to emit the video, the audio-mix, and
one waveform per captioned source in a single sampling run. The same weights run in two modes,
selected by the noise level of the audio-mix:
- **Generation:** a scene caption and one caption per source produce the video, the audio-mix and
  the two stems.
- **Separation:** an observed video and its audio-mix are held clean while the stems are
  denoised, so the captions say what to extract.

![SepGen overview](docs/overview.png)

## Roadmap

- [x] Separation inference (`separate.py`)
- [x] Generation inference (`generate.py`)
- [x] Training code and configs for both checkpoints (`train.py`, `configs/`)
- [x] Checkpoints `sep-12k` and `gen-3k` on Hugging Face
- [x] Localization and new-camera re-rendering, as in the demo videos (`demo/`)
- [x] Training-set manifests: captions, source boxes and provenance ([SepGen-Dataset](https://huggingface.co/datasets/AviadDahan/SepGen-Dataset))
- [ ] Precomputed training latents and text features, to be released soon
- [ ] Evaluation benchmarks and scoring scripts

## Setup

Requirements: a CUDA GPU with 48 GB (all experiments ran on one RTX A6000), Linux, ffmpeg.

```bash
bash setup.sh              # LTX-2 v1.2.0 (LTX-2.5) in a uv venv + SepGen's extra packages
source activate.sh
hf auth login              # accept the LTX-2.5 license on huggingface.co/Lightricks/LTX-2.5 first
bash download_weights.sh   # LTX-2.5 components into ./models/ltx2.5 (~72 GB) + the SepGen LoRAs
```

Use `SEPGEN_MODELS=/path/to/ltx2.5` (or `--models-dir`) to keep the weights elsewhere.

## Checkpoints

| Name | Hugging Face | Training | Use |
|---|---|---|---|
| `sep-12k` | [AviadDahan/SepGen-Separation](https://huggingface.co/AviadDahan/SepGen-Separation) | 12,000 separation steps (audio-mix always clean) | separation |
| `gen-3k` | [AviadDahan/SepGen-Generation](https://huggingface.co/AviadDahan/SepGen-Generation) | `sep-12k` + 3,000 steps with noisy audio-mixes | generation and separation |

Both are LoRA adapters (rank 128, 327M parameters) on the audio stream of LTX-2.5 22B dev.
`--checkpoint sep-12k` / `gen-3k` downloads them on first use (`sepgen/checkpoints.py`); a local
path also works.

## Separation

```bash
python separate.py --batch-manifest examples/separation_manifest.json \
    --checkpoint sep-12k --out-dir outputs/separation
```

To separate a single clip:

```bash
python separate.py --video clip.mp4 \
    --scene-prompt "A blacksmith hammers on an anvil while a church bell rings." \
    --stem0-prompt "A blacksmith hammers rhythmically on an anvil." \
    --stem1-prompt "A church bell rings steadily." \
    --num-frames 113 --frame-rate 25 --out-dir outputs/clip
```

Each clip writes `stem_0.wav` and `stem_1.wav`.
- The base model runs int8-quantized, as in training.
- The paper's settings are the defaults: 30 Euler steps, Cross-Stem Attention Guidance 2.0, 768×512.
- Pass the clip's own frame rate and an 8k+1 frame count, e.g. `--num-frames 137 --frame-rate 24`
  for 24 fps clips.
- Add `--vae-tiling` above about 185 frames.

## Generation

```bash
python generate.py examples/generation_prompts.json --out-dir outputs/generation
python generate.py examples/generation_prompts.json --cells turn_taking_1   # one cell
```

Each cell is `{"segment", "seed", "scene_prompt", "stem_prompts": [stem0, stem1]}`. The examples
are the generation scenes shown on the project page, at the seeds used in the paper. Outputs per
cell:
- `video.mp4` with the generated audio-mix;
- `stem0.wav`, `stem1.wav` and `mix_generated.wav`;
- a muxed mp4 per track, and `sources_split_ears.mp4` with one source per ear.

`configs/generation.json` holds the paper's settings:
- **Rendering:** two stages to 1536×1024, 113 frames at 25 fps, res_2s with 15 steps, the
  LTX-2.5 distilled LoRA at strengths 0.25 and 0.5 as in the upstream two-stage pipeline.
- **Method:** Estimated Separation below σ = 0.97, Cross-Stem Attention Guidance 2.0, checkpoint
  `gen-3k`.

`--stock` renders the same prompts with SepGen off.

## Demo: localization and new-camera re-rendering

`demo/` reproduces the moving-camera videos on the project page:
1. SepGen separates and localizes the two sources on the original clip.
2. LTX-2.3 with the CrossView-Warp IC-LoRA re-renders the clip from an authored moving camera,
   guided by a MoGe-2 depth warp.
3. Each separated source is re-spatialized for that camera (1/r gain, free-field cardioid stereo).

Setup, weights, stages and limitations are in [demo/README.md](demo/README.md).

```bash
bash demo/setup_demo.sh && bash demo/download_demo_weights.sh
bash demo/run_demo.sh <clip.mp4> <mix.wav|-> <scene> <stem0> <stem1> <seed> <out_dir> [--trajectory orbit_rise]
```

## Training

```bash
python train.py configs/train_sep12k.yaml --data-root /path/to/precomputed_train
python train.py configs/train_gen3k.yaml --data-root /path/to/precomputed_train \
    --init-checkpoint sep-12k --stop-at-step 3000
```

The training-set manifests (captions, source boxes and provenance for all 4,843 segments) are on
[Hugging Face](https://huggingface.co/datasets/AviadDahan/SepGen-Dataset); the precomputed latents and
text features will be released soon. [docs/DATA.md](docs/DATA.md) gives the precomputed format.
- **Setup:** one RTX A6000, batch size 1, AdamW at 1e-4 with linear decay. The base model is
  frozen and int8-quantized.
- **`sep-12k`:** about 42 h. The paper's run was resumed once, at step 2000, with the optimizer
  state reset.
- **`gen-3k`:** about 11 h. It warm-starts from `sep-12k` with a fresh optimizer. Its learning-rate
  schedule spans 12,000 steps and training stops at step 3000.

## Code map

| Path | Content |
|---|---|
| `separate.py` | separation (observed video + audio-mix → stems) |
| `generate.py` | generation (captions → video + audio-mix + stems) |
| `train.py` | training entry point on the LTX trainer |
| `sepgen/stemgen/` | training strategy (`joint_stems.py`), separation sampler, attention gates |
| `sepgen/stempipe/` | generation pipeline on the upstream two-stage LTX-2.5 pipeline |
| `sepgen/nag_attn2.py` | Cross-Stem Attention Guidance (NAG with the sibling caption as negative) |
| `configs/` | training configs of both checkpoints and the generation config |
| `demo/` | localization and new-camera re-rendering demo (depth, warp guide, LTX-2.3 render, 3-D lift, audio) |
| `sepgen/localize/` | attention capture hooks and the localization readout |

## Citation

```bibtex
@article{dahan2026sepgen,
  title   = {SepGen: Multi-Stem Audio-Video Separation and Generation in a Single Model},
  author  = {Dahan, Aviad and Khatib, Rajaei and Bitton, Yonatan and Szpektor, Idan and Wolf, Lior and Giryes, Raja},
  journal = {arXiv preprint ARXIV_ID},
  year    = {2026}
}
```

## License

SepGen builds on LTX-2.5 and is a Derivative of LTX-2.x under the
[LTX-2.x Community License Agreement](LICENSE.md). The code and the released weights are
distributed under that agreement, including its use-based restrictions (Attachment A).
`sepgen/stemgen/ltx25_validation_sampler.py` is modified from the LTX-2 trainer.

The demo (`demo/`) also uses:
- LTX-2.3, under the LTX-2 Community License;
- the CrossView-Warp v2 IC-LoRA by Cseti (Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp_v2, Apache-2.0;
  its training renders used CC-BY assets credited in that repository's ATTRIBUTION.md);
- `demo/crossview_warp.py`, ported from ComfyUI-CrossViewWarp
  (github.com/cseti007/ComfyUI-CrossViewWarp, commit 3266a83, Apache-2.0);
- MoGe-2 (microsoft/MoGe code and Ruicheng/moge-2-vitl-normal weights, MIT);
- Gemma-3, under the Gemma terms of use.
