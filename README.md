# SepGen: Multi-Stem Audio-Video Separation and Generation in a Single Model

**Aviad Dahan**<sup>1\*</sup>, **Rajaei Khatib**<sup>1\*</sup>, **Yonatan Bitton**<sup>2</sup>,
**Idan Szpektor**<sup>2</sup>, **Lior Wolf**<sup>1</sup>, **Raja Giryes**<sup>1</sup>

<sup>1</sup>Tel Aviv University, <sup>2</sup>Google  ·  <sup>\*</sup>Equal contribution

[Paper](ARXIV_URL) · [Project page](https://sepgen.github.io/) · [Weights](https://huggingface.co/AviadDahan/SepGen)

SepGen extends a pretrained audio-video generator (LTX-2.5) to emit the video, the audio-mix, and
one waveform per captioned source in a single sampling run. The same weights run in two modes,
selected by the noise level of the audio-mix:
- **Generation:** a scene caption and one caption per source produce the video, the audio-mix and
  the two stems.
- **Separation:** an observed video and its audio-mix are held clean while the stems are
  denoised, so the captions say what to extract.

![SepGen overview](docs/overview.png)

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

| Name | Training | Use |
|---|---|---|
| `sep-12k` | 12,000 separation steps (audio-mix always clean) | separation |
| `gen-3k` | `sep-12k` + 3,000 steps with noisy audio-mixes | generation and separation |

Both are LoRA adapters (rank 128, 327M parameters) on the audio stream of LTX-2.5 22B dev.
The scripts download them on demand from [AviadDahan/SepGen](https://huggingface.co/AviadDahan/SepGen)
when given `--checkpoint sep-12k` / `gen-3k`, or take a local path.

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

## Training

```bash
python train.py configs/train_sep12k.yaml --data-root /path/to/precomputed_train
python train.py configs/train_gen3k.yaml --data-root /path/to/precomputed_train \
    --init-checkpoint sep-12k --stop-at-step 3000
```

The training set is not distributed; [docs/DATA.md](docs/DATA.md) gives the precomputed format.
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
