# Training data format

`train.py` reads a precomputed dataset: one `.pt` file per training segment in each of the
directories below, all under `data.preprocessed_data_root`. A segment is used only if it is
present in every directory. The paper's set has 4,843 two-source segments of 113 frames at 25 fps,
encoded at 768×512. They come from four families:
- CelebV-HQ talking-head pairs;
- URMP two-instrument recordings;
- generated two-source clips;
- VGGSound / MUSIC-21 composites.

Each audio-mix is the sum of its two sources. Its manifests (captions in every register, source boxes,
and where each source comes from) are at
[AviadDahan/SepGen-Dataset](https://huggingface.co/datasets/AviadDahan/SepGen-Dataset); the precomputed
latents and text features will be released soon.

Latents are produced by the LTX-2.5 encoders, and text features by the LTX-2.5 text encoder
(Gemma-4) before the embeddings connector. The upstream `ltx_trainer` preprocessing scripts
`process_videos.py` and `process_captions.py` write exactly these formats.

## Directories

| Directory | Content per segment |
|---|---|
| `latents` | `{"latents": [128, 15, 16, 24] bf16, "num_frames": 15, "height": 16, "width": 24, "fps": 25.0}`: video latent |
| `audio_latents_mix` | `{"latents": [8, T, 16] f32, "num_time_steps": T, "frequency_bins": 16, "duration": s}`: audio-mix latent (T = 114 for 4.52 s) |
| `audio_latents_stem0`, `audio_latents_stem1` | same format, one per source; mix = stem0 + stem1 in the waveform domain |
| `conditions` | scene caption: `{"video_prompt_embeds": [1024, 4096] bf16, "audio_prompt_embeds": [1024, 2048] bf16, "prompt_attention_mask": [1024] int64, "prompt": str}` |
| `conditions_scene_audio` | scene caption for the audio-mix span: `{"audio_prompt_embeds", "audio_prompt_attention_mask", "prompt", "is_padding"}` |
| `conditions_stem_audio0`, `conditions_stem_audio1` | one caption per source, same fields as `conditions` plus `"is_padding"` |
| `*_pos`, `*_short` variants of the five caption directories | the positional and short caption registers (below) |
| `register_avail` | `{"avail": [3] f32, "registers": ["", "_pos", "_short"]}`: which registers exist for this segment |
| `stem_boxes` | `{"boxes": [2, 15, 4] f32, "valid": [2] bool, "source": str}`: per source, per latent frame, the source's region as normalized xyxy |

## Caption registers

Each segment carries up to three caption registers. One register is drawn per training sample, with
the probabilities in `joint_stem.register_mix`:
- **semantic** (no suffix): describes the source, e.g. "A flute, playing.";
- **positional** (`_pos`): describes where the source is, e.g. "The instrument on the left, playing.";
- **short** (`_short`): a noun phrase, e.g. "A flute.".

When the positional register is drawn, `stem_boxes` restricts each stem's audio queries in
video-to-audio attention to the video tokens inside its box (`joint_stem.positional_spatial_mask`).
The bias is training-only: no box is given at inference.
