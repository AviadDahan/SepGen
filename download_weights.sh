#!/usr/bin/env bash
# Download the LTX-2.5 components SepGen uses (Hugging Face Lightricks/LTX-2.5; accept its license
# on the model page and `hf auth login` first) and the two SepGen LoRAs (AviadDahan/SepGen).
#
#   bash download_weights.sh                 # into ./models/ltx2.5 (about 72 GB)
#   SEPGEN_MODELS=/data/ltx2.5 bash download_weights.sh
#
# separate.py / generate.py / train.py read the same directory via --models-dir or $SEPGEN_MODELS.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="${SEPGEN_MODELS:-${HERE}/models/ltx2.5}"
mkdir -p "${MODELS}"

FILES=(
    diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors          # 42 GB, all modes
    text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors            # 26 GB, all modes
    vae/ltx-2.5-video-vae-bf16.safetensors                                 # all modes
    vae/ltx-2.5-audio-vae-bf16.safetensors                                 # all modes
    latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors   # generation
    loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors                  # generation
)
hf download Lightricks/LTX-2.5 "${FILES[@]}" --local-dir "${MODELS}"

# SepGen LoRAs (also fetched on demand by --checkpoint sep-12k / gen-3k).
hf download AviadDahan/SepGen sep-12k/lora_weights.safetensors gen-3k/lora_weights.safetensors

echo "[download] LTX-2.5 weights in ${MODELS}"
