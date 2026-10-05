#!/usr/bin/env bash
# Download the extra weights the demo needs (the LTX-2.5 weights and the sep-12k LoRA come from
# ../download_weights.sh and are fetched on demand):
#
#   LTX-2.3 22B dev + distilled LoRA 384 (v1.1) + spatial upscaler x2 (v1.1)   Lightricks/LTX-2.3
#   CrossView-Warp v2 IC-LoRA (Cseti, Apache-2.0)            Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp_v2
#   Gemma-3 12B (LTX-2.3's text encoder; gated: accept the Gemma terms on its model page first)
#                                                            google/gemma-3-12b-it-qat-q4_0-unquantized
#   MoGe-2 ViT-L (MIT)                                       Ruicheng/moge-2-vitl-normal (HF cache)
#
#   bash demo/download_demo_weights.sh                     # into ./models/ltx2.3 (about 80 GB)
#   LTX23_MODELS=/data/ltx2.3 bash demo/download_demo_weights.sh
#
# Revisions are pinned to the ones the demo clips were made with.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "${HERE}")"
MODELS="${LTX23_MODELS:-${REPO}/models/ltx2.3}"
mkdir -p "${MODELS}/loras" "${MODELS}/gemma3"

hf download Lightricks/LTX-2.3 --revision 76730e634e70a28f4e8d51f5e29c08e40e2d8e74 \
    ltx-2.3-22b-dev.safetensors \
    ltx-2.3-22b-distilled-lora-384-1.1.safetensors \
    ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --local-dir "${MODELS}"
hf download Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp_v2 \
    --revision 17a34a4664586131340d68f73c4aa3152ed27efc \
    LTX2.3-22B_IC-LoRA-CrossView-Warp_v2_6000.safetensors \
    --local-dir "${MODELS}/loras"
hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
    --revision 68f7ee4fbd59087436ada77ed2d62f373fdd4482 \
    --local-dir "${MODELS}/gemma3"
hf download Ruicheng/moge-2-vitl-normal --revision cb0e8bbd6b1e243589717c78e750b1ba4c093acf \
    model.pt

echo "[download] LTX-2.3 weights in ${MODELS}; pass LTX23_MODELS=${MODELS} to run_demo.sh"
