#!/usr/bin/env bash
# Set up the environment: LTX-2 (v1.2.0, the LTX-2.5 release SepGen was built on) in a uv venv,
# plus the few packages SepGen adds.
#
#   bash setup.sh            # clones into ./LTX-2 and creates ./LTX-2/.venv
#   source activate.sh       # puts that venv's python first on PATH
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LTX_DIR="${LTX_DIR:-${HERE}/LTX-2}"
LTX_COMMIT="fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca"   # LTX-2 v1.2.0 (LTX-2.5)

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v ffmpeg >/dev/null 2>&1 || echo "[setup] WARNING: ffmpeg not found on PATH (needed by separate.py)"

if [[ ! -d "${LTX_DIR}" ]]; then
    git clone https://github.com/Lightricks/LTX-2.git "${LTX_DIR}"
fi
git -C "${LTX_DIR}" checkout "${LTX_COMMIT}"

unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
uv sync --project "${LTX_DIR}" --python 3.11
uv pip install --python "${LTX_DIR}/.venv/bin/python" soundfile librosa huggingface_hub pyyaml

"${LTX_DIR}/.venv/bin/python" -c "import ltx_core, ltx_pipelines, ltx_trainer, torch; \
print('[setup] ok: torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cat > "${HERE}/activate.sh" <<EOF
export PATH="${LTX_DIR}/.venv/bin:\${PATH}"
EOF
echo "[setup] done. Next: source activate.sh && bash download_weights.sh"
