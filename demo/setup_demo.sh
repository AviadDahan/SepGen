#!/usr/bin/env bash
# Set up the three extra environments the demo needs, next to the main SepGen env (setup.sh),
# which runs the localization (demo/localize.py):
#
#   LTX-2.3 (uv venv)   the CrossView IC-LoRA new-view render (render_crossview.py). LTX-2 at
#                       7dc613f, the LTX-2.3 release the IC-LoRA was trained for; it cannot share
#                       the LTX-2.5 venv (different torch / transformers pins).
#   MoGe-2  (uv venv)   metric depth (moge_depth.py) and the warp guide (build_warp_guide.py),
#                       pinned to the versions the demo clips were made with.
#   CPU     (uv venv)   the 3-D lift, the audio and the mux: numpy 1.26 / scipy 1.14 only
#                       (requirements_cpu.txt explains why not the main env's numpy 2).
#
#   bash demo/setup_demo.sh
#   LTX23_DIR=/opt/LTX-2.3 MOGE_ENV=/opt/moge CPU_ENV=/opt/cpu bash demo/setup_demo.sh
#
# Writes demo/demo_env.sh, which run_demo.sh sources for the three interpreters.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "${HERE}")"
LTX23_DIR="${LTX23_DIR:-${REPO}/LTX-2.3}"
MOGE_ENV="${MOGE_ENV:-${REPO}/envs/moge}"
CPU_ENV="${CPU_ENV:-${REPO}/envs/cpu}"
LTX_DIR="${LTX_DIR:-${REPO}/LTX-2}"                      # the main env from setup.sh
LTX23_COMMIT="7dc613f80c09c94ffb3d6526023479fcf1c676f8"   # LTX-2 main, 2026-05-28 (LTX-2.3)

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup_demo] installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v ffmpeg >/dev/null 2>&1 || echo "[setup_demo] WARNING: ffmpeg not found on PATH"
command -v ffprobe >/dev/null 2>&1 || echo "[setup_demo] WARNING: ffprobe not found on PATH"
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT

# ------------------------------------------------------------------ LTX-2.3
if [[ ! -d "${LTX23_DIR}" ]]; then
    git clone https://github.com/Lightricks/LTX-2.git "${LTX23_DIR}"
fi
git -C "${LTX23_DIR}" checkout "${LTX23_COMMIT}"
# At this commit ltx_pipelines/utils/blocks.py imports
# ltx_pipelines.multigpu.delegating_builder, which the public repository does not ship. It is
# used only as a type annotation, so a generic stand-in makes the package importable.
MULTIGPU_DIR="${LTX23_DIR}/packages/ltx-pipelines/src/ltx_pipelines/multigpu"
if [[ ! -f "${MULTIGPU_DIR}/delegating_builder.py" ]]; then
    mkdir -p "${MULTIGPU_DIR}"
    printf '"""Stand-in for a package missing from the public LTX-2 repository."""\n' \
        > "${MULTIGPU_DIR}/__init__.py"
    printf '%s\n' \
        '"""Type-annotation-only stand-in; nothing on the single-GPU path instantiates it."""' \
        'from typing import Generic, TypeVar' '' '_T = TypeVar("_T")' '' '' \
        'class DelegatingBuilder(Generic[_T]):' '    """No-op generic stand-in."""' \
        > "${MULTIGPU_DIR}/delegating_builder.py"
fi
uv sync --project "${LTX23_DIR}" --frozen --extra xformers
"${LTX23_DIR}/.venv/bin/python" -c "import ltx_core, ltx_pipelines, xformers, torch; \
from ltx_pipelines.ic_lora import ICLoraPipeline; \
print('[setup_demo] LTX-2.3 ok: torch', torch.__version__, 'xformers', xformers.__version__)"

# ------------------------------------------------------------------ MoGe-2
if [[ ! -x "${MOGE_ENV}/bin/python" ]]; then
    uv venv "${MOGE_ENV}" --python 3.11
fi
uv pip install --python "${MOGE_ENV}/bin/python" \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0 torchvision==0.26.0
uv pip install --python "${MOGE_ENV}/bin/python" -r "${HERE}/requirements_moge.txt"
"${MOGE_ENV}/bin/python" -c "import cv2, numpy, torch; from moge.model.v2 import MoGeModel; \
print('[setup_demo] MoGe ok: torch', torch.__version__, 'numpy', numpy.__version__, \
'cv2', cv2.__version__)"

# ------------------------------------------------------------------ CPU (lift, audio, mux)
if [[ ! -x "${CPU_ENV}/bin/python" ]]; then
    uv venv "${CPU_ENV}" --python 3.11
fi
uv pip install --python "${CPU_ENV}/bin/python" -r "${HERE}/requirements_cpu.txt"
"${CPU_ENV}/bin/python" -c "import numpy, scipy, soundfile; \
print('[setup_demo] CPU env ok: numpy', numpy.__version__, 'scipy', scipy.__version__)"

cat > "${HERE}/demo_env.sh" <<EOF
export SEPGEN_PY="${LTX_DIR}/.venv/bin/python"
export LTX23_PY="${LTX23_DIR}/.venv/bin/python"
export MOGE_PY="${MOGE_ENV}/bin/python"
export CPU_PY="${CPU_ENV}/bin/python"
EOF
echo "[setup_demo] done; interpreters in ${HERE}/demo_env.sh. Next: bash demo/download_demo_weights.sh"
