#!/usr/bin/env bash
# SepGen demo: localize the sources of a clip, re-render it from a moving camera, and re-spatialize
# each separated source for that camera.
#
#   bash demo/run_demo.sh <clip.mp4> <mix.wav | -> <scene> <stem0> <stem1> <seed> <out_dir> \
#       [--trajectory orbit_rise|swing|arc] [--checkpoint sep-12k|PATH] [--widen 2.5|0] \
#       [--stages localize,depth,guide,render,lift,audio,mux]
#
#   <mix.wav | ->   the audio-mix to separate; "-" = the clip's own soundtrack
#   <seed>          seed of the new-view render
#
# The clip must have at least 121 frames at 24 fps (the demo renders 121 frames, 5 s).
# Stages (each writes its own subdirectory of <out_dir>):
#   localize  separation + 2-D localization on the original clip      SepGen env (LTX-2.5), GPU
#   depth     MoGe-2 metric depth of the original clip                MoGe env, GPU
#   guide     camera move + depth-warp guide                          MoGe env, CPU
#   render    new view with the LTX-2.3 CrossView IC-LoRA             LTX-2.3 env, GPU (2 passes)
#   lift      per-stem 3-D tracks from the 2-D tracks + depth         CPU env
#   audio     1/r gain + free-field cardioid stereo per stem for the  CPU env
#             moving camera, and the optional ILD widening (--widen, default 2.5 as on the
#             project page; 0 = off)
#   mux       final videos: original; new view with the original mix,    CPU env
#             with the re-spatialized mix, and with each re-spatialized stem
#
# Interpreters come from demo/demo_env.sh (written by setup_demo.sh), else SEPGEN_PY / LTX23_PY /
# MOGE_PY / CPU_PY. Weights: SEPGEN_MODELS (LTX-2.5, download_weights.sh) and LTX23_MODELS (LTX-2.3,
# download_demo_weights.sh); GEMMA_ROOT overrides the Gemma-3 directory ($LTX23_MODELS/gemma3).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "${HERE}")"
[[ -f "${HERE}/demo_env.sh" ]] && source "${HERE}/demo_env.sh"
SEPGEN_PY="${SEPGEN_PY:-${REPO}/LTX-2/.venv/bin/python}"
LTX23_PY="${LTX23_PY:-${REPO}/LTX-2.3/.venv/bin/python}"
MOGE_PY="${MOGE_PY:-${REPO}/envs/moge/bin/python}"
CPU_PY="${CPU_PY:-${REPO}/envs/cpu/bin/python}"
export SEPGEN_MODELS="${SEPGEN_MODELS:-${REPO}/models/ltx2.5}"
LTX23_MODELS="${LTX23_MODELS:-${REPO}/models/ltx2.3}"
GEMMA_ROOT="${GEMMA_ROOT:-${LTX23_MODELS}/gemma3}"
export PYTHONUNBUFFERED=1

if [[ $# -lt 7 ]]; then
    sed -n '2,12p' "$0"
    exit 1
fi
VIDEO="$(realpath "$1")"; MIX_IN="$2"; SCENE="$3"; STEM0="$4"; STEM1="$5"; SEED="$6"
OUT="$7"
shift 7
TRAJECTORY="orbit_rise"; CHECKPOINT="sep-12k"; WIDEN="2.5"
STAGES="localize,depth,guide,render,lift,audio,mux"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --trajectory) TRAJECTORY="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --widen) WIDEN="$2"; shift 2 ;;
        --stages) STAGES="$2"; shift 2 ;;
        *) echo "unknown option $1"; exit 1 ;;
    esac
done
NUM_FRAMES=121
FRAME_RATE=24
mkdir -p "${OUT}"
OUT="$(realpath "${OUT}")"

if [[ "${MIX_IN}" == "-" ]]; then
    MIX="${OUT}/mix.wav"
    [[ -f "${MIX}" ]] || ffmpeg -y -loglevel error -i "${VIDEO}" -vn -acodec pcm_s16le "${MIX}"
else
    MIX="$(realpath "${MIX_IN}")"
fi

has() { [[ ",${STAGES}," == *",$1,"* ]]; }
stamp() { echo "[demo] $(date '+%F %T') $*"; }

if has localize; then
    stamp "localize"
    "${SEPGEN_PY}" "${HERE}/localize.py" --video "${VIDEO}" --mix-wav "${MIX}" \
        --scene-prompt "${SCENE}" --stem0-prompt "${STEM0}" --stem1-prompt "${STEM1}" \
        --checkpoint "${CHECKPOINT}" --num-frames "${NUM_FRAMES}" --frame-rate "${FRAME_RATE}" \
        --out-dir "${OUT}/localize"
fi
if has depth; then
    stamp "depth"
    "${MOGE_PY}" "${HERE}/moge_depth.py" --video "${VIDEO}" --num-frames "${NUM_FRAMES}" \
        --frame-rate "${FRAME_RATE}" --out-dir "${OUT}/depth"
fi
if has guide; then
    stamp "guide (${TRAJECTORY})"
    "${MOGE_PY}" "${HERE}/build_warp_guide.py" --video "${VIDEO}" \
        --depth-npz "${OUT}/depth/depth.npz" --trajectory "${TRAJECTORY}" \
        --num-frames "${NUM_FRAMES}" --frame-rate "${FRAME_RATE}" --out-dir "${OUT}/guide"
fi
if has render; then
    RENDER_ARGS=(--video "${VIDEO}" --warp "${OUT}/guide/warp.mp4" --mix-wav "${MIX}"
                 --seed "${SEED}" --models-dir "${LTX23_MODELS}" --gemma-root "${GEMMA_ROOT}"
                 --num-frames "${NUM_FRAMES}" --frame-rate "${FRAME_RATE}"
                 --out-dir "${OUT}/render")
    if [[ ! -f "${OUT}/render/prompt_ctx.pt" ]]; then
        stamp "render: prompt encode"     # a separate process, so Gemma's memory is released
        "${LTX23_PY}" "${HERE}/render_crossview.py" "${RENDER_ARGS[@]}"
    fi
    stamp "render"
    "${LTX23_PY}" "${HERE}/render_crossview.py" "${RENDER_ARGS[@]}"
fi
if has lift; then
    stamp "lift"
    "${CPU_PY}" "${HERE}/lift_tracks.py" --localize-dir "${OUT}/localize" \
        --depth-npz "${OUT}/depth/depth.npz" --frame-rate "${FRAME_RATE}" \
        --out "${OUT}/tracks_3d.json"
fi
if has audio; then
    stamp "audio"
    "${CPU_PY}" "${HERE}/spatialize_stems.py" --tracks "${OUT}/tracks_3d.json" \
        --localize-dir "${OUT}/localize" --mix-wav "${MIX}" \
        --trajectory-json "${OUT}/guide/trajectory.json" --frame-rate "${FRAME_RATE}" \
        --out-dir "${OUT}/audio"
    if [[ "${WIDEN}" != "0" ]]; then
        "${CPU_PY}" "${HERE}/exaggerate_spatial.py" --src "${OUT}/audio" \
            --factor "${WIDEN}" --out "${OUT}/audio_widened"
    fi
fi
if has mux; then
    stamp "mux"
    SPATIAL_DIR="${OUT}/audio"
    [[ "${WIDEN}" != "0" ]] && SPATIAL_DIR="${OUT}/audio_widened"
    "${CPU_PY}" "${HERE}/mux.py" --video "${VIDEO}" --mix-wav "${MIX}" \
        --render "${OUT}/render/render.mp4" --spatial-dir "${SPATIAL_DIR}" \
        --out-dir "${OUT}/final"
fi
stamp "done: ${OUT}"
