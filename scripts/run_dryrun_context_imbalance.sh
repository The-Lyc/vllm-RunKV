#!/usr/bin/env bash
# Run four no-pressure dryrun context experiments and summarize per-step imbalance.
#
# Typical use:
#   bash scripts/run_dryrun_context_imbalance.sh
#
# The context values are prompt word counts, matching the existing offline
# workload generator. They are labeled 1k/2k/4k/8k in the analysis output.
# Per-context defaults reproduce the final OPT-2.7B experiment settings:
#   1k: 32 prompts x 128 output tokens
#   2k: 32 prompts x 128 output tokens
#   4k: 32 prompts x  32 output tokens
#   8k: 16 prompts x  32 output tokens

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
RUNNER="${ROOT}/examples/offline_inference/run_opt_feedback_observation.py"
ANALYZER="${ROOT}/tools/analyze_dryrun_imbalance.py"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-/data/models/opt-2.7b-8k}"
CONTEXTS="${CONTEXTS:-1000 2000 4000 8000}"
PREFIX_BLOCKS="${PREFIX_BLOCKS:-10000}"
NUM_DEVICE_BUFFERS="${NUM_DEVICE_BUFFERS:-3}"

RUN_GROUP="${RUN_GROUP:-$(date +%Y%m%d_%H%M)_opt2.7b_dryrun_context}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/exp_results/dryrun_context/${RUN_GROUP}}"
ANALYSIS_OUTPUT_DIR="${ANALYSIS_OUTPUT_DIR:-${ROOT}/exp_results/analysis/dryrun_imbalance/${RUN_GROUP}}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

# Keep the diagnostic collection mode consistent with ordinary dryrun runs.
ENABLE_NSYS="${ENABLE_NSYS:-1}"
ENABLE_OPT_COMPONENT_MFU_PROFILING="${ENABLE_OPT_COMPONENT_MFU_PROFILING:-1}"
ENABLE_NVTX="${ENABLE_NVTX:-1}"
ENABLE_PROFILE="${ENABLE_PROFILE:-1}"
NSYS_SAMPLE="${NSYS_SAMPLE:-cpu}"
NSYS_EXTRA_ARGS="${NSYS_EXTRA_ARGS:---capture-range=cudaProfilerApi --capture-range-end=stop}"

mkdir -p "${OUTPUT_ROOT}"

read -r -a context_values <<< "${CONTEXTS}"
analysis_inputs=()

context_label() {
    case "$1" in
        1000) echo "1k" ;;
        2000) echo "2k" ;;
        4000) echo "4k" ;;
        8000) echo "8k" ;;
        *) echo "$1" ;;
    esac
}

set_context_config() {
    case "$1" in
        1000 | 2000)
            NUM_PROMPTS=32
            MAX_TOKENS=128
            GPU_MEMORY_UTILIZATION=0.7
            GPU_MEMORY_FRACTION=0.95
            CPU_MEMORY_GB=46.566128730773926
            CPU_MEMORY_FRACTION=0.3
            ;;
        4000)
            NUM_PROMPTS=32
            MAX_TOKENS=32
            GPU_MEMORY_UTILIZATION=0.8
            GPU_MEMORY_FRACTION=0.7
            CPU_MEMORY_GB=93.13225746154785
            CPU_MEMORY_FRACTION=0.6
            ;;
        8000)
            NUM_PROMPTS=16
            MAX_TOKENS=32
            GPU_MEMORY_UTILIZATION=0.8
            GPU_MEMORY_FRACTION=0.6
            CPU_MEMORY_GB=93.13225746154785
            CPU_MEMORY_FRACTION=0.6
            ;;
        *)
            echo "Unsupported context ${1}; add its experiment settings to set_context_config()." >&2
            exit 2
            ;;
    esac
    # The recorded final runs left --max-num-seqs unset, so the runner uses
    # its default of --num-prompts.
    MAX_NUM_SEQS=""
}

echo "Dryrun context imbalance experiment"
echo "  model:                  ${MODEL}"
echo "  contexts:               ${CONTEXTS}"
echo "  settings:               final per-context OPT-2.7B experiment settings"
echo "  output_root:            ${OUTPUT_ROOT}"
echo "  analysis_output_dir:    ${ANALYSIS_OUTPUT_DIR}"
echo

for context in "${context_values[@]}"; do
    label="$(context_label "${context}")"
    set_context_config "${context}"
    run_tag="${RUN_GROUP}_${label}_b${NUM_PROMPTS}x${MAX_TOKENS}"
    run_dir="${OUTPUT_ROOT}/${label}"
    prehook_dir="${run_dir}/prehook_timing"

    mkdir -p "${prehook_dir}"
    echo "======================================================================"
    echo "Running context ${label}: PROMPT_WORDS=${context}, batch=${NUM_PROMPTS}x${MAX_TOKENS}"
    echo "  gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}, gpu_memory_fraction=${GPU_MEMORY_FRACTION}"
    echo "  cpu_memory_gb=${CPU_MEMORY_GB}, cpu_memory_fraction=${CPU_MEMORY_FRACTION}"
    echo "  output=${run_dir}"
    echo "======================================================================"

    DRY_RUN=1 \
    USE_STATE_MACHINE=1 \
    RUNKV_PREHOOK_TIMING=1 \
    RUNKV_PREHOOK_TIMING_DIR="${prehook_dir}" \
    ENABLE_NSYS="${ENABLE_NSYS}" \
    ENABLE_OPT_COMPONENT_MFU_PROFILING="${ENABLE_OPT_COMPONENT_MFU_PROFILING}" \
    ENABLE_NVTX="${ENABLE_NVTX}" \
    ENABLE_PROFILE="${ENABLE_PROFILE}" \
    NSYS_SAMPLE="${NSYS_SAMPLE}" \
    NSYS_EXTRA_ARGS="${NSYS_EXTRA_ARGS}" \
    MODEL="${MODEL}" \
    PREFIX_BLOCKS="${PREFIX_BLOCKS}" \
    NUM_PROMPTS="${NUM_PROMPTS}" \
    MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    PROMPT_WORDS="${context}" \
    MAX_TOKENS="${MAX_TOKENS}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    GPU_MEMORY_FRACTION="${GPU_MEMORY_FRACTION}" \
    NUM_DEVICE_BUFFERS="${NUM_DEVICE_BUFFERS}" \
    CPU_MEMORY_GB="${CPU_MEMORY_GB}" \
    CPU_MEMORY_FRACTION="${CPU_MEMORY_FRACTION}" \
    OUTPUT_DIR="${run_dir}" \
    RUN_TAG="${run_tag}" \
    MANIFEST_FILE="${run_dir}/manifest.json" \
    "${PYTHON_BIN}" "${RUNNER}" \
        --resource-pressure-kind none \
        "$@" 2>&1 | tee "${run_dir}/run.log"

    analysis_inputs+=(--input "${run_dir}/*.flat.jsonl" --label "${label}-${NUM_PROMPTS}x${MAX_TOKENS}")
done

if [[ "${RUN_ANALYSIS}" == "1" ]]; then
    echo
    echo "Running per-step imbalance analysis"
    "${PYTHON_BIN}" "${ANALYZER}" \
        "${analysis_inputs[@]}" \
        --only-modal-step-shape \
        --output-dir "${ANALYSIS_OUTPUT_DIR}"
else
    echo
    echo "Skipping analysis because RUN_ANALYSIS=${RUN_ANALYSIS}"
fi

echo
echo "Dryrun context imbalance experiment complete"
echo "  runs:     ${OUTPUT_ROOT}"
echo "  analysis: ${ANALYSIS_OUTPUT_DIR}"
