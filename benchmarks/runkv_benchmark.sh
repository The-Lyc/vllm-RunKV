#!/bin/bash
# =============================================================================
# RunKV vs Vanilla vLLM Benchmark Suite
# Full performance benchmark script for paper evaluation
# =============================================================================

set -e
set -o pipefail

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL=${MODEL:-"/home/lyc/hf_models/Qwen2.5-1.5B-Instruct"}
RESULTS_DIR=${RESULTS_DIR:-"${SCRIPT_DIR}/benchmark_results"}
NUM_PROMPTS=${NUM_PROMPTS:-1000}
SEED=${SEED:-42}
CUDA_DEVICE=${CUDA_DEVICE:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
SKIP_EXISTING=${SKIP_EXISTING:-0}               # Set to 1 to skip existing result files
RUN_VANILLA=${RUN_VANILLA:-1}                  # Set to 0 to skip the Vanilla suite
RUN_RUNKV=${RUN_RUNKV:-1}                      # Set to 0 to skip the RunKV suite

# RunKV-specific configuration
RUNKV_NUM_BUFFERS=${RUNKV_NUM_BUFFERS:-3}       # Number of GPU staging buffers
RUNKV_GPU_FRAC=${RUNKV_GPU_FRAC:-1.0}           # Fraction of GPU memory used for staging buffers
RUNKV_CPU_FRAC=${RUNKV_CPU_FRAC:-0.3}           # Fraction of CPU memory used for KV cache
RUNKV_CPU_GB=${RUNKV_CPU_GB:-"30"}                # Optional: CPU memory limit (GB)

# Concurrency control: ensure Vanilla and RunKV are compared under the same conditions.
# If unset, vLLM's internal default (256) may be used by some benchmarks.
# Setting this to 30-64 can better highlight RunKV offload overheads.
BENCH_MAX_NUM_SEQS=${BENCH_MAX_NUM_SEQS:-30}

# ShareGPT dataset URL (override via env var to use a mirror).
SHAREGPT_URL=${SHAREGPT_URL:-"https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"}
# ShareGPT dataset path (defaults to datasets/ under the current working directory).
SHAREGPT_PATH=${SHAREGPT_PATH:-"datasets/ShareGPT_V3_unfiltered_cleaned_split.json"}

# Create results directory
mkdir -p "$RESULTS_DIR"
TIMESTAMP=${TIMESTAMP:-}

# Model max context length (used to auto-trim long-sequence tests).
# Override via env: MODEL_MAX_LEN=...
MODEL_MAX_LEN=${MODEL_MAX_LEN:-}
if [ -z "${MODEL_MAX_LEN}" ]; then
    MODEL_MAX_LEN=$(python - "$MODEL" <<'PY' 2>/dev/null || true
import json
import os
import sys

model = sys.argv[1]
cfg = os.path.join(model, "config.json")
if os.path.isdir(model) and os.path.isfile(cfg):
    with open(cfg, "r", encoding="utf-8") as f:
        data = json.load(f)
    val = (
        data.get("max_position_embeddings")
        or data.get("n_positions")
        or data.get("seq_length")
        or data.get("max_seq_len")
    )
    if isinstance(val, (int, float)) and val > 0:
        print(int(val))
        sys.exit(0)
print("")
PY
    )
fi
MODEL_MAX_LEN=${MODEL_MAX_LEN:-32768}

# =============================================================================
# Helpers
# =============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

write_error_json() {
    local result_file=$1
    local exit_code=$2
    local bench_type=$3
    local tag=$4
    local config=$5
    local log_file=$6
    local cmd_str=$7

    EXIT_CODE="$exit_code" \
    BENCH_TYPE="$bench_type" \
    BENCH_TAG="$tag" \
    BENCH_CONFIG="$config" \
    BENCH_LOG_FILE="$log_file" \
    BENCH_CMD="$cmd_str" \
    python - "$result_file" <<'PY'
import json
import os
import sys

out_path = sys.argv[1]
payload = {
    "_status": "error",
    "exit_code": int(os.environ.get("EXIT_CODE", "1")),
    "bench_type": os.environ.get("BENCH_TYPE", ""),
    "tag": os.environ.get("BENCH_TAG", ""),
    "config": os.environ.get("BENCH_CONFIG", ""),
    "timestamp": os.environ.get("TIMESTAMP", ""),
    "results_dir": os.environ.get("RESULTS_DIR", ""),
    "log_file": os.environ.get("BENCH_LOG_FILE", ""),
    "cmd": os.environ.get("BENCH_CMD", ""),
    "hint": (
        "If this failed due to max_model_len > model limit, you may set "
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 (use with extreme caution)."
    ),
}

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
with open(out_path, "w") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
PY
}

run_bench_json_allow_fail() {
    local result_file=$1
    local log_file=$2
    local bench_type=$3
    local tag=$4
    local config=$5
    shift 5

    if should_skip "$result_file"; then
        return 0
    fi

    local cmd_str="$*"

    local exit_code=0
    "$@" 2>&1 | tee -a "$log_file" || exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log "Saved result to: $result_file"
        return 0
    fi
    log "ERROR: ${bench_type} failed (tag=${tag}, config=${config}, exit=${exit_code})"
    if write_error_json "$result_file" "$exit_code" "$bench_type" "$tag" "$config" "$log_file" "$cmd_str"; then
        log "Recorded failure marker to: $result_file"
    else
        log "WARN: failed to write failure marker to $result_file (see log: $log_file)"
    fi
    return 0
}

ensure_timestamp() {
    if [ -z "${TIMESTAMP:-}" ]; then
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    fi
    export TIMESTAMP
    export RESULTS_DIR
}

should_skip() {
    local result_file=$1
    if [ "${SKIP_EXISTING:-0}" = "1" ] && [ -s "$result_file" ]; then
        if [[ "$result_file" == *.json ]]; then
            if python - "$result_file" <<'PY' >/dev/null 2>&1; then
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    sys.exit(1)

sys.exit(0 if isinstance(data, dict) and data.get("_status") == "error" else 1)
PY
                log "Not skipping (previous run failed): $result_file"
                return 1
            fi
        fi
        log "Skipping (already exists): $result_file"
        return 0
    fi
    return 1
}

detect_timestamp_from_results() {
    local latest_file
    latest_file=$(ls -t "${RESULTS_DIR}"/throughput_vanilla_in*_out*_*.json 2>/dev/null | head -n 1 || true)
    if [ -z "$latest_file" ]; then
        return 1
    fi
    if [[ "$latest_file" =~ _([0-9]{8}_[0-9]{6})\.json$ ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    return 1
}

all_serving_results_exist() {
    local tag=$1
    local qps
    for qps in 1 2 4 8 16; do
        local result_file="${RESULTS_DIR}/serving_${tag}_qps${qps}_${TIMESTAMP}.json"
        if [ ! -s "$result_file" ]; then
            return 1
        fi
        # If previous attempt recorded an error marker, consider it incomplete.
        if python - "$result_file" <<'PY' >/dev/null 2>&1; then
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
sys.exit(0 if isinstance(data, dict) and data.get("_status") == "error" else 1)
PY
            return 1
        fi
    done
    return 0
}

wait_for_server() {
    local port=$1
    local max_wait=${2:-300}
    log "Waiting for server to be ready (port: $port)..."
    timeout $max_wait bash -c "
        until curl -s localhost:${port}/v1/models > /dev/null 2>&1; do
            sleep 2
        done" && return 0 || return 1
}

kill_vllm_servers() {
    log "Cleaning up existing vLLM processes..."
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "python.*vllm" 2>/dev/null || true
    sleep 3
}

is_probably_valid_sharegpt_json() {
    local dataset_path=$1
    if [ ! -f "$dataset_path" ]; then
        return 1
    fi

    python - "$dataset_path" <<'PY'
import os
import sys

path = sys.argv[1]
size = os.path.getsize(path)
if size < 16:
    sys.exit(1)

with open(path, "rb") as f:
    head = f.read(4)

# Handle optional UTF-8 BOM.
if head.startswith(b"\xef\xbb\xbf"):
    head = head[3:]

if not head.startswith(b"["):
    sys.exit(1)

with open(path, "rb") as f:
    f.seek(max(0, size - 65536))
    tail = f.read().rstrip()

if not tail.endswith(b"]"):
    sys.exit(1)
PY
}

download_sharegpt_dataset() {
    local dataset_path=$1

    if [ -z "$dataset_path" ]; then
        log "ERROR: dataset_path is empty; cannot download ShareGPT dataset"
        return 1
    fi

    if [[ "$dataset_path" == "~/"* ]]; then
        dataset_path="${HOME}/${dataset_path#~/}"
    fi

    log "Downloading ShareGPT dataset to: $dataset_path"
    mkdir -p "$(dirname "$dataset_path")"

    local tmp_path="${dataset_path}.tmp.$$"
    rm -f "$tmp_path"

    if command -v wget >/dev/null 2>&1; then
        wget -O "$tmp_path" "$SHAREGPT_URL"
    elif command -v curl >/dev/null 2>&1; then
        curl -fL -o "$tmp_path" "$SHAREGPT_URL"
    else
        log "ERROR: neither wget nor curl found; cannot download dataset"
        return 1
    fi

    mv -f "$tmp_path" "$dataset_path"
}

# =============================================================================
# Benchmark 1: Offline throughput (Throughput)
# The most important benchmark; directly reflects inference throughput.
# =============================================================================
benchmark_throughput() {
    ensure_timestamp
    local tag=$1
    local extra_args=$2
    log "=========================================="
    log "Running offline throughput benchmark: $tag"
    log "=========================================="
    
    # Fix concurrency so Vanilla and RunKV are comparable.
    local max_num_seqs=${BENCH_MAX_NUM_SEQS:-256}

    # Test different input/output length combinations.
    for input_len in 128 512 1024 2048; do
        for output_len in 128 256 512; do
            log "Test config: input=$input_len, output=$output_len, max_num_seqs=$max_num_seqs"
            
            local result_file="${RESULTS_DIR}/throughput_${tag}_in${input_len}_out${output_len}_${TIMESTAMP}.json"
            run_bench_json_allow_fail "$result_file" "${RESULTS_DIR}/throughput_${tag}.log" "throughput" "$tag" "in${input_len}_out${output_len}" \
                env CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
                python -m vllm.entrypoints.cli.main bench throughput \
                    --model $MODEL \
                    --dataset-name random \
                    --random-input-len $input_len \
                    --random-output-len $output_len \
                    --num-prompts $NUM_PROMPTS \
                    --max-num-seqs $max_num_seqs \
                    --seed $SEED \
                    --output-json "$result_file" \
                    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
                    $extra_args
        done
    done
}

# =============================================================================
# Benchmark 2: Latency (Latency)
# Measure latency distribution at fixed batch sizes.
# =============================================================================
benchmark_latency() {
    ensure_timestamp
    local tag=$1
    local extra_args=$2
    log "=========================================="
    log "Running latency benchmark: $tag"
    log "=========================================="
    
    # Test different batch sizes.
    for batch_size in 1 4 8 16 32; do
        for input_len in 128 512 1024; do
            log "Test config: batch=$batch_size, input=$input_len"
            
            local result_file="${RESULTS_DIR}/latency_${tag}_bs${batch_size}_in${input_len}_${TIMESTAMP}.json"
            run_bench_json_allow_fail "$result_file" "${RESULTS_DIR}/latency_${tag}.log" "latency" "$tag" "bs${batch_size}_in${input_len}" \
                env CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
                python -m vllm.entrypoints.cli.main bench latency \
                    --model $MODEL \
                    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
                    --input-len $input_len \
                    --output-len 128 \
                    --batch-size $batch_size \
                    --num-iters 30 \
                    --num-iters-warmup 5 \
                    --output-json "$result_file" \
                    $extra_args
        done
    done
}

# =============================================================================
# Benchmark 3: Serving (Serving)
# Simulate an online serving workload.
# =============================================================================
benchmark_serving() {
    ensure_timestamp
    local tag=$1
    local port=$2
    local extra_args=$3
    log "=========================================="
    log "Running serving benchmark: $tag"
    log "=========================================="
    
    # Test different request rates.
    for qps in 1 2 4 8 16; do
        log "Test config: QPS=$qps"
        
        local result_file="${RESULTS_DIR}/serving_${tag}_qps${qps}_${TIMESTAMP}.json"
        run_bench_json_allow_fail "$result_file" "${RESULTS_DIR}/serving_${tag}.log" "serving" "$tag" "qps${qps}" \
            env CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
	            python -m vllm.entrypoints.cli.main bench serve \
	                --backend vllm \
	                --model $MODEL \
	                --host localhost \
	                --port $port \
	                --ready-check-timeout-sec 0 \
	                --endpoint /v1/completions \
	                --dataset-name random \
	                --random-input-len 512 \
	                --random-output-len 128 \
	                --num-prompts 200 \
                --request-rate $qps \
                --save-result \
                --result-dir "$RESULTS_DIR" \
                --result-filename "$(basename "$result_file")" \
                $extra_args
    done
}

# =============================================================================
# Benchmark 4: Prefix Caching
# Especially useful for evaluating KV cache sharing efficiency.
# =============================================================================
benchmark_prefix_caching() {
    ensure_timestamp
    local tag=$1
    local extra_args=$2
    log "=========================================="
    log "Running Prefix Caching benchmark: $tag"
    log "=========================================="
    
    # Test different prefix lengths and repeat counts.
    for prefix_len in 256 512 1024; do
        for repeat_count in 5 10 20; do
            log "Test config: prefix=$prefix_len, repeat=$repeat_count"
            
            local result_file="${RESULTS_DIR}/prefix_${tag}_plen${prefix_len}_rep${repeat_count}_${TIMESTAMP}.log"
            if should_skip "$result_file"; then
                continue
            fi
            
            if env CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "${SCRIPT_DIR}/benchmark_prefix_caching.py" \
                --model $MODEL \
                --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
                --enable-prefix-caching \
                --num-prompts 50 \
                --repeat-count $repeat_count \
                --prefix-len $prefix_len \
                --input-length-range 512:1024 \
                --output-len 128 \
                $extra_args 2>&1 | tee "$result_file"; then
                :
            else
                log "ERROR: prefix benchmark failed (tag=${tag}, prefix=${prefix_len}, repeat=${repeat_count})"
            fi
            
            log "Saved result to: $result_file"
        done
    done
}

# =============================================================================
# Benchmark 5: ShareGPT dataset
# Evaluate performance on real conversational data.
# =============================================================================
benchmark_sharegpt() {
    ensure_timestamp
    local tag=$1
    local extra_args=$2
    local dataset_path=${SHAREGPT_PATH}
    
    log "=========================================="
    log "Running ShareGPT dataset benchmark: $tag"
    log "=========================================="
    
    if [ -f "$dataset_path" ] && ! is_probably_valid_sharegpt_json "$dataset_path"; then
        log "ShareGPT dataset file appears incomplete/corrupted; re-downloading: $dataset_path"
        rm -f "$dataset_path"
    fi

    if [ ! -f "$dataset_path" ]; then
        download_sharegpt_dataset "$dataset_path"
    fi
    
    local result_file="${RESULTS_DIR}/sharegpt_${tag}_${TIMESTAMP}.json"
    local max_num_seqs=${BENCH_MAX_NUM_SEQS:-256}

    run_bench_json_allow_fail "$result_file" "${RESULTS_DIR}/sharegpt_${tag}.log" "sharegpt" "$tag" "" \
        env CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
        python -m vllm.entrypoints.cli.main bench throughput \
            --model $MODEL \
            --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
            --dataset-name sharegpt \
            --dataset-path "$dataset_path" \
            --num-prompts 500 \
            --max-num-seqs $max_num_seqs \
            --seed $SEED \
            --output-json "$result_file" \
            $extra_args
}

# =============================================================================
# Benchmark 6: Long sequence (RunKV key scenario!)
# RunKV's main advantage is supporting very long sequences; this is the key benchmark.
# =============================================================================
benchmark_long_sequence() {
    ensure_timestamp
    local tag=$1
    local extra_args=$2
    log "=========================================="
    log "Running long-sequence benchmark: $tag (RunKV key scenario)"
    log "=========================================="
    log "Model max context length: ${MODEL_MAX_LEN}"
    
    # RunKV's key advantage: very long sequence support.
    # Test different total sequence lengths (must not exceed the model limit).
    local max_num_seqs=${BENCH_MAX_NUM_SEQS:-256}

    for total_len in 4096 8192 16384 32768; do
        if [ "$total_len" -gt "$MODEL_MAX_LEN" ]; then
            log "Skipping total_len=$total_len (exceeds model max context length $MODEL_MAX_LEN)"
            continue
        fi
        for output_len in 128 256 512; do
            local input_len=$((total_len - output_len))
            if [ "$input_len" -le 0 ]; then
                log "Skipping total=$total_len, output=$output_len (input_len<=0)"
                continue
            fi
            log "Long-sequence test: total=$total_len, input=$input_len, output=$output_len"
            
            local result_file="${RESULTS_DIR}/longseq_${tag}_in${input_len}_out${output_len}_${TIMESTAMP}.json"
            run_bench_json_allow_fail "$result_file" "${RESULTS_DIR}/longseq_${tag}.log" "longseq" "$tag" "total${total_len}_in${input_len}_out${output_len}" \
                env CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
                python -m vllm.entrypoints.cli.main bench throughput \
                    --model $MODEL \
                    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
                    --dataset-name random \
                    --random-input-len $input_len \
                    --random-output-len $output_len \
                    --num-prompts 100 \
                    --max-model-len $total_len \
                    --max-num-seqs $max_num_seqs \
                    --seed $SEED \
                    --output-json "$result_file" \
                    $extra_args
        done
    done
}

# =============================================================================
# Benchmark 7: High concurrency (KV cache memory pressure)
# =============================================================================
benchmark_high_concurrency() {
    ensure_timestamp
    local tag=$1
    local extra_args=$2
    log "=========================================="
    log "Running high-concurrency benchmark: $tag"
    log "=========================================="
    
    # Test performance under larger batch sizes (higher concurrency).
    for max_num_seqs in 16 32 64 128 256; do
        log "Concurrency test: max_num_seqs=$max_num_seqs"
        
        local result_file="${RESULTS_DIR}/concurrency_${tag}_seqs${max_num_seqs}_${TIMESTAMP}.json"
        run_bench_json_allow_fail "$result_file" "${RESULTS_DIR}/concurrency_${tag}.log" "concurrency" "$tag" "seqs${max_num_seqs}" \
            env CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
            python -m vllm.entrypoints.cli.main bench throughput \
                --model $MODEL \
                --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
                --dataset-name random \
                --random-input-len 512 \
                --random-output-len 128 \
                --num-prompts 500 \
                --max-num-seqs $max_num_seqs \
                --seed $SEED \
                --output-json "$result_file" \
                $extra_args
    done
}

# =============================================================================
# Main flow
# =============================================================================
main() {
    ensure_timestamp
    log "=========================================="
    log "RunKV vs Vanilla vLLM Benchmark Suite"
    log "=========================================="
    log "Model: $MODEL"
    log "Results dir: $RESULTS_DIR"
    log "Timestamp: $TIMESTAMP"
    
    # Clean up environment.
    kill_vllm_servers
    
    # =========================================================================
    # Part 1: Vanilla vLLM benchmark suite
    # =========================================================================
    if [ "${RUN_VANILLA:-1}" = "1" ]; then
        log "=========================================="
        log "Starting Vanilla vLLM benchmarks"
        log "=========================================="
        
        # 1.1 Offline throughput
        benchmark_throughput "vanilla" ""
        
        # 1.2 Latency
        benchmark_latency "vanilla" ""
        
        # 1.3 ShareGPT dataset
        benchmark_sharegpt "vanilla" ""
        
        # 1.4 Prefix Caching
        benchmark_prefix_caching "vanilla" ""
        
        # 1.5 Long sequence (important: RunKV key comparison)
        benchmark_long_sequence "vanilla" ""
        
        # 1.6 High concurrency
        benchmark_high_concurrency "vanilla" ""
        
        # 1.7 Serving (requires starting a server)
        if all_serving_results_exist "vanilla"; then
            log "Serving results already exist; skipping Vanilla vLLM server startup"
        else
            log "Starting Vanilla vLLM server..."
            CUDA_VISIBLE_DEVICES=$CUDA_DEVICE vllm serve $MODEL \
                --port 8000 \
                --max-model-len 8192 \
                --gpu-memory-utilization $GPU_MEMORY_UTILIZATION &
            VANILLA_PID=$!
            
            if wait_for_server 8000; then
                benchmark_serving "vanilla" 8000 ""
            else
                log "ERROR: Vanilla vLLM server failed to start"
            fi
            
            kill $VANILLA_PID 2>/dev/null || true
            kill_vllm_servers
            sleep 5
        fi
    else
        log "Skipping Vanilla vLLM benchmarks (RUN_VANILLA=0)"
    fi
    
    # =========================================================================
    # Part 2: RunKV vLLM benchmark suite
    # RunKV is a layer-wise KV cache offload mechanism. It stores KV cache on CPU
    # and stages it to GPU on demand to support very long sequences.
    # =========================================================================
    if [ "${RUN_RUNKV:-1}" != "1" ]; then
        log "Skipping RunKV vLLM benchmarks (RUN_RUNKV=0)"
        return 0
    fi

    log "=========================================="
    log "Starting RunKV vLLM benchmarks"
    log "=========================================="
    
    # RunKV key args:
    # --enable-runkv                    Enable layer-wise KV cache offload
    # --runkv-num-device-buffers N      Number of GPU staging buffers (ring buffer)
    # --runkv-gpu-memory-fraction F     Fraction of GPU memory reserved for staging buffers
    # --runkv-cpu-memory-fraction F     Fraction of CPU memory reserved for KV cache
    # --runkv-cpu-memory-gb G           Absolute CPU memory limit for KV cache (GB)
    # --runkv-max-staging-blocks N      Max blocks per staging buffer
    
    RUNKV_ARGS="--enable-runkv \
        --runkv-num-device-buffers ${RUNKV_NUM_BUFFERS:-3} \
        --runkv-cpu-memory-fraction ${RUNKV_CPU_FRAC:-0.3}"
    
    if [ -n "${RUNKV_CPU_GB:-}" ]; then
        RUNKV_ARGS="$RUNKV_ARGS --runkv-cpu-memory-gb $RUNKV_CPU_GB"
    fi

    if [ -n "${RUNKV_GPU_FRAC:-}" ]; then
        RUNKV_ARGS="$RUNKV_ARGS --runkv-gpu-memory-fraction $RUNKV_GPU_FRAC"
    fi
    
    log "RunKV args: $RUNKV_ARGS"
    
    # 2.1 Offline throughput
    benchmark_throughput "runkv" "$RUNKV_ARGS"
    
    # 2.2 Latency
    benchmark_latency "runkv" "$RUNKV_ARGS"
    
    # 2.3 ShareGPT dataset
    benchmark_sharegpt "runkv" "$RUNKV_ARGS"
    
    # 2.4 Prefix Caching (RunKV can be combined with prefix caching)
    # benchmark_prefix_caching "runkv" "$RUNKV_ARGS --enable-prefix-caching"
    
    # 2.5 Long sequence (RunKV key advantage!)
    benchmark_long_sequence "runkv" "$RUNKV_ARGS"
    
    # 2.6 High concurrency
    benchmark_high_concurrency "runkv" "$RUNKV_ARGS"
    
    # 2.7 Serving
    if all_serving_results_exist "runkv"; then
        log "Serving results already exist; skipping RunKV vLLM server startup"
    else
        log "Starting RunKV vLLM server..."
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE vllm serve $MODEL \
            --port 8000 \
            --max-model-len 8192 \
            --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
            $RUNKV_ARGS &
        RUNKV_PID=$!
        
        if wait_for_server 8000; then
            benchmark_serving "runkv" 8000 ""
        else
            log "ERROR: RunKV vLLM server failed to start"
        fi
        
        kill $RUNKV_PID 2>/dev/null || true
        kill_vllm_servers
    fi
    
    # =========================================================================
    # Summary
    # =========================================================================
    log "=========================================="
    log "Benchmark complete!"
    log "Results saved to: $RESULTS_DIR"
    log "=========================================="
    
    # Generate summary report
    python - << 'EOF'
import json
import os
import glob

results_dir = os.environ.get('RESULTS_DIR', './benchmark_results')
timestamp = os.environ.get('TIMESTAMP', '')

summary = {"vanilla": {}, "runkv": {}}

for json_file in glob.glob(f"{results_dir}/*.json"):
    filename = os.path.basename(json_file)
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict) and data.get("_status") in ("error", "skipped"):
            # Skip error markers written by the benchmark runner.
            continue
        
        if "vanilla" in filename:
            tag = "vanilla"
        elif "runkv" in filename:
            tag = "runkv"
        else:
            continue
            
        if "throughput" in filename:
            key = f"throughput_{filename}"
            summary[tag][key] = {
                "tokens_per_second": data.get("tokens_per_second", 0),
                "requests_per_second": data.get("requests_per_second", 0)
            }
        elif "latency" in filename:
            key = f"latency_{filename}"
            summary[tag][key] = {
                "avg_latency": data.get("avg_latency", 0),
                "p99_latency": data.get("percentiles", {}).get("99", 0)
            }
    except Exception as e:
        print(f"Error processing {json_file}: {e}")

# Save summary
summary_file = f"{results_dir}/summary_{timestamp}.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved to: {summary_file}")
EOF
}

# =============================================================================
# Entry point
# =============================================================================
case "${1:-all}" in
    download_sharegpt)
        download_sharegpt_dataset "${2:-${SHAREGPT_PATH}}"
        ;;
    resume)
        SKIP_EXISTING=1
        if [ -z "${TIMESTAMP:-}" ]; then
            detected_ts=$(detect_timestamp_from_results || true)
            if [ -z "$detected_ts" ]; then
                log "ERROR: no TIMESTAMP found to resume. Please set: TIMESTAMP=YYYYMMDD_HHMMSS SKIP_EXISTING=1 $0 all"
                exit 1
            fi
            TIMESTAMP="$detected_ts"
            log "Existing results detected; resuming with TIMESTAMP=$TIMESTAMP (SKIP_EXISTING=1)"
        else
            log "Resuming with user-specified TIMESTAMP=$TIMESTAMP (SKIP_EXISTING=1)"
        fi
        main
        ;;
    throughput)
        benchmark_throughput "${2:-test}" "${3:-}"
        ;;
    latency)
        benchmark_latency "${2:-test}" "${3:-}"
        ;;
    serving)
        benchmark_serving "${2:-test}" "${3:-8000}" "${4:-}"
        ;;
    prefix)
        benchmark_prefix_caching "${2:-test}" "${3:-}"
        ;;
    sharegpt)
        benchmark_sharegpt "${2:-test}" "${3:-}"
        ;;
    all)
        main
        ;;
    *)
        echo "Usage: $0 {all|resume|throughput|latency|serving|prefix|sharegpt|download_sharegpt} [tag] [extra_args]"
        echo "  Env:"
        echo "    RESULTS_DIR=...           (default: ${SCRIPT_DIR}/benchmark_results)"
        echo "    TIMESTAMP=YYYYMMDD_HHMMSS (set to reuse a run id)"
        echo "    SKIP_EXISTING=1           (skip existing result files)"
        echo "    RUN_VANILLA=0             (skip Vanilla suite)"
        echo "    RUN_RUNKV=0               (skip RunKV suite)"
        exit 1
        ;;
esac
