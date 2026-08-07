# 在 RunKV 中运行 Llama 2 7B

本仓库已经能够通过 vLLM 的模型注册表和标准 Llama 权重加载器读取
`LlamaForCausalLM`。RunKV 的普通逐层 KV 换入/换出以及
`io_hidden_states` 重计算也不依赖 OPT。Llama 需要单独适配的是
`prev_layer_output_dynamic` 动态逐层回放，因为它使用 RoPE positions 和
`(hidden_states, residual)` 两路 residual-stream 表示。

本文以本机模型目录为例：

```text
/data/models/Llama-2-7b-hf-8k
```

该目录中的配置是 FP16 Llama 2 7B：32 层、hidden size 4096、32 个
attention/KV heads、最大上下文 8192，并包含完整的 safetensors 分片。

## 1. 先验证模型配置

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python - <<'PY'
from vllm.config import ModelConfig

path = "/data/models/Llama-2-7b-hf-8k"
config = ModelConfig(
    model=path,
    tokenizer=path,
    dtype="float16",
    max_model_len=8192,
    enforce_eager=True,
)
print(
    config.hf_config.model_type,
    config.architecture,
    config.max_model_len,
    config.dtype,
)
PY
```

预期输出包含：

```text
llama LlamaForCausalLM 8192 torch.float16
```

## 2. 只启用 RunKV KV offload

这个模式不执行 hidden-state 重计算，是最适合先验证权重加载和普通 RunKV
路径的配置：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
.venv/bin/vllm serve /data/models/Llama-2-7b-hf-8k \
  --dtype float16 \
  --load-format safetensors \
  --max-model-len 8192 \
  --enable-runkv \
  --runkv-cpu-memory-gb 8 \
  --runkv-max-staging-blocks 512
```

## 3. 启用 Llama 动态逐层回放

首版动态路径要求单 GPU、TP/PP/DP/DCP 均为 1、完全 eager、关闭 cascade
attention 和 ubatching，并且只允许一个 KV cache group：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
.venv/bin/vllm serve /data/models/Llama-2-7b-hf-8k \
  --dtype float16 \
  --load-format safetensors \
  --max-model-len 8192 \
  --enforce-eager \
  --disable-cascade-attn \
  --enable-runkv \
  --runkv-cpu-memory-gb 8 \
  --runkv-max-staging-blocks 512 \
  --runkv-enable-layer-recompute \
  --runkv-layer-recompute-mode prev_layer_output_dynamic \
  --runkv-layer-recompute-planner static \
  --runkv-layer-recompute-io-prefix-blocks 128
```

`--runkv-layer-recompute-io-prefix-blocks 128` 表示所有层都直接从 CPU
换入前 128 个 KV block，并回放其后的 token。它只是一个起始值，应根据
PCIe 带宽和 GPU 计算能力调优：

- 值越大：H2D KV 流量越多，回放计算越少。
- 值越小：H2D KV 流量越少，回放计算越多。
- 也可以传 32 个逗号分隔的整数，为每层单独配置。

可以把 planner 改为 `feedback` 做在线调整。Llama 也可以使用
`tightllm` planner，但必须先用当前版本的离线 profiler 生成
model-aware profile；profiler 会按 Llama 的 RMSNorm、SwiGLU
gate/up/down 投影和 attention/KV head 配置校准计算模型。

### 3.1 生成 Llama TightLLM profile

先确认 `nvidia-smi` 和 PyTorch CUDA 均正常，并停止 Flux contender、其他
GPU workload 以及会占用 PCIe 的后台任务。然后在实际运行 benchmark 的
同一块 GPU、同一 PCIe 拓扑和同一软件环境中执行：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python -m vllm.v1.profiling.tightllm_offline_profiler \
  --model /data/models/Llama-2-7b-hf-8k \
  --output exp_results/tightllm_profiles/ubuntu/Llama-2-7b-hf-8k.json \
  --seq-lengths 128 256 512 1024 2048 4096 8192 \
  --device cuda:0
```

profile 同时记录 attention/FFN MFU、PCIe H2D 带宽、GPU 峰值 FLOPS 和
Llama 架构系数。它是硬件与运行环境相关的测量结果：更换 GPU、PCIe
拓扑、CUDA/PyTorch 栈或影响频率的功耗设置后应重新生成；profiling
期间存在资源竞争也会使后续 TightLLM 决策失真。

不要使用旧的 lowercase 文件
`exp_results/tightllm_profiles/ubuntu/llama2-7b-8k.json`。该 legacy
profile 没有 `model_type` 和 Llama SwiGLU 架构数据，会被当作旧 OPT
profile，不能作为 Llama 的有效输入。新的 canonical 文件名区分大小写，
是 `Llama-2-7b-hf-8k.json`。

## 4. 容量配置

GPU staging slot 数必须至少覆盖一次调度中所有活跃请求引用的不同 KV
block；动态回放只会跳过这些 block 的 H2D，不会取消 slot 映射。

默认 block size 为 16 时：

- 单条 8192-token 请求需要 `8192 / 16 = 512` 个 staging blocks。
- 多请求并发时，按它们同时存活的 block 总数继续增加，或限制
  `max_num_seqs` / 上下文长度。
- Llama 2 7B FP16 权重约 12.55 GiB，24 GiB GPU 适合先做单请求验证。

Llama 2 7B 的 8192-token KV 约为 4 GiB；启用重计算后，32 层
hidden-state store 还会增加约 2 GiB CPU 内存。
`--runkv-cpu-memory-gb` 是 KV 与 hidden-state backing store 的总预算；
此外还要为权重加载、pageable buffers 和进程本身保留足够的主机内存。

## 5. 自动化 benchmark

Llama 使用与 OPT 相同的两级入口形式：

```text
configs/*.json
  -> scripts/run_llama_*_batch.py
    -> scripts/run_llama_*_pipeline.py
      -> examples/offline_inference/run_llama_feedback_observation.py (RunKV)
      -> examples/offline_inference/run_tightllm_observation.py (TightLLM)
```

普通环境的 1k/2k/4k/8k 批量配置是
`configs/benchmark_batch_llama2_7b.json`。默认并发依次为 32/32/16/8，
配合 47.9 GiB CPU backing-store 预算，使长上下文 case 不会因默认预算不足
而退化为排队执行。先检查展开后的全部命令：

```bash
.venv/bin/python scripts/run_llama_benchmark_batch.py \
  configs/benchmark_batch_llama2_7b.json \
  --dry-run
```

确认模型、显存和 nsys 可用后执行：

```bash
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python scripts/run_llama_benchmark_batch.py \
  configs/benchmark_batch_llama2_7b.json
```

如需让每个 case 与现有 Flux contender 竞争，配置文件不变，只替换入口：

```bash
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python scripts/run_llama_benchmark_batch_with_flux.py \
  configs/benchmark_batch_llama2_7b.json
```

资源分阶段变化的 IO/SM 配置分别为：

```bash
# PCIe H2D contention
.venv/bin/python scripts/run_llama_staged_resource_benchmark_batch.py \
  configs/staged_resource_benchmark_llama2_7b_io.json \
  --dry-run

# SM contention
.venv/bin/python scripts/run_llama_staged_resource_benchmark_batch.py \
  configs/staged_resource_benchmark_llama2_7b_sm.json \
  --dry-run
```

去掉 `--dry-run` 即会顺序执行各 workload。staged 配置把并发数缩放为
8×1k、4×2k、2×4k、1×8k，使同时存活的 prompt token 总量约为 8K，
并显式设置 `max_staging_blocks=640`，适合作为第一轮资源竞争实验。

normal 和 staged 的 Llama JSON 默认都设置
`tightllm_profile: "exp_results/tightllm_profiles/ubuntu/Llama-2-7b-hf-8k.json"`
与 `skip_tightllm: false`。因此每个 workload 会分别运行 RunKV 和
TightLLM，并收集两套 artifacts；在真正执行前必须先按 3.1 节生成
profile。只有明确做单系统实验时，才把 `skip_tightllm` 改为 `true`。

这些 JSON 可以控制：

- planner、初始 `prefix_blocks`、prompt 构造词和 batch/context/decode；
- CPU 总 backing-store 预算、GPU staging fraction/buffer 数/显式 block 数；
- H2D copy 与 replay allocation policy；
- nsys、NVTX、CUDA capture、component timing 和 prehook timing；
- staged IO/SM pressure 的 schedule、模式、设备、带宽或矩阵参数。

Llama 配置使用 `prompt_word: "the"`。本地 tokenizer 下，1000/2000/4000/
8000 个 `the` 加固定前缀约为 1013/2013/4013/8013 tokens；最高档再生成
32 tokens 仍低于 8192。不要直接改回 OPT 使用的 `replay`：该词在此
Llama tokenizer 中会拆成两个 token，使 8k case 超出模型上下文。

普通结果写到 `exp_results/llama2_7b_benchmark/<run-tag>/`，staged 结果写到
`exp_results/staged_llama2_7b/<pattern>/<run-tag>/`。每个 run 包含 manifest、
RunKV/TightLLM 各自的 component timing JSONL、可选的
`.nsys-rep`/`.sqlite` 和双系统分析目录。普通 pipeline manifest 的
`systems.runkv` 与 `systems.tightllm` 分别指向两套 artifacts。

## 6. 验证

不需要 GPU 的定向单元测试：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  -p no:cacheprovider -q \
  tests/v1/profiling/test_tightllm_offline_profiler.py \
  tests/v1/profiling/test_tightllm_ilp_planner.py \
  tests/v1/kv_offload/test_layer_recompute_manager.py \
  tests/v1/kv_offload/test_llama_benchmark_runner.py \
  tests/benchmarks/test_llama_benchmark_scripts.py \
  tests/v1/kv_offload/test_llama_dynamic_replay_forward.py \
  tests/v1/kv_offload/test_opt_dynamic_replay_plan.py
```

在 NVIDIA GPU 可见的机器上，先做小上下文的 baseline 对比：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
tests/v1/kv_offload/test_dynamic_replay_e2e_concurrent.py \
  --model /data/models/Llama-2-7b-hf-8k \
  --enable-dynamic-replay \
  --compare-baseline \
  --num-requests 2 \
  --min-tokens 4 \
  --max-tokens 4 \
  --max-num-seqs 2 \
  --max-model-len 256 \
  --gpu-memory-utilization 0.9 \
  --cpu-memory-gb 4 \
  --cpu-memory-fraction 0.1 \
  --max-staging-blocks 64 \
  --layer-recompute-io-prefix-blocks 1
```

该脚本会打印 baseline 与动态回放结果的差异；确认 token IDs 一致后，再逐步
提高上下文、并发数和 staging blocks。启动日志中应能看到
`Llama dynamic replay forward entered`。
