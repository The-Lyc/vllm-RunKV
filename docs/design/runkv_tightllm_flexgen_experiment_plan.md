# RunKV / TightLLM / FlexGen 对比实验方案

> 状态: 实验设计草案
> 适用范围: RunKV feedback planner、复现版 TightLLM ILP planner、原版 FlexGen baseline
> 当前已有入口: `examples/offline_inference/opt_replay_component_mfu.py`、`examples/offline_inference/run_tightllm_observation.py`

---

## 1. 目标

本实验用于回答三个问题：

1. 在系统资源稳定时，RunKV 的 feedback-driven replay 是否能在离线和在线推理中达到或超过 TightLLM / FlexGen 的吞吐和延迟表现。
2. 在系统资源阶段式变化时，RunKV 是否比依赖离线 profile 的 TightLLM 更快适应 IO 带宽或 GPU SM 可用性的变化。
3. 在真实负载切换场景中，RunKV 的在线反馈控制是否能降低资源受限阶段的尾延迟和吞吐损失，并在资源恢复后避免过量 replay 的副作用。

核心对比对象：

| 系统 | 实验中名称 | 说明 |
|---|---|---|
| RunKV | `runkv-feedback` | `prev_layer_output_dynamic` + feedback planner，建议默认启用 state machine controller |
| 复现版 TightLLM | `tightllm-replay` | 当前 vLLM 内复现的 TightLLM ILP planner；每个 step 基于当前 batch 重新求 ILP budget，但代价模型使用离线 profile |
| 复现版 TightLLM + feedback | `tightllm-feedback` | 可选 ablation，在每 step ILP budget 上叠加 per-layer runtime feedback correction |
| 原版 FlexGen | `flexgen-original` | 外部原版 FlexGen baseline，结果统一汇总到同一 schema |

主论文/报告里建议以三方对比为主：`runkv-feedback`、`tightllm-replay`、`flexgen-original`。`tightllm-feedback` 作为 ablation，用来证明“每 step 离线模型 ILP + 少量 per-layer 反馈”能缓解阶段式变化，但仍不同于 RunKV 的 layer-wise feedback control。

---

## 2. 公共实验约束

### 2.1 固定环境

除非实验组明确要改变资源，否则每个 run 固定以下条件：

| 维度 | 默认值 |
|---|---|
| GPU | 单卡，记录 GPU 型号、显存容量、PCIe/NVLink 信息 |
| CPU memory | 足够容纳 offloaded KV cache，记录 NUMA / pinned memory 设置 |
| CUDA / PyTorch | 记录版本 |
| vLLM commit | 记录 git commit |
| FlexGen commit | 记录 git commit 或 release |
| 模型 | 优先 `facebook/opt-2.7b` 或本地 `/home/lyc/hf_models/opt-2.7b-8k` |
| dtype | fp16 |
| tensor parallel | 1 |
| random seed | 固定 |
| warmup | 每个 setting 预热 1 次，不计入结果 |
| repeats | 每个 setting 至少 3 次，报告 median 和 min/max 或 std |

如果资源允许，主模型用 OPT-2.7B，补充模型用 OPT-6.7B 或 OPT-13B。主线先跑通 OPT-2.7B，避免 FlexGen 和 RunKV 因模型规模差异引入额外变量。

### 2.2 KV offload 公共配置

RunKV 和 TightLLM 复现版统一使用：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `gpu_memory_fraction` | `0.9` | KV offload 可用 GPU fraction |
| `num_device_buffers` | `3` | DMA device buffer 数 |
| `gpu_memory_utilization` | `0.7` | vLLM engine GPU memory utilization |
| `enable_async_prefetch` | `true` | 开启异步 prefetch |
| `enable_async_offload` | `true` | 开启异步 offload |
| `cpu_memory_limit` | `5e10` | CPU KV memory limit |
| `enable_layer_recompute` | `true` | 非 baseline 实验开启 |
| `layer_recompute_mode` | `prev_layer_output_dynamic` | 当前 RunKV/TightLLM 对比路径 |

RunKV planner：

```text
--planner feedback
--use-state-machine
```

TightLLM planner：

```text
--planner tightllm
--tightllm-profile-path tightllm_profile.json
```

TightLLM + feedback correction：

```text
--planner tightllm
--tightllm-profile-path tightllm_profile.json
--tightllm-feedback-correction
```

### 2.3 统一输出

每个 run 需要产出一个 manifest，至少包含：

```json
{
  "run_id": "...",
  "system": "runkv-feedback | tightllm-replay | tightllm-feedback | flexgen-original",
  "experiment_group": "steady_offline | steady_online | staged_offline | staged_online | case_study",
  "model": "...",
  "workload": "...",
  "resource_pattern": "steady | io_low_high_low | sm_low_high_low | load_burst",
  "settings": {},
  "metrics": {},
  "artifacts": {}
}
```

RunKV / TightLLM 复现版应保存：

- `opt_component_mfu_*.jsonl`
- `opt_component_mfu_*.flat.jsonl`，如果已启用 flatten 输出
- `nsys-rep`，对关键 settings 开启
- stdout/stderr log
- run manifest

FlexGen 原版应保存：

- 原始 stdout/stderr
- throughput / latency summary
- 可解析的 JSON 或 CSV 结果
- run manifest

### 2.4 指标

必须报告：

| 指标 | 离线推理 | 在线推理 |
|---|---|---|
| 吞吐 | output tokens/s、total tokens/s | served output tokens/s、completed requests/s |
| 延迟 | E2E latency P50/P95/P99 | TTFT P50/P95/P99、TPOT P50/P95/P99、E2E P50/P95/P99 |
| 稳定性 | step latency std、吞吐波动 | SLO violation rate、queueing delay |
| 资源 | GPU utilization、SM active、PCIe/H2D/D2H 带宽 | 同左 |
| planner 行为 | replay blocks、replay ratio、imbalance、budget、controller state | 同左，按时间桶聚合 |

阶段式变化实验额外报告：

- stage 内平均吞吐、P95 latency 和 step latency 分布
- S2 资源受限阶段相对 S1 稳态的 throughput drop 和 latency inflation
- S2 资源受限阶段的累计性能损失，例如 latency area-under-curve 或 token throughput deficit
- S2 内 RunKV 的控制动作是否带来更低的系统层损失，而不是只看 imbalance 是否回到 deadband
- S3 资源恢复后的 replay 过量/不足副作用，例如 budget overshoot、latency overshoot、恢复后前若干 step 的吞吐损失
- stage 切换瞬间的最大尾延迟放大倍数

---

## 3. 实验组 1: 稳态资源 + 离线推理

### 3.1 目的

在没有外部资源扰动时，比较三种系统在离线批处理推理中的吞吐上限、延迟和 replay planner 决策质量。

### 3.2 入口

RunKV / TightLLM 复现版使用：

```bash
python examples/offline_inference/opt_replay_component_mfu.py \
  --model $MODEL \
  --prefix-blocks $PREFIX_BLOCKS \
  --num-prompts $NUM_PROMPTS \
  --prompt-words $PROMPT_WORDS \
  --max-tokens $MAX_TOKENS \
  --gpu-memory-fraction 0.9 \
  --num-device-buffers 3 \
  --planner $PLANNER \
  --output-dir $OUTPUT_DIR \
  --run-tag $RUN_TAG
```

TightLLM 可继续用 wrapper：

```bash
python examples/offline_inference/run_tightllm_observation.py
```

FlexGen 原版使用独立 runner，但 workload 参数必须和本组一致。

### 3.3 Settings

主扫描矩阵：

| 轴 | 取值 |
|---|---|
| system | `runkv-feedback`, `tightllm-replay`, `flexgen-original` |
| `num_prompts` | `8, 16, 32, 64, 128` |
| `prompt_words` | `1000, 4000, 8000` |
| `max_tokens` | `32, 128` |
| `prefix_blocks` | `1000` |

建议先跑 smoke subset：

| setting | 值 |
|---|---|
| `num_prompts` | `32` |
| `prompt_words` | `4000` |
| `max_tokens` | `32` |
| `prefix_blocks` | `1000` |

完整矩阵规模：

```text
3 systems × 5 num_prompts × 3 prompt_lengths × 2 decode_lengths × 3 repeats
= 270 runs
```

如果时间紧，先固定 `max_tokens=32`，完整矩阵降为 135 runs。

### 3.4 预期图表

- throughput vs `num_prompts`
- E2E latency CDF
- per-layer replay ratio heatmap
- per-layer imbalance time series
- attention / FFN / H2D / D2H 时间占比堆叠图

### 3.5 关键判断

- RunKV 是否在大 batch / 长 prompt 下维持更高 throughput。
- TightLLM 纯离线 profile 在稳态下是否接近 RunKV。
- FlexGen 是否因 coarse-grained offload / scheduling 在短 decode 或长 prompt 场景中尾延迟更高。

---

## 4. 实验组 2: 稳态资源 + 在线推理

### 4.1 目的

在系统资源稳定时，比较三种系统在在线 serving 中的吞吐-延迟曲线，尤其是 knee point、TTFT、TPOT 和 SLO 达成率。

### 4.2 入口

RunKV / TightLLM 复现版建议使用 vLLM OpenAI server + `benchmarks/benchmark_serving.py`：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --gpu-memory-utilization 0.7 \
  --kv-offload-config "$KV_OFFLOAD_CONFIG"
```

```bash
python benchmarks/benchmark_serving.py \
  --backend vllm \
  --model $MODEL \
  --dataset-name sharegpt \
  --dataset-path $SHAREGPT_PATH \
  --request-rate $REQUEST_RATE \
  --num-prompts $NUM_PROMPTS \
  --save-result \
  --result-dir $OUTPUT_DIR
```

FlexGen 原版使用其 online/serving runner。如果原版没有 OpenAI-compatible server，应使用 replayed-arrival driver，把同一批 request trace 按到达时间喂给 FlexGen，并输出同一组 TTFT/TPOT/E2E 指标。

### 4.3 Settings

| 轴 | 取值 |
|---|---|
| system | `runkv-feedback`, `tightllm-replay`, `flexgen-original` |
| arrival process | Poisson |
| `request_rate` | `2, 4, 8, 16, 32` req/s |
| `num_prompts` | `1000` |
| dataset | ShareGPT，另加 long-prompt synthetic |
| output length | dataset 原始分布 + capped `max_tokens=128` |

long-prompt synthetic 设置：

| 参数 | 取值 |
|---|---|
| prompt words | `4000` 或 `8000` |
| output tokens | `32` 或 `128` |
| request count | `1000` |

### 4.4 预期图表

- throughput-latency curve
- request_rate vs TTFT P95
- request_rate vs TPOT P95
- SLO violation rate vs request_rate
- queueing delay time series

### 4.5 关键判断

- RunKV 的 feedback 是否能把在线场景的 P95/P99 压低。
- TightLLM 纯离线 profile 在 arrival rate 接近 knee point 时是否出现更明显的尾延迟。
- FlexGen 在在线到达下是否因 batch formation / offload 粒度导致较高 TTFT。

---

## 5. 实验组 3: 阶段式资源变化 + 离线推理

### 5.1 目的

在离线推理中人为控制系统资源阶段式变化，验证 RunKV 是否能通过 signed imbalance feedback 适应资源变化。该组分为 IO 带宽变化和 GPU SM 占用变化。这里的关键区别不是 TightLLM 是否每个 step 重算 plan，而是 TightLLM 默认每个 step 重算时仍使用离线 profile 中的固定带宽 / MFU 假设；资源实际变化后，纯 `tightllm-replay` 不会把新的 IO/compute 条件纳入 ILP，除非启用 `tightllm-feedback`。

### 5.2 控制变量时钟

正式实验不能让资源扰动器自己按 wall-clock 独立启动。原因是一次 offline runner 包含模型加载、engine build、权重加载、request add、CUDA warmup 等不可控时间；如果压力进程先独立跑 `0s:0,30s:15,60s:0`，那么真正进入 `engine.step()` 时可能已经落在任意阶段，资源变化这个变量就没有被控制住。

资源变化必须由测试程序在 inference loop 内统一控制。可选两种时钟：

| 时钟 | 阶段边界 | 适用问题 | 优点 | 风险 |
|---|---|---|---|---|
| step-index clock | `engine.step()` 的 step id，例如 `step 0-15 / 16-47 / 48+` | 同样数量的 scheduler step 内，资源受限阶段造成多大性能损失 | 与 RunKV/TightLLM 的控制粒度一致，完全避开加载时间不确定性 | 不同系统每个阶段对应的真实秒数不同，不能直接用作真实时间吞吐结论 |
| inference-time clock | 第一次 `engine.step()` 即将开始时记 `t0`，例如 `t=0-30s / 30-60s / 60s+` | 真实系统在同一时间窗内的吞吐/延迟损失 | 每个系统经历同样真实时间的资源阶段 | 每个系统在 S2 内经历的 step 数不同，planner 行为比较会混入吞吐差异 |

首轮离线对比建议用 **step-index clock 作为主实验**，再用 inference-time clock 做 robustness check。step-index clock 的目的不是比较“几个 step 后把 imbalance 拉回 deadband”。在 S2 资源持续受限时，纯 `tightllm-replay` 不会因为真实 IO/SM 条件变化而主动修正离线 profile，资源一旦恢复又会在下一步自然回到原假设附近；这个“恢复 step 数”指标没有可比性。更合理的因变量是：S2 固定资源受限窗口内，RunKV 的反馈调整是否降低 stage-level latency inflation、throughput deficit 和尾延迟放大；S3 资源恢复后，RunKV 是否避免因 replay budget 调整带来的过量 replay 或反向抖动。

推荐阶段定义：

| stage | step-index clock | inference-time clock | 资源状态 |
|---|---|---|---|
| S1 | `step 0-15` | `t=0-30s` | 正常资源 |
| S2 | `step 16-47` | `t=30-60s` | 资源受限 |
| S3 | `step 48+` | `t=60s+` | 资源恢复 |

执行要求：

- pressure worker 必须在模型加载和 engine build 之前完成 buffer / matrix 预分配，但保持 idle。
- `run_prompts_with_engine()` 完成 `engine.add_request(...)` 后、第一次 `engine.step()` 前，测试程序发出 `experiment_start`。
- step-index clock 下，每次调用 `engine.step()` 前根据当前 step id 设置 pressure target；该 target 作用于本 step。
- inference-time clock 下，以 `experiment_start` 为 `t0`，由测试程序内的 controller thread 按 `t - t0` 切换 pressure target。
- 每个 JSONL step 需要记录 `resource_stage`、`resource_target`、`control_clock`、`stage_start_step` 或 `stage_start_time_s`，否则事后无法严格对齐。

### 5.3 IO 带宽阶段式变化

推荐用受测试程序控制的 CPU↔GPU memcpy pressure worker 模拟共享 PCIe / DMA 路径被抢占。worker 可以是子进程或同进程后台线程，但阶段切换必须由 offline runner 在 inference loop 中下发，不应独立按启动 wall-clock 自行切换。

| pattern | S1 | S2 | S3 |
|---|---:|---:|---:|
| `io_mild` | 0 GB/s | 5 GB/s | 0 GB/s |
| `io_severe` | 0 GB/s | 15 GB/s | 0 GB/s |

pressure worker 要求：

- 可设置 target GB/s。
- 使用 pinned CPU memory。
- 在 `experiment_start` 前完成 buffer 预分配，并保持 idle。
- 支持 runner 通过 IPC / queue / shared state 修改 target。
- 输出每秒实际 memcpy 带宽。
- 每次 target 变化记录 `step_id` 或 `elapsed_s`。
- 若用子进程，runner 必须先发 `start` 信号，再按 step/time 发 `set_target` 信号。

### 5.4 GPU SM 占用阶段式变化

推荐用受测试程序控制的 CUDA spin kernel 或 GEMM pressure worker 模拟同卡共租户占用 SM。和 IO pressure 一样，worker 可以提前启动，但不能提前进入阶段计时；阶段切换必须以 `experiment_start` 为锚点。

| pattern | S1 | S2 | S3 |
|---|---:|---:|---:|
| `sm_mild` | 0% | 25% | 0% |
| `sm_severe` | 0% | 50% | 0% |

pressure worker 要求：

- 尽量主要占用 SM，少占 PCIe 和 HBM。
- 可通过 duty cycle 或 kernel occupancy 控制强度。
- 输出每秒 target/actual pressure。
- 在 `experiment_start` 前完成 GPU matrix / kernel warmup，避免把首次 kernel 编译或首次分配计入 S1。
- 每次 target 变化记录 `step_id` 或 `elapsed_s`。

### 5.5 Settings

| 轴 | 取值 |
|---|---|
| system | `runkv-feedback`, `tightllm-replay`, `tightllm-feedback`, `flexgen-original` |
| resource pattern | `io_mild`, `io_severe`, `sm_mild`, `sm_severe` |
| `num_prompts` | `64` 或 `128` |
| `prompt_words` | `4000, 8000` |
| `max_tokens` | `128` |
| `prefix_blocks` | `1000` |

建议完整矩阵：

```text
4 systems × 4 resource patterns × 2 prompt_lengths × 3 repeats
= 96 runs
```

如果时间紧，先跑：

```text
runkv-feedback / tightllm-replay / flexgen-original
× io_severe / sm_severe
× prompt_words=8000
× 3 repeats
= 18 runs
```

### 5.6 预期图表

- per-second throughput time series，标出 stage 边界
- per-step latency time series
- imbalance time series
- replay budget / replay blocks time series
- stage-level latency CDF
- S2 throughput deficit / latency inflation bar chart
- S3 replay overshoot / latency overshoot check

### 5.7 关键判断

IO 变化场景中，TightLLM 会继续每个 step 基于当前 batch 重新求 ILP budget，但纯离线 profile 的带宽假设会失效，预期表现为：

- S2 进入后 imbalance 明显偏正。
- replay budget 仍会随 batch shape 变化，但不会因真实 H2D 带宽下降而主动改变，或改变不足。
- S2 内 P95 latency 上升、throughput deficit 变大；不要用“imbalance 几步回到 deadband”评价它，因为资源不恢复时纯 TightLLM 没有反馈通道可把真实资源条件写回 ILP，资源恢复时又会自然回到离线 profile 假设附近。

RunKV 预期表现为：

- S2 进入后检测到 `IO_ready - compute_end > 0`。
- 增加 replay budget，用更多 compute 填补 IO window。
- 评价重点是 S2 内 stage-level latency / throughput 损失是否低于 TightLLM，而不是 imbalance 曲线是否回到 0。
- S3 恢复后检查 replay budget 是否及时回落，避免 compute 过量；这是 RunKV 自身控制副作用检查，不作为 RunKV/TightLLM 的主要公平对比指标。

SM 变化场景中，预期 RunKV 会减少 replay budget，避免在 compute 变慢时继续引入过多 recompute。

### 5.8 首轮 pilot: 只对比 RunKV feedback 和 TightLLM

第一轮先不要把 FlexGen、online serving、mild/severe 全矩阵都拉进来。目标是用最少 run 验证“系统资源阶段式变化”这条主结论是否成立。

#### 对比对象

| system | planner 配置 | 说明 |
|---|---|---|
| `runkv-feedback` | `PLANNER=feedback DRY_RUN=0 USE_STATE_MACHINE=1` | RunKV live feedback planner，真正接管 replay plan |
| `tightllm-replay` | `--planner tightllm`，`TIGHTLLM_FEEDBACK_CORRECTION=0` | 每 step 基于当前 batch 重新 ILP 求解，但使用离线 profile 的带宽 / MFU 假设 |

第一轮不启用 `tightllm-feedback`，否则会把“纯 TightLLM 离线模型”与“加了反馈的 TightLLM ablation”混在一起。等主对比跑出差异后，再单独补 `tightllm-feedback`。

#### 工作负载

| 参数 | pilot 值 | 说明 |
|---|---:|---|
| `MODEL` | `/home/lyc/hf_models/opt-2.7b-8k` | 与现有 profile / runner 默认一致 |
| `PREFIX_BLOCKS` | `1000` | 保持当前离线 observation 默认路径 |
| `NUM_PROMPTS` | `128` | 尽量让单次 run 覆盖完整 90s 阶段 |
| `PROMPT_WORDS` | `8000` | 长上下文，放大 KV offload / replay 差异 |
| `MAX_TOKENS` | `128` | 增加 decode 步数，保证 S1/S2/S3 都有足够样本 |
| `GPU_MEMORY_FRACTION` | `0.9` | 固定 KV offload fraction |
| `NUM_DEVICE_BUFFERS` | `3` | 固定 DMA buffer 数 |

如果实际 run 少于 90s，优先把 `MAX_TOKENS` 调到 `256`；如果仍不足，再把 `NUM_PROMPTS` 调到 `256`。如果 OOM，则把 `NUM_PROMPTS` 降到 `64`，但保留 `PROMPT_WORDS=8000` 和较大的 `MAX_TOKENS`。

#### 资源变化 pattern

第一轮只跑两个 severe pattern：

| pattern | pressure 脚本 | 阶段 |
|---|---|---|
| `io_severe_step` | integrated IO pressure worker | `step 0:0GB/s, step 16:15GB/s, step 48:0GB/s` |
| `sm_severe_step` | integrated SM pressure worker | `step 0:0%, step 16:50%, step 48:0%` |

每个 pattern 跑：

```text
2 systems × 3 repeats = 6 runs
```

两个 pattern 共 12 个正式 run。正式 run 前先做 2 个 smoke run：`runkv-feedback` 和 `tightllm-replay` 各跑一次 `io_severe_step`，确认日志字段齐全、run 有足够 step 覆盖 S1/S2/S3。

#### 目录结构

建议每个 run 单独输出到一个目录：

```text
exp_results/staged_offline_pilot/
   io_severe_step/
      runkv_feedback/r0/
      runkv_feedback/r1/
      runkv_feedback/r2/
      tightllm_replay/r0/
      tightllm_replay/r1/
      tightllm_replay/r2/
   sm_severe_step/
      ...
```

每个目录至少包含：

- `manifest.json`
- `pressure.csv`
- `resource_steps_*.jsonl`
- `opt_component_mfu_*_<run_tag>.jsonl`
- `opt_component_mfu_*_<run_tag>.flat.jsonl`，如果已启用 flat 输出
- stdout/stderr log

#### 启动方式

正式 run 应该由 offline runner 自己创建并控制 pressure worker，而不是手动开两个终端。推荐把资源控制接进 `examples/offline_inference/opt_replay_component_mfu.py` 的 `run_prompts_with_engine()`：

1. build engine、add requests 完成。
2. pressure worker 已完成预分配并 idle。
3. 在第一次 `engine.step()` 前调用 `resource_controller.start()`。
4. 每次 `engine.step()` 前调用 `resource_controller.before_step(step)`，按 step-index schedule 设置 target。
5. 每次 `engine.step()` 后记录 step 的 `resource_stage` / `resource_target` / `step_start_time_s` / `step_end_time_s`。
6. inference 结束后调用 `resource_controller.stop()` 并 flush `pressure.csv`。

建议新增 runner 参数：

| 参数 | 示例 | 说明 |
|---|---|---|
| `--resource-pressure-kind` | `none|io|sm` | 资源扰动类型 |
| `--resource-pressure-clock` | `step|time` | 阶段控制时钟；首轮用 `step` |
| `--resource-pressure-pattern` | `0:0,16:15,48:0` | step clock 下为 `step:target`；time clock 下为 `second:target` |
| `--resource-pressure-log-path` | `$OUT/pressure.csv` | pressure worker 日志 |
| `--resource-pressure-buffer-mb` | `256` | IO pressure buffer 大小 |
| `--resource-pressure-direction` | `h2d` | IO pressure 方向 |
| `--resource-pressure-matrix-size` | `4096` | SM pressure GEMM 矩阵大小 |
| `--resource-pressure-step-log-path` | `$OUT/resource_steps.jsonl` | 每个 `engine.step()` 的 stage 对齐 sidecar 日志；未显式设置时随 `--output-dir` 自动生成 |
| `--resource-pressure-max-fraction` | `0.5` | background pressure target 上限；IO 为校准/手动带宽的比例，SM 为 duty-cycle 上限比例 |
| `--resource-pressure-io-calibration-s` | `0.5` | inference 前用同一 copy 路径校准 IO 可达带宽的秒数 |
| `--resource-pressure-io-max-gbps` | 空 | 手动指定 IO 可达带宽，优先于自动校准 |

这个接口已经接入 `examples/offline_inference/opt_replay_component_mfu.py`；`examples/offline_inference/run_opt_feedback_observation.py` 和 `examples/offline_inference/run_tightllm_observation.py` 会把未知 CLI 参数透传给主 runner。`benchmarks/runkv_resource_pressure/io_bandwidth_pressure.py` 和 `benchmarks/runkv_resource_pressure/sm_pressure.py` 仍适合做 standalone calibration：测这台机器上 `15GB/s` 或 `50%` target 是否能产生预期 pressure。

注意：`--resource-pressure-pattern` 里的 target 是 requested target，不再无条件按原值施加。controller 会把它裁剪成 effective target，并同时记录两者。IO 模式下，默认先用 0.5s 测同方向 H2D/D2H copy 的可达带宽，再把 background pressure 限制到 `calibrated_gbps × --resource-pressure-max-fraction`；例如机器实测 H2D 只有 10GB/s，pattern 里写 `16:15`，默认 `max_fraction=0.5` 时实际施加 target 会是约 5GB/s，而不是 15GB/s。SM 模式下 target 是后台 GEMM worker 的 duty-cycle，默认最多 50%。`pressure.csv` 会记录 `requested_target`、effective `target`、`target_cap`、`calibrated_capacity`；`resource_steps.jsonl` 和 MFU JSONL 里会记录 `resource_requested_target`、`resource_target`、`resource_target_cap`、`resource_calibrated_capacity`。

公共变量：

```bash
export MODEL=/home/lyc/hf_models/opt-2.7b-8k
export TIGHTLLM_PROFILE_PATH=tightllm_profile.json
export PREFIX_BLOCKS=1000
export NUM_PROMPTS=128
export PROMPT_WORDS=8000
export MAX_TOKENS=128
export GPU_MEMORY_FRACTION=0.9
export NUM_DEVICE_BUFFERS=3
export ENABLE_OPT_COMPONENT_MFU_PROFILING=1
export ENABLE_NVTX=1
export ENABLE_NSYS=1
export ENABLE_PROFILE=1
export RUNKV_PREHOOK_TIMING=1
```

当前 staged-resource pipeline 默认打开 Nsight Systems / NVTX / CUDA profiler，并保留 `ENABLE_OPT_COMPONENT_MFU_PROFILING=1`。这样一次 run 同时得到两类数据：一类是项目内 CUDA event JSONL，用于 stage-level latency、imbalance、replay budget；另一类是 Nsight sqlite，用于 NVTX prehook timing、CUPTI memcpy/kernel timeline 和 per-layer 多项指标分析。

不开 Nsight 时仍可得到这些轻量数据：每个 step 的 wall time / stage 对齐、per-layer `compute_start_ms_from_anchor`、`forward_start_ms_from_anchor`、`compute_end_ms_from_anchor`、`load_start_ms_from_anchor`、`load_ready_ms_from_anchor`、`kv_ready_ms_from_anchor`、`hs_ready_ms_from_anchor`、`imbalance_ms`、replay ratio / replay tokens、controller budget update，以及 pressure target / actual samples。它们足够支撑 staged-resource 正式对比里的 S2 throughput deficit、P95 latency inflation、S3 overshoot 和 planner 行为统计。

但 NVTX 标记的 CPU timeline 和 CUPTI 明细必须来自 Nsight Systems 导出的 sqlite，例如：`runkv_recompute:pre_hook:*` / `runkv:prehook:imbalance:*` / `runkv:prehook:build_plan:*` / `runkv:prehook:schedule_io:*` 的绝对 start/end、`runkv:layer_compute:L*` 内的 kernel launch 关联、`CUPTI_ACTIVITY_KIND_MEMCPY` 中的真实 H2D memcpy duration / streamId、kernel active time 和 GPU bubble。这些数据没有 `.nsys-rep` 或 sqlite 就拿不到。

`RUNKV_PREHOOK_TIMING=1` 是一个折中诊断模式：它不需要 nsys，会额外写 `prehook_timing_*.jsonl` 和 `prehook_summary_*.txt`，记录每 step 聚合的 `sync_wait_ms`、`imbalance_ms`、`build_plan_ms`、`build_meta_ms`、`skip_ids_ms`、`schedule_io_ms` 以及 schedule_io 子项。但它没有 NVTX 的绝对开始时刻，也不能关联 GPU kernel / memcpy timeline。

如果只做快速 smoke，可以显式关闭 timeline 采集：`--disable-nsys --disable-nvtx --disable-profile --disable-prehook-timing`。这种轻量模式仍能跑 staged-resource summary，但不会生成 sqlite，也不会跑 per-layer timing analysis。

RunKV feedback，IO severe step-clock：

```bash
OUT=exp_results/staged_offline_pilot/io_severe_step/runkv_feedback/r0
mkdir -p "$OUT"
PLANNER=feedback \
DRY_RUN=0 \
USE_STATE_MACHINE=1 \
ENABLE_OPT_COMPONENT_MFU_PROFILING=1 \
ENABLE_NVTX=1 \
ENABLE_NSYS=1 \
ENABLE_PROFILE=1 \
RUNKV_PREHOOK_TIMING=1 \
OUTPUT_DIR="$OUT" \
RUN_TAG=io_severe_step_runkv_r0 \
MANIFEST_FILE="$OUT/manifest.json" \
python /home/lyc/inference/vllm/examples/offline_inference/run_opt_feedback_observation.py \
   --resource-pressure-kind io \
   --resource-pressure-clock step \
   --resource-pressure-pattern 0:0,16:15,48:0 \
   --resource-pressure-direction h2d \
   --resource-pressure-buffer-mb 256 \
   --resource-pressure-step-log-path "$OUT/resource_steps.jsonl" \
   --resource-pressure-log-path "$OUT/pressure.csv" 2>&1 \
   | tee "$OUT/run.log"
```

TightLLM，IO severe step-clock：

```bash
OUT=exp_results/staged_offline_pilot/io_severe_step/tightllm_replay/r0
mkdir -p "$OUT"
TIGHTLLM_FEEDBACK_CORRECTION=0 \
ENABLE_OPT_COMPONENT_MFU_PROFILING=1 \
ENABLE_NVTX=1 \
ENABLE_NSYS=1 \
ENABLE_PROFILE=1 \
RUNKV_PREHOOK_TIMING=1 \
OUTPUT_DIR="$OUT" \
RUN_TAG=io_severe_step_tightllm_r0 \
MANIFEST_FILE="$OUT/manifest.json" \
python /home/lyc/inference/vllm/examples/offline_inference/run_tightllm_observation.py \
   --resource-pressure-kind io \
   --resource-pressure-clock step \
   --resource-pressure-pattern 0:0,16:15,48:0 \
   --resource-pressure-direction h2d \
   --resource-pressure-buffer-mb 256 \
   --resource-pressure-step-log-path "$OUT/resource_steps.jsonl" \
   --resource-pressure-log-path "$OUT/pressure.csv" 2>&1 \
   | tee "$OUT/run.log"
```

SM severe 只替换 pressure 参数：

```bash
--resource-pressure-kind sm \
--resource-pressure-clock step \
--resource-pressure-pattern 0:0,16:50,48:0 \
--resource-pressure-matrix-size 4096 \
--resource-pressure-step-log-path "$OUT/resource_steps.jsonl" \
--resource-pressure-log-path "$OUT/pressure.csv"
```

这些 CLI 由 offline runner 在 `engine.step()` loop 内控制 pressure worker；不要用未同步的双终端方式做正式数据。

更推荐用全自动 pipeline 展开 RunKV / TightLLM 对比、收集产物并调用 stage-level 分析：

```bash
python scripts/run_staged_resource_benchmark.py \
   --resource-pressure-kind io \
   --resource-pressure-clock step \
   --resource-pressure-pattern 0:0,16:15,48:0 \
   --resource-pattern-name io_severe_step \
   --repeats 3 \
   --model /home/lyc/hf_models/opt-2.7b-8k \
   --prefix-blocks 1000 \
   --num-prompts 128 \
   --prompt-words 8000 \
   --max-tokens 128 \
   --tightllm-profile-path tightllm_profile.json
```

SM severe 只替换：

```bash
python scripts/run_staged_resource_benchmark.py \
   --resource-pressure-kind sm \
   --resource-pressure-clock step \
   --resource-pressure-pattern 0:0,16:50,48:0 \
   --resource-pattern-name sm_severe_step \
   --repeats 3 \
   --resource-pressure-matrix-size 4096
```

默认 pipeline 会打开 Nsight / NVTX / CUDA profiler；每个 run 输出 `manifest.json`、`run.log`、`pressure.csv`、`resource_steps.jsonl`、`opt_component_mfu_*.jsonl`、`.flat.jsonl`、`.nsys-rep`、导出的 `.sqlite`，以及 `prehook_timing/prehook_timing_*.jsonl`。stage-level 分析结果写到 `exp_results/analysis/staged_resource/<pattern>/<run_tag>/`，包含 `analysis_summary.md`、`stage_summary.csv`、`comparison_summary.csv` 和 `summary.json`。per-layer timing 分析按 repeat 写到 `exp_results/analysis/staged_resource_per_layer/<pattern>/<run_tag>/r*/`，由 `tools/analyze_per_layer_timing.py` 生成 imbalance、compute-vs-IO、H2D DMA、prehook、layer_compute、kernel timing、replay token 等多项指标图表和 summary。

如果只想跑轻量 stage summary，不采集 Nsight timeline，可追加：

```bash
--disable-nsys --disable-nvtx --disable-profile --disable-prehook-timing
```

如果只想保留 `.nsys-rep` 而不自动导出 sqlite，可追加 `--skip-sqlite-export`；per-layer timing analysis 仍需要 run 目录中已有 `.sqlite`，否则应同时追加 `--skip-per-layer-analysis`。

如果手动运行 wrapper，Nsight / NVTX 相关环境变量如下：

```bash
mkdir -p "$OUT/prehook_timing"
export RUNKV_PREHOOK_TIMING=1
export RUNKV_PREHOOK_TIMING_DIR="$OUT/prehook_timing"
export ENABLE_NSYS=1
export ENABLE_NVTX=1
export ENABLE_PROFILE=1
export NSYS_SAMPLE=cpu
export NSYS_EXTRA_ARGS="--capture-range=cudaProfilerApi --capture-range-end=stop"
```

#### 第一轮只看这些结果

先不急着画复杂图，先检查 5 个量：

1. run 是否覆盖 S1/S2/S3 三段。
2. S2 期间 `imbalance_ms` 是否被资源扰动显著推离 S1 分布，用来证明扰动确实生效。
3. RunKV 的 `controller_update.budget_after` / replay blocks 是否在 S2 按预期调整；这是机制解释，不是最终胜负指标。
4. TightLLM 的 ILP budget 是否主要随 batch shape 变化，而不是随 pressure 阶段变化；这用于验证对比对象确实是纯离线 profile planner。
5. 每个 stage 的 step latency / throughput：主比较看 S2 的 throughput deficit、P95 latency inflation 和尾延迟放大；S3 只检查 RunKV 是否有 replay overshoot 或恢复后抖动。

额外必须检查 stage 对齐：

- `pressure.csv` 的第一次非零 target 必须出现在 `step_id=16` 前后，而不是模型加载期间。
- JSONL 中 `resource_stage` 的 S2 起点必须和 `pressure.csv` 中的 target 切换一致。
- 如果某个 run 在 `step < 48` 就结束，该 run 作废，需要增加 `MAX_TOKENS` 或 `NUM_PROMPTS`。

如果这 12 个 run 里 IO severe 已经能拉开差距，就优先围绕 IO severe 扩 repeats 和画图；SM severe 可以作为补充验证 compute-side disturbance。

---

## 6. 实验组 4: 阶段式资源变化 + 在线推理

### 6.1 目的

把实验组 3 的资源阶段式变化放到在线 serving 中，重点观察 SLO、尾延迟、资源受限阶段的服务能力下降，以及资源恢复后队列和 SLO violation 的服务级回落，而不是只看平均吞吐或 planner imbalance 是否归零。

### 6.2 Settings

先从实验组 2 找到每个系统在稳态下的 knee point request rate：

```text
knee = 使 TTFT P95 或 TPOT P95 开始明显上升的 request_rate
```

然后选两个在线负载点：

| 负载点 | 取值 |
|---|---|
| below-knee | `0.8 × knee` |
| near-knee | `1.0 × knee` |

资源 pattern 沿用实验组 3：

| 轴 | 取值 |
|---|---|
| system | `runkv-feedback`, `tightllm-replay`, `tightllm-feedback`, `flexgen-original` |
| resource pattern | `io_severe`, `sm_severe` |
| request_rate | `below-knee`, `near-knee` |
| dataset | ShareGPT + long-prompt synthetic |
| duration | 至少 120s，保证覆盖 `30s normal + 30s disturbed + 60s recovery` |

### 6.3 在线阶段定义

| stage | 时间窗口 | 行为 |
|---|---|---|
| S1 | `0-30s` | 稳态请求到达，无资源干扰 |
| S2 | `30-60s` | 请求到达不变，开启 IO 或 SM 干扰 |
| S3 | `60-120s` | 请求到达不变，资源恢复 |

### 6.4 预期图表

- 每秒 completed requests
- 每秒 TTFT P95 / TPOT P95
- 每秒 SLO violation rate
- queue length / waiting time
- per-layer imbalance 按时间桶聚合
- replay budget 按时间桶聚合

### 6.5 SLO 建议

根据模型和硬件实际表现设定 SLO。初始建议：

| 指标 | SLO |
|---|---|
| TTFT P95 | 小于稳态 RunKV P95 的 `1.5×` |
| TPOT P95 | 小于稳态 RunKV P95 的 `1.5×` |
| E2E P95 | 小于稳态 RunKV P95 的 `1.5×` |

如果绝对值更适合报告，可在 smoke run 后固定，例如：

```text
TTFT P95 < 2s
TPOT P95 < 100ms/token
```

### 6.6 关键判断

- RunKV 在 S2 期间的 SLO violation 是否更低。
- 资源恢复后 RunKV 的 violation rate 是否更快回落。
- TightLLM + feedback 是否接近 RunKV；如果仍有差距，需要用 per-layer timing 解释差异来自反馈粒度、执行路径或 profile mismatch。

---

## 7. 实验组 5: 真实场景 Case Study

### 7.1 目的

构造一个接近真实服务的阶段式负载变化场景，用少量但高质量的 run 展示系统行为，而不是只做矩阵扫描。

建议做两个 case，主文放一个，附录放一个。

### 7.2 Case A: Coding agent burst

场景：IDE / coding agent 服务平时低负载，用户集中触发长上下文请求时出现 burst，同时机器上有轻度 IO 背景任务。

Workload：

| 阶段 | 时间窗口 | 请求到达 | prompt | output |
|---|---|---:|---:|---:|
| S1 | `0-60s` | 2 req/s | 4000-8000 words | 64 tokens |
| S2 | `60-90s` | 20 req/s | 4000-8000 words | 64 tokens |
| S3 | `90-180s` | 2 req/s | 4000-8000 words | 64 tokens |

资源：

| 阶段 | IO pressure |
|---|---:|
| S1 | 0 GB/s |
| S2 | 5 GB/s |
| S3 | 0 GB/s |

对比系统：

```text
runkv-feedback
vs tightllm-replay
vs flexgen-original
```

可选加入：

```text
tightllm-feedback
```

主图：

- request_rate / completed_requests time series
- TTFT P95 / TPOT P95 time series
- replay budget time series
- imbalance time series
- SLO violation rate time series

### 7.3 Case B: Long-context QA 切到 short-chat

场景：同一服务从长文档 QA 切到短对话，资源不变但 workload 的 compute/IO ratio 阶段式变化。

Workload：

| 阶段 | 请求类型 | prompt | output | 到达率 |
|---|---|---:|---:|---:|
| S1 | long-context QA | 8000 words | 32 tokens | 4 req/s |
| S2 | short chat | 512 words | 512 tokens | 4 req/s |
| S3 | long-context QA | 8000 words | 32 tokens | 4 req/s |

资源：

```text
steady，无外部 IO/SM 干扰
```

对比重点：

- workload 从 KV/IO-bound 切到 decode/compute-bound 后，RunKV 是否能降低 replay budget。
- TightLLM 的离线 profile 是否需要按 workload 预先分 regime；如果没有 regime，是否出现过量 replay。
- FlexGen 是否在长短请求切换时出现 batching 或 offload schedule 的尾部放大。

### 7.4 Case study 报告格式

每个 case 输出一页 summary：

1. workload timeline。
2. system-level latency/throughput timeline。
3. planner-level replay budget/imbalance timeline。
4. 三条观察：进入扰动、扰动期间、恢复阶段。
5. 一个结论：RunKV 的反馈控制在哪里带来收益，哪里仍有不足。

---

## 8. 扰动器设计

### 8.1 IO pressure 进程

推荐实现为 `ResourcePressureController` + worker。worker 可以复用独立脚本入口做 calibration，但正式实验必须由 runner 在 inference loop 中控制其 `start/set_target/stop`。

建议文件：

```text
benchmarks/runkv_resource_pressure/controller.py
benchmarks/runkv_resource_pressure/io_bandwidth_pressure.py
```

参数：

| 参数 | 说明 |
|---|---|
| `--device cuda:0` | 使用的 GPU |
| `--buffer-mb` | pinned CPU buffer 和 GPU buffer 大小 |
| `--pattern 0:0,16:15,48:0` | step/time schedule，由 runner 解释 |
| `--clock step|time` | 阶段控制时钟 |
| `--direction h2d|d2h|bidirectional` | 压力方向 |
| `--log-path` | 每秒实际带宽日志 |

实现要求：

- 使用 pinned host memory。
- 使用 `cudaMemcpyAsync` 或 PyTorch non_blocking copy。
- 用 duty cycle 控制目标带宽。
- `experiment_start` 前只完成预分配和 warmup，不进入阶段计时。
- step-clock 模式下，只有 runner 调用 `set_target(step_id)` 才允许切换 target。
- 每秒记录 actual GB/s。
- 日志写入 `step_id`、`elapsed_s`、`stage_id`、`target_gbps`、`actual_gbps`。

### 8.2 SM pressure 进程

推荐实现为同一个 `ResourcePressureController` 下的 SM worker。独立脚本仍可用于校准 duty cycle 和矩阵规模，正式实验由 runner 控制阶段切换。

建议文件：

```text
benchmarks/runkv_resource_pressure/controller.py
benchmarks/runkv_resource_pressure/sm_pressure.py
```

参数：

| 参数 | 说明 |
|---|---|
| `--device cuda:0` | 使用的 GPU |
| `--pattern 0:0,16:50,48:0` | step/time schedule，由 runner 解释 |
| `--clock step|time` | 阶段控制时钟 |
| `--kernel spin|gemm` | 压力 kernel 类型 |
| `--log-path` | 每秒实际 duty cycle 日志 |

实现要求：

- 尽量避免大规模 H2D/D2H。
- 通过 duty cycle 控制压力。
- 如果使用 GEMM，需要固定矩阵常驻 GPU，避免引入 PCIe pressure。
- `experiment_start` 前完成 matrix 分配和 warmup。
- step-clock 模式下，只有 runner 调用 `set_target(step_id)` 才允许切换 target。
- 每秒记录 `step_id`、`elapsed_s`、`stage_id`、target pressure 和 observed kernel time。

---

## 9. 推荐执行顺序

按风险从低到高推进：

1. 组 1 smoke subset：验证三方 baseline 都能跑通，结果 schema 能统一。
2. 组 1 完整矩阵：建立稳态离线基线。
3. 组 2 smoke subset：验证 online runner、request trace 和 TTFT/TPOT 统计。
4. 组 2 完整矩阵：找到 knee point。
5. 组 3 IO severe：先验证最能体现 RunKV feedback 的场景。
6. 组 3 SM severe：验证 compute-side 资源变化。
7. 组 4 below-knee / near-knee：在线阶段式变化。
8. 组 5 case study：从前面结果中挑最能解释系统差异的设置。

---

## 10. 最小可执行版本

如果只想先产出一版可用结果，建议先跑以下 24 个 run：

| 组 | systems | settings | repeats |
|---|---|---|---:|
| steady offline | RunKV / TightLLM / FlexGen | `num_prompts=32`, `prompt_words=4000`, `max_tokens=32` | 3 |
| steady offline long | RunKV / TightLLM / FlexGen | `num_prompts=64`, `prompt_words=8000`, `max_tokens=32` | 3 |
| staged offline IO | RunKV / TightLLM / FlexGen | `io_severe`, `num_prompts=128`, `prompt_words=8000`, `max_tokens=128` | 3 |
| staged offline SM | RunKV / TightLLM / FlexGen | `sm_severe`, `num_prompts=128`, `prompt_words=8000`, `max_tokens=128` | 3 |

这组最小结果能回答：

- 稳态下 RunKV 是否有竞争力。
- IO 阶段变化下 RunKV 是否比 TightLLM/FlexGen 更稳。
- SM 阶段变化下 RunKV 是否避免过量 replay。

在线组和 case study 在最小结果之后补充。

---

## 11. 后续需要补的工程项

为了把上面的方案完全自动化，建议新增：

1. `benchmarks/runkv_compare/run_experiment.py`
   - 统一展开 system × workload × resource pattern 矩阵。
   - 负责启动 offline runner 或 online server/client。
   - 写 manifest。

2. `benchmarks/runkv_resource_pressure/controller.py`
   - 提供 `ResourcePressureController(kind, clock, pattern, log_path)`。
   - 在 `run_prompts_with_engine()` 内被调用：`prepare()`、`start()`、`before_step(step)`、`after_step(step)`、`stop()`。
   - 支持 step-index clock 和 inference-time clock。
   - 负责把 `resource_stage` / `resource_target` 回写到 profiler JSONL。

3. `benchmarks/runkv_compare/normalize_results.py`
   - 把 RunKV/TightLLM JSONL、benchmark_serving 输出、FlexGen 输出归一成同一 schema。

4. `benchmarks/runkv_compare/plot_results.py`
   - 生成 throughput-latency 曲线、stage time series、S2 损失和 S3 overshoot / 服务级回落图。

5. `benchmarks/runkv_resource_pressure/io_bandwidth_pressure.py`
   - IO pressure worker / calibration entrypoint。

6. `benchmarks/runkv_resource_pressure/sm_pressure.py`
   - GPU SM pressure worker / calibration entrypoint。

---

## 12. 结果解释优先级

分析结果时优先按下面顺序定位差异：

1. 先看 stage-level throughput / latency，确认系统层收益。
2. 再看 TTFT / TPOT / E2E，区分 prefill、decode 和排队。
3. 再看 per-layer imbalance，确认瓶颈来自 IO 还是 compute。
4. 再看 replay budget / replay blocks，确认 planner 是否按预期响应。
5. 最后看 nsys NVTX 和 kernel timeline，解释 H2D/D2H/attention/FFN 的具体重叠情况。

这样可以避免只用平均吞吐解释所有现象，也能把 RunKV 的 feedback 机制和最终收益直接对应起来。
