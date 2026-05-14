# RunKV Weight Offload — Layer-wise Weight H2D 接入设计

> 状态: 设计草案
> 日期: 2026-05-03
> 适用范围: `runkv` 分支、`prev_layer_output_dynamic` 模式、OPT、单机单卡、单 KV-group

---

## 1. 动机与定位

RunKV 当前已经把 **per-layer KV cache** 从 GPU 搬到 CPU pinned memory，并通过
`load_stream` 双流流水线把每层 KV 的 H2D DMA 与 compute overlap（详见
[runkv_overview.md](runkv_overview.md) §3、[layer_dynamic_replay_design.md](layer_dynamic_replay_design.md)）。
这一机制让 KV 容量不再是 GPU 显存的硬约束。

随着模型尺寸进一步增长，**weight 也开始撑爆显存**——典型场景：

- 模型 weight + activation + 少量 KV 已超过单卡可用显存；
- 不希望走 ZeRO-Infinity 这类需要训练框架介入的方案；
- 仍要保留 RunKV 的 per-layer pipeline 语义（IO/compute overlap、feedback 控制）。

本方案在 RunKV 现有的 layer-wise H2D 框架内增加一类 IO：**per-layer weight
prefetch**。语义与 KV offload 完全平行——layer L 的权重在 `pre_hook(L-1)` 阶段
通过 `load_stream` 拉到一个 GPU ring buffer，在 layer L compute 启动前同步 ready，
compute 结束后该 buffer slot 被 layer L+N 复用。

> 非目标：本方案**不**讨论 partial offload（"部分层常驻 + 部分层 offload"）、
> 量化 weight、训练时 weight 的反向。第一版只覆盖纯 inference、单一 dtype、
> 全部 decoder 层均 offload 的最朴素配置。

---

## 2. 与现有模块的关系

### 2.1 复用 RunKV 既有抽象

| RunKV 既有抽象 | 是否复用 | 备注 |
|---|---|---|
| `load_stream` (CUDA stream) | 复用 | 与 KV DMA 共享一条 stream，保持原有 stream 拓扑 |
| `_runkv_pre_hook` | 复用 | 在 hook 内增加 `weight_ready[L].wait()` + `launch_weight_io(L+1)` |
| Per-layer event 记录 (`load_start_event` / `load_ready_event` 命名空间) | 平行新增 | `weight_load_start_event[L]` / `weight_ready_event[L]` |
| Ring buffer `buffer_idx = layer_idx % num_buffers` | 平行新增 | 独立的 weight ring buffer，N≥2 |
| Pinned CPU cache (`cpu_caches_per_layer`) | 平行新增 | `cpu_weights_per_layer`，初始化阶段一次性建好 |
| Speculative plan builder | **不复用** | weight DMA 不依赖 plan，无需进 builder pipeline |
| `LayerReplayPlan` / `skip_block_ids` | **不复用** | weight 没有 logical id / mapping / skip 语义 |
| `FeedbackReplayPlanProvider` / `ImbalanceController` | 间接耦合 | 见 §6 |

### 2.2 与 `prev_layer_output_dynamic` 的兼容性

[layer_dynamic_replay_design.md](layer_dynamic_replay_design.md) 已经强制要求：

- 仅 OPT、单机单卡、单 KV-group；
- 禁用 cudagraph、DCP、TP、PP、cascade attention、ubatching。

weight offload 的限制集合是它的**子集**——本方案不放宽其中任何一项。第一版同样
强制 `eager`，并在 OPT decoder layer 的 `forward` 之前注入 `wait_weight_ready(L)`
同步点。

---

## 3. 数据与控制流

### 3.1 模块分层（在 RunKV 既有图上增量）

```
┌─────────────── GpuModelRunner ─────────────────────────────┐
│  _prepare_layer_recompute_step_metadata()                  │
│  _prepare_dynamic_replay_runtime()                         │
│  _prepare_weight_offload_runtime()         ← 新增          │
│  _runkv_pre_hook()                                         │
│    ├─ (existing) KV sync / imbalance / plan / launch KV IO │
│    └─ (new)     weight sync(L) / launch weight IO(L+1)     │
└─────────────────────────────────┬──────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
┌───────▼───────────┐    ┌────────▼─────────┐    ┌──────────▼──────────┐
│ LayerRecompute    │    │ OPTDynamicReplay │    │ WeightOffloadRuntime│
│   Manager (KV)    │    │   Runtime (KV)   │    │     (新增)          │
│                   │    │                  │    │  - cpu_weights      │
│                   │    │                  │    │  - gpu_buffers (N)  │
│                   │    │                  │    │  - load_start_evt   │
│                   │    │                  │    │  - ready_evt        │
│                   │    │                  │    │  - layer→param map  │
└───────────────────┘    └──────────────────┘    └─────────────────────┘
```

`WeightOffloadRuntime` 与 `LayerRecomputeManager` (KV mapper) 是**对等关系**：都
是 `load_stream` 上的生产者，都在 `_runkv_pre_hook` 中被同步与调度。

### 3.2 Step 边界

1. `__init__`（worker 启动一次性）：
   1. 把每个 decoder 层的 weight `Parameter` 拷到 host pinned tensor，组织为
      `cpu_weights_per_layer: dict[int, list[torch.Tensor]]`（按 param 顺序）；
   2. 在 GPU 上分配 N 槽 ring buffer，每槽容纳一个 layer 的全部权重（contiguous
      arena，见 §4.2）；
   3. 把每层 `nn.Module` 的 `Parameter` 重定向到 ring buffer 中本层对应 slot 的
      view（`module._parameters[name] = buffer_view`），以避免 forward 中 import 真实
      参数；
   4. 标记 layer 0 的 weight 槽为 "需 step 启动前同步 prefetch"。

2. `execute_model()` 进入：
   1. **Step 启动同步 prefetch**：layer 0 的 weight H2D 在第一次 forward kernel
      launch 前完成（同 KV 的 `cpu_fill_h2d(0)`）。
   2. layer 0 的 `pre_hook(-1)` 等价位置：launch weight IO(1)（与 KV IO(1) 同
      stream，按 issue 顺序串行）。
   3. 后续 layer 边界处理见 §3.3。

3. step 结束：weight ring buffer 槽自然被下一 step 的 layer 复用——**weight 是
   每 step 全量重传**，这是本方案的固有代价（见 §5）。

### 3.3 Layer 边界（pre_hook 内事件顺序）

在 [runkv_overview.md §3.2](runkv_overview.md) 的事件序列基础上增量：

```
assembly(L) → LN(L) → QKV(L) ── pre_hook(L) ── KVwrite → FlashAttn(L) → O → LN → FFN → ...
                              │
                              ├─ record qkv_end_event(L)
                              │
                              ├─ KV side  (existing):
                              │    ├─ sync_load_layer(L)
                              │    ├─ sync_cpu_fill_h2d(L)
                              │    ├─ qkv_end_event.sync()
                              │    ├─ load_ready_event(L).sync()
                              │    ├─ imbalance = IO_ready(L) − qkv_end(L)
                              │    └─ observe_layer_feedback(L, imbalance)
                              │
                              ├─ Weight side (NEW):
                              │    ├─ weight_ready_event(L+1) is needed only at
                              │    │   pre_hook(L+1); here we **launch** L+1's
                              │    │   weight DMA, not wait for it.
                              │    │
                              │    │  对当前层 L 的 weight：QKV(L) 已经在 compute
                              │    │  stream 上跑过，说明 weight(L) 早已 ready；
                              │    │  无需在此再 sync。
                              │    │
                              │    └─ (no extra CPU sync here)
                              │
                              ├─ pre_hook 二分支 (existing): build_stable_successor
                              │   或 pop_speculative
                              │
                              ├─ runtime.set_layer_plan/metadata/skip_ids(L+1)
                              │
                              ├─ launch_KV_IO(L+1)        (existing, non-blocking)
                              └─ launch_weight_IO(L+1)    (NEW, non-blocking)
```

**对应的 weight wait 注入点**——发生在 layer L+1 的 `forward` 入口、QKV 之前：

```
OPTDecoderLayer.forward(L+1):
    weight_offload_runtime.sync_weight_layer(L+1)   ← 新增
        └─ default_stream.wait(weight_ready_event[L+1])
    [assembly + LN + QKV + pre_hook + ...]
```

> 注：理论上也可以把 `weight_ready_event[L+1].wait` 塞进 `pre_hook(L)` 的最末尾
> （`launch_KV_IO(L+1)` 之后），让 stream 等待先发出去；但这样会让 launch_KV_IO
> 也排在 weight ready 之后串行，反而更慢。**正确做法**是 weight wait 由 layer
> L+1 的 forward 自己负责，pre_hook(L) 只负责 `launch_weight_IO(L+1)`。

### 3.4 Stream 与依赖关系

```
load_stream:    [KV(L)──][KV(L+1)─][W(L+1)──][KV(L+2)─][W(L+2)──][KV(L+3)─][W(L+3)──]
                  │         │         │          │         │          │         │
                  ├─load_ready_KV(L)  ├─weight_ready(L+1)             ...
                  ↓         ↓         ↓          ↓         ↓
default_stream: ...QKV(L)─⏐sync_KV(L)─Attn(L)─O─LN─FFN─⏐sync_W(L+1)─assembly(L+1)─...
```

关键不变量：

- `load_stream` 上 IO 按 issue 顺序串行（H2D queue 性质）；
- `weight(L+1)` 在 `pre_hook(L)` 中 issue，因此**总是排在 `KV(L+1)` 之后**；
- compute stream 只在 layer L+1 自身入口处 wait `weight_ready(L+1)`，KV 仍按
  既有 `pre_hook(L+1)` 内的 `sync_load_layer(L+1)` 同步——两者解耦。

---

## 4. 实现细节

### 4.1 `WeightOffloadRuntime` 接口

```python
class WeightOffloadRuntime:
    def __init__(
        self,
        model: nn.Module,
        layer_indices: list[int],
        load_stream: torch.cuda.Stream,
        num_buffers: int = 2,
        device: torch.device,
    ): ...

    # 一次性：把每层权重物化到 pinned CPU tensor，并把 module 参数重定向到
    # ring buffer view。
    def initialize(self) -> None: ...

    # 主线程：从 _runkv_pre_hook 调用，issue 一次 H2D DMA，non-blocking。
    def launch_weight_io(self, layer_idx: int) -> None: ...

    # 主线程：从 OPTDecoderLayer.forward 入口调用，让 default stream 等
    # weight_ready_event[layer_idx]。
    def sync_weight_layer(self, layer_idx: int) -> None: ...

    # Step 启动：layer 0 的 weight 必须在第一次 compute kernel launch 前 ready。
    def prefetch_layer0_blocking(self) -> None: ...

    # Observability: 暴露 per-layer load_start / ready event 给外部
    # collector 写 JSONL（与 KV 端 schema 对齐）。
    def get_layer_events(self, layer_idx: int) -> WeightLayerEvents: ...
```

实现上 `launch_weight_io` 与 KV 的 `load_layer_async`
（[gpu_model_runner.py L722](/home/lyc/inference/vllm/vllm/v1/worker/gpu_model_runner.py#L722)）
模板一致，但**简化**如下：

- 无 `mapping` / `skip_block_ids`（weight 是固定全量）；
- 无 `_sync_buffer_reuse` 回写等待（weight 只读，无 D2H 路径）；
- 无 `block distribution capture` 调试路径（weight 不参与 RunKV 的 block 调试）；
- DMA 模板：单次 `dst.copy_(src, non_blocking=True)`，src 为整层连续 pinned arena。

### 4.2 Pinned arena 与 ring buffer 布局

为了把每层若干个 `Parameter`（OPT 层约 8 个：q/k/v/o + ffn_up/ffn_down + 2×LN）合
并成一次 DMA：

1. 初始化时，按 dtype/device 计算每层 weight 总字节数 `nbytes_per_layer`；
2. 在 host 上分配单块 pinned `uint8 arena_cpu[L * nbytes_per_layer]`，把每个
   `Parameter` 的字节连续拷入；
3. 在 GPU 上分配 `arena_gpu[N * nbytes_per_layer]`（N=2 双 buffer）；
4. 每个 `Parameter` 在 GPU 上的"实际地址"是 `arena_gpu` 内的某个 view（按 dtype/
   shape `view`），其位置随 layer L 的 `buffer_idx = L % N` 变化；
5. **重定向**：用 `module._parameters[name] = buffer_view` 替换原有 `nn.Parameter`，
   这样 forward 看到的 weight 永远是 ring buffer 当前 slot 中的 view。

> 风险点：`module._parameters[name]` 必须是 `nn.Parameter` 实例，而 view 对应
> `requires_grad=False` 的 leaf tensor。需要包成 `nn.Parameter(view,
> requires_grad=False)`，并验证 OPT 各层 `forward` 不会访问 `Parameter.data` 之外的
> 属性（如 `Parameter.grad`）。OPT 推理路径已确认只用 `weight` / `bias` 字段，OK。

### 4.3 Layer 0 冷启动

第一个 step 的 layer 0 没有"上一层 pre_hook"来 issue weight IO(0)。处理方式：

- 在 `_prepare_dynamic_replay_runtime()` 之后立即调用
  `weight_runtime.prefetch_layer0_blocking()`：在 `load_stream` 上 issue weight
  IO(0)，并让 default stream `wait` 该 event。
- 与已有的 `cpu_fill_h2d(0)` / `load_layer_async(0)` 的处理对称，不增加新的同步
  点种类。

### 4.4 Pre_hook 改动总结

`_runkv_pre_hook(layer_idx=L)` 在现有结构（[runkv_overview.md §3.2](runkv_overview.md)
的"pre_hook 二分支"块）末尾追加一行：

```python
# existing
launch_KV_IO(L + 1)
# NEW
weight_runtime.launch_weight_io(L + 1)
```

`OPTDecoderLayer.forward` 入口追加：

```python
weight_runtime.sync_weight_layer(layer_idx)
```

不引入新的 CPU 阻塞 sync——`sync_weight_layer` 只是 `stream.wait_event`，CPU 不阻塞。

---

## 5. 性能预算

PCIe Gen4 ×16 实测有效带宽 ~25–28 GB/s。以 OPT-2.7B（hidden=2560、ffn=10240、
L=32、fp16）为例：

| 量 | 估算 |
|---|---:|
| Per-layer weight 字节 | ≈ (4·H² + 2·H·F + LN/bias) · 2B ≈ **157 MB** |
| 单层 weight DMA | **~5.6 ms** |
| 当前 KV IO pacing（[runkv_overview §8.3](runkv_overview.md)）| 36.7 ms/layer |
| 当前 compute_dur | 36.6 ms/layer |
| 加 weight 后 IO pacing 估算 | **~42.3 ms/layer** |
| 加 weight 后 max(compute, IO) | **42.3 ms/layer**（IO-bound） |
| Σ step 估算 | 1133 ms → **~1300 ms（+15 %）** |

**结论**：

1. 若 weight 能完全装进显存，weight offload 是**确定的负收益**（~+15 %）。
2. 若 weight 装不下（本方案的真正适用场景），对比项不是"常驻 GPU"，而是
   "OOM / 单层 swap 但无 KV 流水线"。此时接进 RunKV 仍是最佳实现。
3. Per-step 全量重传的代价不可避免——除非 N→L（全驻），否则 ring buffer 只能在
   step 内复用，跨 step 不能省。

> 第一版的验证目标不是"和常驻 GPU 比 wall-clock"，而是 **"在 weight 装不下的
> workload 上让 RunKV 仍然能跑，且 compute 仍能被 IO 隐藏到 max(compute, IO)
> 的下界"**。

---

## 6. 与 Feedback 控制器的耦合

`ImbalanceController`（[imbalance_state_machine_controller.md](imbalance_state_machine_controller.md)）
观测的是 `IO_ready(L) − qkv_end(L)`。weight DMA 接进来后：

- `IO_ready(L)` 包含 `weight(L)` 的 H2D 完成时间——但 weight wait 是在 layer L
  自己 forward 入口、不在 pre_hook(L) 内观测，所以**控制器观测的 imbalance 不直
  接包含 weight DMA 的延迟**。
- 真正的耦合发生在 `load_stream` queue 上：weight(L+1) 排在 KV(L+1) 之后，**会推
  迟 KV(L+1) 的 ready 时间**——这才是控制器观测得到的部分。
- 系统效应：IO 侧整体被抬高一段常数（~weight DMA 单层耗时），控制器看到的
  imbalance 系统性偏负 → STEADY 的 deadband 自然吸收，**不会引发振荡**。

需要做的工程动作：

1. 用 [tools/profiler/stdev_check.py](/home/lyc/inference/vllm/tools/profiler/stdev_check.py)
   重新校准 `σ_baseline`：weight DMA 引入的抖动（pinned alloc 本身较稳，但 PCIe
   queue 抖动会变大）会改变 baseline 噪声尺度。
2. STEADY 的 deadband 若需要扩大，沿用现有 `ImbalanceController` 的 config 字段
   即可（不新增控制维度）。
3. **不**让 controller 直接调节 weight offload 行为（例如 "哪些层不 offload"）—
   这是另一类设计，超出第一版"和 KV offload 平行"的范围（参见 §10 后续工作）。

`FeedbackReplayPlanProvider` 不需要任何接口改动。

---

## 7. 配置

在 `RunKVOffloadConfig`（[vllm/v1/core/kv_cache_offload_config.py](/home/lyc/inference/vllm/vllm/v1/core/kv_cache_offload_config.py)）
新增字段：

| 字段 | 默认 | 作用 |
|---|---|---|
| `enable_weight_offload` | False | weight offload 总开关 |
| `weight_offload_num_buffers` | 2 | GPU ring buffer 槽数 |
| `weight_offload_dtype` | None | None=继承 model dtype；预留给后续量化场景 |

CLI 暴露：

- `--runkv-enable-weight-offload`
- `--runkv-weight-offload-num-buffers`

启用约束（与 `prev_layer_output_dynamic` 相同子集，启动时校验，不 silent fallback）：

- `enable_layer_recompute=True` 且 `layer_recompute_mode="prev_layer_output_dynamic"`；
- 模型为 OPT；
- 单机单卡、单 KV-group；
- 禁用 cudagraph、DCP、TP、PP、cascade attention、ubatching；
- `weight_offload_num_buffers ≥ 2`。

---

## 8. 可观测性

复用 RunKV 现有 JSONL / NVTX 框架：

- **CUDA events**（每层）：
  - `weight_load_start_event[L]`：weight DMA 在 load_stream 上 issue 的时间；
  - `weight_ready_event[L]`：weight DMA 完成时间；
  - `weight_wait_done[L]`：default stream 完成 wait 的时间（用于 cross-check）。
- **NVTX ranges**：
  - `runkv:prehook:launch_weight_io:L*`
  - `runkv:weight_dma:L*`
  - `runkv:weight_wait:L*`（在 forward 入口）
- **JSONL 字段**（与现有 per-layer summary 对齐，新增列）：
  - `weight_dma_ms`（= ready − load_start）
  - `weight_pacing_ms`（连续两层 weight load_start 间隔，用于度量 weight IO 串
    行等效耗时，对应 [runkv_overview §8.3](runkv_overview.md) 的 IO pacing 概念）
  - `weight_wait_ms`（forward 入口 wait 的 CPU 阻塞时长，正常应≈0）
- **离线工具**：[tools/analyze_per_layer_timing.py](/home/lyc/inference/vllm/tools/analyze_per_layer_timing.py)
  扩展支持 weight 字段（添加新列、不破坏既有列）。

---

## 9. 测试与验收

### 9.1 正确性

1. **数值一致性测试**：在 OPT-2.7B 上对比 `enable_weight_offload=True`
   vs `False`，per-token logits 必须 bit-identical（fp16 同一 kernel 路径，仅
   weight 物理地址不同）。
2. **跨 step 正确性**：连续 ≥ 100 step 推理，验证 ring buffer 复用没有把"还在
   被 compute 使用的 weight slot"覆盖（compute stream 已自然顺序保证，但需要
   显式测试用例）。
3. **异常分支**：
   - layer 0 冷启动同步 prefetch；
   - step 中途 batch 变化（不影响 weight DMA，但需确认 `_runkv_pre_hook`
     的 launch 顺序仍正确）；
   - speculative plan rebuild 路径下，weight launch 顺序不被打乱。

### 9.2 性能验收

参照 [runkv_overview §8](runkv_overview.md) 的快照对比格式：

| 指标 | 目标 |
|---|---|
| 数值正确性 | bit-identical |
| Σ step wall-clock | 不超过 §5 的预算上限（~+15 %） |
| `\|imbalance\|` mean | 与 weight-off 同 step 比，不应增加超过 1 ms |
| controller 状态分布 | TRANSIT 触发率不超过 weight-off 基线的 2× |
| weight ring buffer 显存 | 实测 = `N × nbytes_per_layer ± 5 %` |

新增产物目录：`exp_results/analysis/per_layer/weight_offload_<date>/`。

---

## 10. 当前状态与后续工作

第一版（本设计）只覆盖：

- 全量层 offload；
- 单 dtype；
- N=2 ring buffer；
- 不进 controller。

后续可能的演进（**不在本方案范围内**）：

- **Partial weight offload**：按层显存压力 / 重要性选择性 offload；与 controller
  耦合（"调 offload 比例" 作为新控制维度）。
- **Weight + KV 共享带宽预算**：在 controller 中显式建模 `load_stream` 总带宽，
  联合调节 KV replay budget 与 weight offload 集合。
- **量化 weight 路径**：pinned cache 存 packed 形式，GPU 端做 dequant；和
  RunKV 现有 fp16 路径会分叉，需要单独设计。
- **跨 step weight 复用**：当 buffer 槽数 N 接近 L 时退化到 "全驻 + 渐进 swap"，
  和现在的"每 step 全量重传"是连续光谱，可作为第二阶段优化。

---

## 11. 参考

- [runkv_overview.md](runkv_overview.md) — RunKV 总体设计
- [layer_dynamic_replay_design.md](layer_dynamic_replay_design.md) — `prev_layer_output_dynamic` 模式
- [feedback_driven_replay_planner_design.md](feedback_driven_replay_planner_design.md) — feedback 控制框架
- [imbalance_state_machine_controller.md](imbalance_state_machine_controller.md) — 三态控制器
- [deferred_speculative_plan_building.md](deferred_speculative_plan_building.md) — pre_hook 关键路径设计
- 关键源文件（实现锚点）：
  - [vllm/v1/worker/gpu_model_runner.py](/home/lyc/inference/vllm/vllm/v1/worker/gpu_model_runner.py) `load_layer_async` / `sync_load_layer` / `_runkv_pre_hook`
  - [vllm/v1/worker/layer_recompute.py](/home/lyc/inference/vllm/vllm/v1/worker/layer_recompute.py)
  - [vllm/model_executor/models/opt.py](/home/lyc/inference/vllm/vllm/model_executor/models/opt.py) — 注入 `sync_weight_layer` 的位置
  - [vllm/v1/core/kv_cache_offload_config.py](/home/lyc/inference/vllm/vllm/v1/core/kv_cache_offload_config.py) — 新增配置字段
