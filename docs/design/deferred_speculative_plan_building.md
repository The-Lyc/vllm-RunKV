# Deferred Speculative Plan Building — QKV-based Sync

> 状态: 设计草案  
> 日期: 2026-04-20

---

## 1. 动机

当前 RunKV 逐层流水线中，`_runkv_pre_hook` 在关键路径上执行：

1. **plan building** (`_build_dynamic_layer_plan`)：~0.5-1ms CPU
2. **attn metadata 构建** (`_build_layer_attn_metadata`)：~0.3-0.5ms CPU
3. **skip block IDs 计算** (`compute_skip_block_ids_from_plan`)：~0.1ms CPU

这些开销直接延迟了 FlashAttn 的启动，而此时 GPU compute stream 空闲等待。

同时，imbalance 的参考点是 `layer_end[L-1]`（上一层整个 forward 结束），
未能将当前层 assembly + LayerNorm + QKV projection 的 GPU 时间纳入 IO overlap 窗口。

本方案解决两个问题：
- **Imbalance 参考点改为 `qkv_end[L]`**：IO 额外获得 ~1.5ms 的 overlap 窗口
- **Plan building 移出关键路径**：在 `layer()` 返回后 CPU 空闲期投机构建，
  同步点只做微秒级 steady-state 判断

---

## 2. 背景知识

### 2.1 OPT 层内 GPU kernel 序列

以 OPT-2.7b Step 2 L0 为例（nsys 实测）：

```
Replay assembly (Index/Gather + Elementwise)     ~0.07ms
LayerNorm                                         ~0.05ms
GEMM: QKV_proj                                    ~1.59ms
─────── pre_hook 触发点（注册在 self_attn.attn）───────
KV_cache_write (reshape_and_cache)                ~0.08ms
FlashAttn                                          ~1.36ms
─────── pre_hook 结束 ───────
GEMM: O_proj                                       ~1.42ms
Elementwise (residual add)                         ~0.31ms
LayerNorm                                          ~0.55ms
GEMM: FFN_up                                       ~2.11ms
Elementwise (activation)                           ~0.15ms
GEMM: FFN_down                                     ~2.05ms
Elementwise (residual add)                         ~0.06ms
Index/Gather (output split)                        ~0.04ms
```

**关键事实**：pre_hook 注册在 `Attention` 模块
（`model.decoder.layers.X.self_attn.attn`，通过 `static_forward_context`），
触发时 QKV_proj 已经在 compute stream 上 launch 完毕。

### 2.2 CPU-GPU 时间错位

CPU launch 比 GPU execution 快约 7ms（async kernel launch）。  
当 CPU 在 `pre_hook(L)` 中阻塞等待 `prev_end_event(L-1).synchronize()` 时，
GPU 上还在跑 L-1 的后半段 kernel（O_proj → FFN → residual）：

```
CPU timeline:   [layer_compute:L0  ][    layer_compute:L1                ]
                 0ms         4.8ms  5.2ms                          16.3ms

GPU timeline:   [─── L0 全部 kernel ──────────────][─── L1 全部 kernel ────
                 0ms                         11.8ms 11.9ms
```

### 2.3 当前 pre_hook 数据流

```
pre_hook(L) 触发 [QKV(L) 已 launch，FlashAttn(L) 未开始]
│
├── 1. sync_load_layer(L)              GPU stream wait: compute 等 IO(L) 完成
├── 2. sync_cpu_fill_h2d(L)            GPU stream wait: cpu_fill H2D 完成
├── 3. prev_end_event(L-1).sync()      CPU 阻塞: 等上一层 GPU forward 完毕
├── 4. load_ready_event(L).sync()      CPU 阻塞: 等 IO(L) 的 GPU event
├── 5. imbalance = IO_ready(L) - layer_end(L-1)
├── 6. observe_layer_feedback()        controller Newton step 更新 budget
├── 7. build_plan(L+1)                 ~0.5-1ms CPU  ← 关键路径
├── 8. build_attn_metadata(L+1)        ~0.3-0.5ms CPU ← 关键路径
├── 9. compute_skip_block_ids(L+1)     ~0.1ms CPU ← 关键路径
├── 10. launch_IO(L+1)                 non-blocking DMA launch
└── return → FlashAttn(L) 开始执行
```

步骤 7-9 的 CPU 耗时 (~1-2ms) 直接延迟了 FlashAttn 的 GPU 启动。

---

## 3. 设计方案

### 3.1 核心思路

```
                    ┌──────────────── GPU compute stream ─────────────────────┐
                    │                                                         │
Layer L timeline:   │ assembly → LN → QKV(L) ═══╤═══ Attn(L) → O → LN → FFN │
                    │                            │                            │
                    │                        sync point                       │
                    │                            │                            │
IO stream:          │ ════ IO prefetch(L) ═══════╧                            │
                    │                                                         │
Main Py thread:     │       [idle / GIL free]     │   [etc]                   │
Builder thread:     │                                 spec build(L+2)         │
                    └─────────────────────────────┘───────────────────────────┘
```

**三路并行**（同步点之后）：

| 通道 | 任务 | 耗时预估 |
|------|------|----------|
| GPU compute | FlashAttn(L) → O → LN → FFN → residual → assembly(L+1) → LN(L+1) → QKV(L+1) | ~8-10ms |
| GPU IO stream | DMA prefetch(L+1) | ~2-3ms |
| CPU builder 线程 | 无条件投机构建 plan(L+2) + metadata + skip_ids | ~1-2ms |

CPU 1-2ms 构建完全被 GPU 8-10ms 计算隐藏。**后置构建无条件提交，不做稳态预判**：
稳态与否只在下一轮消费端（pre_hook 二分支）判断，稳态则丢弃未用的 spec。原因：
- CPU 反正空闲，多做一次 1-2ms 构建不占任何关键路径；
- `observe_feedback(L)` 的 deadband 结果只是**本层**反馈，预测 pre_hook(L+1) 下一轮
  是否稳态并不准（budget 受新一次 observe_feedback(L+1) 影响）；
- 预测失误的代价：若错判为稳态而实际非稳态，`pop_speculative` 拿到 None，只能退化
  回同步 build 或走一次 stale plan，得不偿失。

### 3.2 新 pre_hook 流程

```
pre_hook(L) 触发 [QKV(L) 已 launch]
│
├── 1. qkv_end_event(L).record()           在 compute stream 上标记 QKV 结束
├── 2. sync_load_layer(L)                  GPU stream wait: compute 等 IO(L)
├── 3. sync_cpu_fill_h2d(L)                GPU stream wait: cpu_fill 等 H2D
├── 4. qkv_end_event(L).synchronize()      CPU 等 QKV(L) 在 GPU 上执行完
├── 5. load_ready_event(L).synchronize()   CPU 等 IO(L) 完成
├── 6. imbalance = IO_ready(L) - qkv_end(L)   ← 新参考点
├── 7. observe_layer_feedback(L)           controller 根据本层 imbalance 更新 budget
│                                            （归属层 L，见 §3.5）
│
├── 8. 二选一：稳态 vs 非稳态（基于 observe_layer_feedback 的返回/deadband 分支）
│     │
│     ├─ IF 稳态（|imbalance| < deadband，budget 未变）：
│     │     # plan(L+1) ← 直接复用 plan(L)；LayerReplayPlan 在单 KV-group、
│     │     # 相同 budget/inputs 下跨层完全相同（见 §3.4 证明）。
│     │     plan_next     = runtime.get_layer_plan(L)
│     │     metadata_next = runtime.get_layer_metadata(L)
│     │     skip_next     = runtime.get_layer_skip_ids(L)
│     │     # 无需任何 build；也无需消费 spec。
│     │
│     └─ ELSE 非稳态（budget 被 observe_layer_feedback 改动）：
│           # 使用上一轮在 CPU 空闲期预构建的 spec。
│           # spec(L+1) 基于 budget(L-1) 构建，与刚刚更新后的 budget 偏差 ≤ 1 block。
│           plan_next, metadata_next, skip_next = runtime.pop_speculative(L + 1)
│
├── 9. runtime.set_layer_plan(L+1, plan_next)
├── 10. runtime.set_layer_metadata(L+1, metadata_next)
├── 11. launch_IO(L+1, skip_next)          non-blocking
└── return → FlashAttn(L) 开始
```

**关键变化**：
- 稳态路径（期望是大多数步数）：pre_hook 里**完全没有 CPU 构建开销**，三次
  `runtime.get_*` 加 IO launch，< 50μs。
- 非稳态路径：消费一个已在后台构建完成的 spec，也**没有阻塞式构建**。

两条路径都把 1-2ms 的 `build_plan + build_metadata + skip_ids` 从 FlashAttn 启动前移除。

### 3.3 后置投机构建（无条件提交到 builder 线程）

在 `OPTDecoder.forward()` 循环中，每层 `layer()` 返回后：

```python
# layer() 已返回, GPU 正在跑 assembly(L+1) + LN(L+1) + QKV(L+1)
# 主 Python 线程接下来会在 pre_hook(L+1) 里卡在 cuda event.synchronize()
# → GIL 释放 → builder 线程在此窗口真正并行运行

_nvtx.range_pop()  # layer_compute:L
layer_end_event.record()

# ---- Speculative build for L+2 —— 无条件提交，不预测稳态 ----
if layer_idx + 2 < self.end_layer:
    runtime.submit_speculative_build(
        target_layer_idx=layer_idx + 2,
        current_plan=plan,
    )
```

`submit_speculative_build` 是**非阻塞**调用，内部向 runtime 持有的单 worker
`ThreadPoolExecutor` 提交任务并存 Future：

```python
def submit_speculative_build(self, target_layer_idx, current_plan):
    future = self._builder_executor.submit(
        self._builder_fn,            # = model_runner._build_speculative_for_layer_impl
        target_layer_idx, current_plan, self,
    )
    self._speculative_futures[target_layer_idx] = future
```

builder 线程里执行的工作：
1. `_build_dynamic_layer_plan(target_layer_idx)`：使用当前 controller budget
2. `_build_layer_attn_metadata(target_layer_idx, spec_plan, prev_plan=current_plan,
   prev_metadata=runtime.get_layer_metadata(target_layer_idx - 2))`：
   **单 KV-group 且 budget 不变时命中早退路径**（L3825-3833），直接返回 prev_metadata
   的引用，builder 线程不创建任何新 GPU tensor。
3. `compute_skip_block_ids_from_plan(spec_plan)`：纯 CPU/Python set 运算。
4. `runtime.set_speculative(target_layer_idx, spec_plan, spec_metadata, spec_skip_ids)`
   写入 runtime 的 speculative 槽。

**消费端**（pre_hook(L+2) 非稳态分支）：

```python
def pop_speculative(self, target_layer_idx, timeout_ms=50):
    fut = self._speculative_futures.pop(target_layer_idx, None)
    if fut is not None:
        try:
            fut.result(timeout=timeout_ms / 1000)   # 一般瞬间拿到
        except TimeoutError:
            fut.result()                             # unbounded 兜底
    plan     = self._speculative_plans.pop(target_layer_idx, None)
    metadata = self._speculative_metadatas.pop(target_layer_idx, None)
    skip_ids = self._speculative_skip_ids.pop(target_layer_idx, None)
    if plan is None:
        return None
    return plan, metadata, skip_ids
```

**用 vs 不用 spec 的判断**：完全由 pre_hook(L+1) 里的二分支（§3.2 / §4 Phase 4）决定。
- 稳态分支：不调 `pop_speculative`，spec(L+1) 仍在 future 槽中；下一层 pre_hook(L+2)
  走哪个分支时可能需要或丢弃。为避免堆积，pre_hook 每次拿到稳态判定后，无论自己用不用
  都调 `runtime.clear_speculative(next_layer_idx)` 把 future 与槽一起丢掉（Future
  可以被 gc；但如果线程还在跑，建议 `future.cancel()` + 静默忽略结果，避免阻塞主线程）。
- 非稳态分支：调 `pop_speculative(L+1)`。若返回 None（bootstrap 尚未提交或 future
  因某种原因缺失），落到一次性 warning 的同步 build fallback。

**并发正确性**：
- `self._speculative_*` 三个 dict 和 `_speculative_futures` 的写操作都是 Python
  **单次字典赋值**，在 CPython 下对单 key 原子；builder 线程写完后主线程 pop。
- builder 线程读 `_runkv_layer_info`、`num_scheduled_tokens_np`、`req_id_to_index`
  等 step-级元数据，这些在 step 内 read-only，无竞争。
- builder 线程若命中 `_build_layer_attn_metadata` 早退（绝大多数情况），不 touch GPU；
  非早退时在 default stream 上做少量 H2D 小 tensor 拷贝，不与 compute/IO stream 冲突。
- Executor 生命周期随 runtime 创建一次、销毁一次，不每步重建；避免线程启动开销。

### 3.4 稳态分支为什么可以"直接复用上一层 plan"

**核心事实**：对 OPT 这类单 KV-cache-group 模型，在同一 step 内若 controller budget
不变，**`LayerReplayPlan` 跨层完全恒等**。依据代码证据：

[gpu_model_runner.py:3825-3833](vllm/v1/worker/gpu_model_runner.py#L3825-L3833) 的
早退路径证明 metadata 构建也走同一等价类：

```python
if (
    prev_metadata is not None
    and prev_plan is not None
    and plan.num_actual_tokens == prev_plan.num_actual_tokens
    and plan.max_query_len == prev_plan.max_query_len
    and torch.equal(plan.query_start_loc, prev_plan.query_start_loc)
    and torch.equal(plan.slot_mapping, prev_plan.slot_mapping)
):
    return prev_metadata
```

**`LayerReplayPlan` 各字段的层无关性**：

| 字段 | 层无关？ | 理由 |
|------|:-------:|------|
| `kv_replay_start_per_req` / `computed_lens_per_req` / `prev_gpu_start_per_req` | ✓ | 来自 step 级 token 元数据 |
| `replay_blocks_per_req` / `replay_block_count` / `skip_logical_block_ids` | ✓ | 由 budget + inputs 决定，budget 稳态下不变 |
| `cpu_fill_*` (positions/logical_ids/block_offsets) | ✓ | token 级 replay 目标，层无关 |
| `gpu_reuse_slice_per_req` / `gpu_reuse_token_count` | ✓ | 索引范围，层无关 |
| `query_start_loc` / `slot_mapping` | ✓ | 对单 KV-group，所有层共享一张 block table |
| `combined_replay_indices` / `combined_scheduled_indices` | ✓ | 索引拼接，层无关 |

**因此稳态分支实现 O(1)**：pre_hook(L+1) 直接

```python
runtime.set_layer_plan(L+1, runtime.get_layer_plan(L))
runtime.set_layer_metadata(L+1, runtime.get_layer_metadata(L))
skip_next = runtime.get_layer_skip_ids(L)   # 也可直接复用
```

没有任何 tensor 拷贝或 CPU 计算；`attn_metadata` 的早退路径也会在后续任何
`_build_layer_attn_metadata` 调用中继续命中，进一步减少 fallback 分支的开销。

**稳态判定来源**：`FeedbackReplayPlanProvider.observe_layer_feedback` 自带 deadband
检查（[opt_dynamic_replay.py:667](vllm/v1/worker/opt_dynamic_replay.py#L667)）。
该调用的 `action` 字段会写入 `FeedbackControllerLayerUpdate`；我们在
runtime 里缓存最后一次 `action == "deadband"` 即可，无需新增 provider 方法。

**约束**：此优化只对 **单 KV-cache-group** 模型（OPT 及类似）成立。多 group
模型各层 `block_table` 不同，`slot_mapping` 不恒等，必须走非稳态分支。
代码中通过 `len(self.kv_cache_config.kv_cache_groups) == 1` assert 保护。

### 3.5 Imbalance 参考点变化

| | 旧参考 | 新参考 |
|---|---|---|
| **公式** | `IO_ready(L) - layer_end(L-1)` | `IO_ready(L) - qkv_end(L)` |
| **语义** | IO(L) 相对上一层 GPU 结束的偏差 | IO(L) 相对本层 QKV GPU 结束的偏差 |
| **归属层** | **L-1**（`observe_layer_feedback(L-1, ...)`） | **L**（`observe_layer_feedback(L, ...)`） |
| **值域** | 正=IO慢, 负=IO快 | 正=IO慢, 负=IO快（不变） |
| **效果** | IO overlap 窗口 = layer_end(L-1) 到 pre_hook(L) | IO overlap 窗口 = 从 IO launch 到 qkv_end(L) |

新参考点给 IO 额外 ~1.5ms overlap 时间（assembly + LN + QKV 的 GPU 执行耗时），
意味着 controller 的 budget 可以更大（加载更多 blocks），从而提高 KV cache 命中率。

**⚠️ 归属层语义变化**：当前代码 `gpu_model_runner.py:3186-3189` 的调用是
`observe_layer_feedback(layer_idx=layer_idx - 1, imbalance_ms=imbalance_ms)`，
因为旧公式测的是 L-1 的 pipeline 平衡。新公式测的是 L 的 pipeline 平衡，
必须改为 `observe_layer_feedback(layer_idx=layer_idx, ...)`。同时
`runtime.set_layer_imbalance_ms(layer_idx - 1, ...)` 和 profiler 的
`set_layer_imbalance_ms(layer_idx - 1, ...)` / `set_layer_controller_update(layer_idx - 1, ...)`
都要改为 `layer_idx`。这是 Phase 1 的隐含必须变更，不可遗漏。

**首层 (L=0) 行为**：新公式下 `qkv_end(0)` 和 `IO_ready(0)` 都可测，可以计算
L=0 的 imbalance。建议**保留** `layer_idx > 0` 的 guard 一段时间作为稳妥过渡；
若验证 L=0 反馈无异常（如 prefill 首步 KV 为空不会触发不合理 budget 跳变），
后续可移除此 guard。

---

## 4. 代码修改方案

### Phase 1: Event 基础设施

| 文件 | 修改 |
|------|------|
| `vllm/v1/worker/opt_dynamic_replay.py` | `OPTDynamicReplayRuntime` 添加 `_qkv_end_events: list[Event \| None]` + getter/setter |
| `vllm/v1/worker/gpu_model_runner.py` | pre_hook 入口处 record `qkv_end_event` |
| `vllm/v1/worker/gpu_model_runner.py` | imbalance 公式改为 `IO_ready(L) - qkv_end(L)` |

**详细说明**：

在 `OPTDynamicReplayRuntime.__post_init__` 添加:
```python
self._qkv_end_events: list[torch.cuda.Event | None] = [None] * self.num_layers
```

在 `_runkv_pre_hook` 的 dynamic replay 分支入口处（sync_load_layer 之前）:
```python
# record() 不指定 stream 时用当前/默认流（即 compute stream）。
# pre_hook 在 self.attn 调用前被触发，此时 qkv_proj 已 launch 到 compute stream，
# 因此此 event 会在 QKV kernel 完成后 signal。
qkv_end_event = torch.cuda.Event(enable_timing=True)
qkv_end_event.record()  # on compute stream
runtime.set_qkv_end_event(layer_idx, qkv_end_event)
```

imbalance 计算段改为（注意归属层从 `layer_idx - 1` 改为 `layer_idx`）:
```python
qkv_end_event = runtime.get_qkv_end_event(layer_idx)
if qkv_end_event is not None and final_ready_event is not None:
    qkv_end_event.synchronize()
    final_ready_event.synchronize()
    qkv_ms = float(step_anchor_event.elapsed_time(qkv_end_event))
    ready_ms = float(step_anchor_event.elapsed_time(final_ready_event))
    imbalance_ms = ready_ms - qkv_ms

    runtime.set_layer_imbalance_ms(layer_idx, imbalance_ms)     # ← 改为 layer_idx
    self.replay_plan_provider.observe_layer_feedback(
        layer_idx=layer_idx,                                     # ← 改为 layer_idx
        imbalance_ms=imbalance_ms,
    )
    # profiler 路径同样：
    opt_component_profiler.set_layer_imbalance_ms(layer_idx, imbalance_ms)
    opt_component_profiler.set_layer_controller_update(
        layer_idx, _ctrl_upd.to_dict()
    )
```

### Phase 2: Speculative Plan + per-layer skip_ids 存储

| 文件 | 修改 |
|------|------|
| `vllm/v1/worker/opt_dynamic_replay.py` | `OPTDynamicReplayRuntime` 添加 speculative plan/metadata/skip_ids 存储，以及 per-layer `skip_ids` 槽（稳态复用需要）|

```python
# ---- 数据槽（builder 线程写、主线程 pop；CPython 单 key 赋值原子）----
_speculative_plans: list[LayerReplayPlan | None]
_speculative_metadata: list[dict[str, Any] | None]
_speculative_skip_ids: list[set[int] | None]

# ---- builder 线程基础设施（runtime 生命周期内仅创建一次）----
_builder_executor: ThreadPoolExecutor   # max_workers=1
_speculative_futures: dict[int, Future]
_builder_fn: Callable | None            # 由 gpu_model_runner 注入

# 当前层使用的 skip_ids（稳态分支要从 layer L 拷贝给 layer L+1）
_layer_skip_ids: list[set[int] | None]

# ---- 异步提交 / 消费 ----
def bind_speculative_builder(self, builder_fn: Callable) -> None:
    self._builder_fn = builder_fn

def submit_speculative_build(self, target_layer_idx, current_plan) -> None:
    assert self._builder_fn is not None
    future = self._builder_executor.submit(
        self._builder_fn, target_layer_idx, current_plan, self,
    )
    self._speculative_futures[target_layer_idx] = future

def pop_speculative(self, layer_idx, timeout_ms: float = 50.0):
    """阻塞等 future 完成（预期瞬间），取出并 clear。"""
    fut = self._speculative_futures.pop(layer_idx, None)
    if fut is not None:
        try:
            fut.result(timeout=timeout_ms / 1000)
        except TimeoutError:
            fut.result()   # unbounded 兜底
    plan     = self._speculative_plans.pop(layer_idx, None)
    metadata = self._speculative_metadata.pop(layer_idx, None)
    skip_ids = self._speculative_skip_ids.pop(layer_idx, None)
    return (plan, metadata, skip_ids) if plan is not None else None

def clear_speculative(self, layer_idx) -> None:
    """稳态分支调用：丢弃未消费的 future，避免堆积；不阻塞。"""
    fut = self._speculative_futures.pop(layer_idx, None)
    if fut is not None:
        fut.cancel()   # 已跑完则 no-op
    self._speculative_plans.pop(layer_idx, None)
    self._speculative_metadata.pop(layer_idx, None)
    self._speculative_skip_ids.pop(layer_idx, None)

def set_speculative(self, layer_idx, plan, metadata, skip_ids) -> None:
    """builder 线程调用，将构建结果写入槽。"""
    ...

def set_layer_skip_ids(self, layer_idx, skip_ids): ...
def get_layer_skip_ids(self, layer_idx) -> set[int] | None: ...
```

### Phase 3: 稳态缓存（轻量，不新增 provider 方法）

| 文件 | 修改 |
|------|------|
| `vllm/v1/worker/opt_dynamic_replay.py` | `OPTDynamicReplayRuntime` 记录最后一次 observe_feedback 的 action |

```python
# 新增字段
_last_observe_action: str | None = None   # "deadband" / "newton_step" / ...

# 新增方法
def note_observe_action(self, action: str) -> None:
    self._last_observe_action = action

def last_observed_in_deadband(self) -> bool:
    return self._last_observe_action == "deadband"
```

`FeedbackReplayPlanProvider.observe_layer_feedback` 已产出
`FeedbackControllerLayerUpdate.action`（见 [opt_dynamic_replay.py:122](vllm/v1/worker/opt_dynamic_replay.py#L122)、L672 周边 deadband 分支），
pre_hook 在调用 `observe_layer_feedback` 后从
`replay_plan_provider.get_layer_controller_update(layer_idx)` 取出 action
传给 `runtime.note_observe_action`。不改动 provider 本身。

### Phase 4: Pre_hook 精简

| 文件 | 修改 |
|------|------|
| `vllm/v1/worker/gpu_model_runner.py` | 重构 dynamic replay pre_hook 的中间段 |

**移除**（实际行号已核对当前代码）：
- `_build_dynamic_layer_plan` 调用 [gpu_model_runner.py:3229-3243](vllm/v1/worker/gpu_model_runner.py#L3229-L3243)
- `_build_layer_attn_metadata` 调用 [gpu_model_runner.py:3246-3270](vllm/v1/worker/gpu_model_runner.py#L3246-L3270)
- `compute_skip_block_ids` / `compute_skip_block_ids_from_plan` 调用 [gpu_model_runner.py:3273-3317](vllm/v1/worker/gpu_model_runner.py#L3273-L3317)

**替换为**：
```python
if next_layer_info is not None:
    next_layer_name, next_layer_idx, next_gid = next_layer_info

    # ---- 路径选择 ----
    # 稳态：observe_layer_feedback(layer_idx) 刚走了 deadband 分支，budget 未变
    #       → plan(L+1) 与 plan(L) 完全恒等（见 §3.4）
    if runtime.last_observed_in_deadband():
        next_plan           = runtime.get_layer_plan(layer_idx)
        next_metadata       = runtime.get_layer_metadata(layer_idx)
        next_skip_block_ids = runtime.get_layer_skip_ids(layer_idx)
        # spec 没被用到，若存在则清理
        runtime.clear_speculative(next_layer_idx)
    else:
        # 非稳态：消费后台构建的 spec
        spec = runtime.pop_speculative(next_layer_idx)
        if spec is not None:
            next_plan, next_metadata, next_skip_block_ids = spec
        else:
            # 防御性 fallback：bootstrap / 后置 build 出错；打日志一次后同步构建
            if not self._dynamic_replay_spec_fallback_logged:
                logger.warning(
                    "RunKV dynamic replay: spec(L%d) missing at pre_hook(L%d) "
                    "non-steady branch; falling back to synchronous build.",
                    next_layer_idx, layer_idx,
                )
                self._dynamic_replay_spec_fallback_logged = True
            next_plan = self._build_dynamic_layer_plan(
                layer_idx=next_layer_idx, gid=next_gid,
                num_reqs=self._lr_num_reqs,
                num_scheduled_tokens_np=self._lr_num_scheduled_tokens_np,
                prev_layer_plan=current_plan,
            )
            next_metadata = self._build_layer_attn_metadata(
                layer_idx=next_layer_idx, plan=next_plan,
                prev_plan=current_plan, prev_metadata=current_metadata,
                base_seq_lens=self.seq_lens.cpu,
                base_max_seq_len=(...),
                block_table_tensor=self.paged_block_tables[next_gid],
                num_reqs=self._lr_num_reqs,
            )
            next_skip_block_ids = manager.compute_skip_block_ids_from_plan(next_plan)

    # ---- 写入 runtime 给下一层 forward 使用 ----
    runtime.set_layer_plan(next_layer_idx, next_plan)
    runtime.set_layer_metadata(next_layer_idx, next_metadata)
    runtime.set_layer_skip_ids(next_layer_idx, next_skip_block_ids)
```

要点：
- `pop_speculative` 取出后立即清槽，避免跨 step 残留。
- **当前层的 `skip_block_ids` 也要写入 `runtime._layer_skip_ids[layer_idx]`**（目前
  代码里没有这个存储；Phase 2 新增的槽需要在本段和 bootstrap 里都 set）。否则
  下一层走稳态分支时 `get_layer_skip_ids(layer_idx)` 会取到 None。

### Phase 5: 后置投机构建

| 文件 | 修改 |
|------|------|
| `vllm/v1/worker/gpu_model_runner.py` | 新增 `_build_speculative_for_layer()` 方法 |
| `vllm/v1/worker/opt_dynamic_replay.py` | `OPTDynamicReplayRuntime` 添加对 model_runner 方法的 callback |
| `vllm/model_executor/models/opt.py` | `OPTDecoder.forward()` 循环中 `layer()` 返回后调用 |

**opt.py 变更**（`OPTDecoder.forward` 循环尾部，`layer_end_event.record()` 之后）：
```python
layer_end_event.record()
runtime.set_layer_end_event(layer_idx, layer_end_event)

# Speculative build for L+2 — 无条件提交到 builder 线程。
# 稳态预测不在这里做；pre_hook(L+2) 的二分支负责决定用还是丢弃。
if layer_idx + 2 < self.end_layer:
    runtime.submit_speculative_build(
        target_layer_idx=layer_idx + 2,
        current_plan=plan,
    )
```

**callback 注入**：在 `_prepare_dynamic_replay_runtime` 中将
`_build_speculative_for_layer_impl` 的引用连同 step 级只读参数绑定进 runtime，供
builder 线程调用：
```python
runtime.bind_speculative_builder(
    builder_fn=functools.partial(
        self._build_speculative_for_layer_impl,
        num_reqs=num_reqs,
        num_scheduled_tokens_np=num_scheduled_tokens_np,
    )
)
```

**runtime 持有的执行器**（在 `OPTDynamicReplayRuntime.__post_init__` 里惰性创建
一次，整个 runtime 生命周期共享，**不每步重建**）：
```python
from concurrent.futures import ThreadPoolExecutor

self._builder_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="runkv-spec-builder",
)
self._speculative_futures: dict[int, Future] = {}
# runtime 析构或模型 unload 时调用 self._builder_executor.shutdown(wait=False)
```

**`_build_speculative_for_layer_impl`**：
```python
def _build_speculative_for_layer_impl(
    self,
    target_layer_idx: int,
    current_plan: LayerReplayPlan,
    runtime: OPTDynamicReplayRuntime,
    *,
    num_reqs: int,
    num_scheduled_tokens_np: np.ndarray,
) -> None:
    gid = ...  # 从 _runkv_layer_info 查找
    spec_plan = self._build_dynamic_layer_plan(
        layer_idx=target_layer_idx, gid=gid,
        num_reqs=num_reqs,
        num_scheduled_tokens_np=num_scheduled_tokens_np,
        prev_layer_plan=current_plan,
    )
    spec_metadata = self._build_layer_attn_metadata(
        layer_idx=target_layer_idx, plan=spec_plan,
        prev_plan=current_plan,
        prev_metadata=runtime.get_layer_metadata(target_layer_idx - 2),
        ...
    )
    spec_skip_ids = self.layer_recompute_manager.compute_skip_block_ids_from_plan(
        spec_plan
    )
    runtime.set_speculative(target_layer_idx, spec_plan, spec_metadata, spec_skip_ids)
```

### Phase 6: DMA Segment 预组织（可选优化）

| 文件 | 修改 |
|------|------|
| `vllm/v1/worker/gpu_model_runner.py` | `PagedBlockMapper.load_layer_async()` 添加 `precomputed_segments` 参数 |

在 speculative build 阶段预计算好 segments：

```python
def precompute_dma_segments(skip_block_ids, mapping):
    skip = skip_block_ids or set()
    load_ids = sorted(lid for lid in mapping if lid not in skip)
    segments = []
    if load_ids:
        seg_start, seg_len = load_ids[0], 1
        for j in range(1, len(load_ids)):
            if load_ids[j] == load_ids[j-1] + 1:
                seg_len += 1
            else:
                segments.append((seg_start, seg_len))
                seg_start, seg_len = load_ids[j], 1
        segments.append((seg_start, seg_len))
    return segments, load_ids
```

`load_layer_async` 如果收到 `precomputed_segments`，跳过自己的 segment 构建。

### Phase 7: Bootstrap 和边界处理

| 文件 | 修改 |
|------|------|
| `vllm/v1/worker/gpu_model_runner.py` | `_prepare_dynamic_replay_runtime` 中额外构建 plan(1) |
| `vllm/v1/worker/gpu_model_runner.py` | pre_hook 层 0 特殊处理 |

**Bootstrap 变更**：

当前 bootstrap 已经构建 plan(0) + metadata(0) + launch IO(0)。
新方案需额外构建 plan(1) 的 speculative 数据：

```python
# 现有: 构建 plan(0), metadata(0), skip_ids(0), launch IO(0)

# 新增 (a): 把 layer0 的 skip_ids 也写入 runtime._layer_skip_ids[0]，
#          稳态分支下 pre_hook(0) 会从此读取。
runtime.set_layer_skip_ids(layer0_idx, skip_block_ids)

# 新增 (b): 为 pre_hook(0) 的非稳态分支提交 spec(1) 构建。
#          pre_hook(0) 时 observe_feedback 从未调用过，
#          `last_observed_in_deadband()` 返回 False（初始值），
#          所以会走非稳态分支 → 必须有 spec(1) 可消费。
#          这里 **submit 而非同步 build**，让 builder 线程在 GPU 执行
#          前 prologue 期间并行构建；pre_hook(0) 的 pop_speculative 再等 future。
if len(self._runkv_layer_info) > 1:
    layer1_name, layer1_idx, layer1_gid = self._runkv_layer_info[1]
    runtime.submit_speculative_build(
        target_layer_idx=layer1_idx,
        current_plan=layer0_plan,
    )
```

**关于 `last_observed_in_deadband()` 的初始值**：初始化为 `False` 很重要 —
bootstrap 时 observe_feedback 从未调用过，如果初始值是 True 则 pre_hook(0)
会误走稳态分支读 `get_layer_plan(-1)` 造成越界。`_last_observe_action = None`
→ `last_observed_in_deadband() == False`，天然正确。

**边界条件**：
- **Layer 0**：pre_hook(0) 中 imbalance 仍跳过（`layer_idx > 0` guard 已有）；
  使用 bootstrap 预构建的 speculative plan(1)
- **最后一层 (L=31)**：`layer()` 返回后不做 speculative build（L+2=33 > num_layers）
- **Layer 30**：`layer()` 返回后 speculative build for L=32 不存在，跳过；
  但 pre_hook(31) 需要的 plan 已在 pre_hook(30) 后的投机构建中准备好
  （实际是 layer(30) 返回后 build for L=32，但 32 不存在 → 直接跳过）

修正：投机构建目标是 `L+2`。
- layer(0) 返回 → build spec(2) ✓
- layer(1) 返回 → build spec(3) ✓
- ...
- layer(29) 返回 → build spec(31) ✓
- layer(30) 返回 → build spec(32) → 跳过（超出范围）
- layer(31) 返回 → 循环结束

pre_hook 的 speculative 数据来源：
- pre_hook(0)：bootstrap 预构建的 spec(1)
- pre_hook(1)：bootstrap 预构建的 spec(1)? — 不对。

重新梳理时序：

```
Bootstrap:       build plan(0), metadata(0), launch IO(0)
                 build spec plan(1), metadata(1), skip_ids(1)  ← 新增

pre_hook(0):     sync IO(0), record qkv_end(0), [skip imbalance for L=0]
                 use spec(1) → set plan(1), launch IO(1)
layer(0) 返回:   build spec(2)  ← CPU 空闲期

pre_hook(1):     sync IO(1), record qkv_end(1), imbalance = IO(1) - qkv_end(1)
                 observe_feedback, use spec(2) → set plan(2), launch IO(2)
layer(1) 返回:   build spec(3)

...

pre_hook(30):    use spec(31) → set plan(31), launch IO(31)
layer(30) 返回:  L+2=32 > num_layers → 跳过

pre_hook(31):    sync IO(31), last layer, no next layer → 不需要 spec
layer(31) 返回:  循环结束
```

这样每个 pre_hook(L) 都能拿到上一轮构建的 spec(L+1)。✓

---

## 5. Plan 准确性分析

### 5.1 稳态分支：**零偏差**（新）

稳态下 pre_hook(L+1) 直接复用 plan(L)/metadata(L)/skip_ids(L)。
由于 `observe_feedback(L)` 走了 deadband 分支 → `global_budget_blocks` 未变 →
对同一 step 内相同 input batch，`_build_dynamic_layer_plan` 的输出对所有层都
恒等（见 §3.4 证明表）。因此"复用 plan(L)"与"现场重新 build plan(L+1)"
**结果完全一致**（逐字段相等），无任何偏差。

### 5.2 非稳态分支：spec 的 1-block 偏差

非稳态下消费的 spec(L+1) 是在 layer(L-1) 返回后 CPU 构建的，当时依据的是
`observe_feedback(L-1)` 更新后的 budget。到 pre_hook(L+1) 时，
`observe_feedback(L)` 又动了 budget。

最大 budget 偏差 = `damping × max_step = 0.3 × 4 = 1.2 blocks ≈ 1 block`。

偏差后果：某个 request 可能多加载或少加载 1 个 block 的 KV。在 replay 场景下
这是完全安全的：多加载 = 浪费少量带宽，少加载 = 下一层追上。

### 5.3 Controller 收敛不受影响

`observe_layer_feedback` 始终在 pre_hook 中以实测 imbalance 调用，
controller 的 Newton step 使用真实反馈，不受 speculative plan 或稳态复用的影响。
两条路径都只改变**执行路径**，不改变**控制循环**。

---

## 6. 关键文件清单

| 文件 | 关键位置 | 改动类型 |
|------|---------|---------|
| [gpu_model_runner.py](vllm/v1/worker/gpu_model_runner.py) | `_runkv_pre_hook` (~L3046-3492) | 重构：精简 plan/meta/skip 段为"二选一"；imbalance 公式与归属层变更；在 observe_feedback 之后调 `runtime.note_observe_action` |
| | `_prepare_dynamic_replay_runtime` (~L8225-8355) | bootstrap 时调 `set_layer_skip_ids(0)`；预构 spec(1)；绑定 builder callback |
| | `PagedBlockMapper.load_layer_async` (~L721-895) | 可选：添加 `precomputed_segments` 参数 |
| | 新增 `_build_speculative_for_layer_impl` | 新方法 |
| [opt_dynamic_replay.py](vllm/v1/worker/opt_dynamic_replay.py) | `OPTDynamicReplayRuntime` (~L988-1195) | 添加 `qkv_end_events`、speculative storage、`_layer_skip_ids`、`_last_observe_action`、`_builder_executor` (单 worker)、`_speculative_futures` 及相应 get/set/submit/pop/clear |
| | `FeedbackReplayPlanProvider` | **不改动**；沿用现有 `get_layer_controller_update(...).action` 读取 deadband 结果 |
| [opt.py](vllm/model_executor/models/opt.py) | `OPTDecoder._forward_dynamic_replay` loop (~L501-512) | 每层 `layer()` 返回后**无条件** `runtime.submit_speculative_build(L+2, plan)`（非阻塞） |

---

## 7. 验证方案

### 7.1 正确性

1. **Kernel 序列不变**：`extract_nvtx_kernel_map.py --step 5 --detail --layers 0 1 2`
   对比改前改后每层 kernel 类型和顺序完全一致

2. **NVTX 时序验证**：投机构建的 NVTX range（`runkv:speculative_build:L{X}`）
   应出现在 `layer_compute:L{X-2}` 之后、`layer_compute:L{X-1}` 之前

3. **Assembly-Plan 一致性**：打印每层 assembly 使用的 plan 的 `replay_token_count`，
   确认和实际 `combined_hidden_states.shape[0]` 一致

### 7.2 性能

4. **Pre_hook 耗时下降**：通过 `RUNKV_PREHOOK_TIMING_DIR` 输出对比
   - 改前：`build_plan_ms` + `build_meta_ms` + `skip_ids_ms` ≈ 1-2ms
   - 改后：这三项消失

5. **Imbalance 曲线对比**：`analyze_per_layer_timing.py`
   - 新 imbalance 初始应更负（IO 获得更多 overlap 时间）
   - Controller 仍收敛到稳态

6. **端到端吞吐**：`benchmark_throughput.py` 对比 tokens/s

### 7.3 Profile 对比

```bash
# 改前
nsys profile --trace=cuda,nvtx -o baseline.nsys-rep python ...
nsys export --type sqlite baseline.nsys-rep -o baseline.sqlite

# 改后
nsys profile --trace=cuda,nvtx -o restructured.nsys-rep python ...
nsys export --type sqlite restructured.nsys-rep -o restructured.sqlite

# 对比 kernel 映射
python3 tools/extract_nvtx_kernel_map.py \
    --sqlite baseline.sqlite --sqlite2 restructured.sqlite --step 5
```

---

## 8. 实现顺序与依赖

```
Phase 1 (event infra) ──┐
Phase 2 (spec+skip storage) ──┤
Phase 3 (steady cache) ──────┴──→ Phase 4 (two-branch pre_hook) ──→ Phase 5 (spec build) ──→ Phase 7 (bootstrap)
                                                                            ↑
                                                                    Phase 6 (optional)
```

- Phase 1/2/3 相互独立，可并行
- Phase 4 依赖 1+2+3
- Phase 5 依赖 4
- Phase 7 在最后
- Phase 6 可随时加入，不阻塞主线

---

## 9. 待决项

### 9.1 opt.py 如何访问 plan builder

**推荐方案**：`OPTDynamicReplayRuntime` 持有 builder callback + 单 worker 线程池。
opt.py 只调用 `runtime.submit_speculative_build(L+2, current_plan)`（非阻塞），
runtime 内部向 `_builder_executor` 提交任务、由注入的 callback 执行真正的构建。

理由：保持 opt.py 对 gpu_model_runner 内部实现无感知；同时把"用何种并发原语"
的选择封装在 runtime 侧，便于未来改成 multi-worker 或替换为 asyncio。

### 9.2 TightLLM 模式适配

`TightLLMReplayPlanProvider` 也使用 `observe_layer_feedback`。
重构时需验证 TightLLM 路径不受影响。如果 TightLLM 也想受益于 deferred build，
需要确认其 `get_layer_plan` 也兼容 speculative 模式。

### 9.3 DMA Staging Buffer（远期）

更激进优化：投机构建时在 CPU 侧把散碎 blocks memcpy 到连续 staging buffer，
DMA 一次传输。代价：额外 CPU memcpy。
**建议先实现 segment 预计算（Phase 6），观察效果后再决定。**

### 9.4 cpu_fill_h2d 的处理

当前 `cpu_fill_token_count > 0` 时需要 `load_cpu_fill_h2d_async`。
投机构建需要预计算 cpu_fill 相关数据（positions、logical_ids、block_offsets），
这些都在 `LayerReplayPlan` 中已有，不需要额外存储。

---

## 10. 审阅总结（2026-04-20）

针对前 9 节的一次 walk-through 审阅，结论：**设计方向合理、整体可行**；
已就下列 4 类问题直接修订了上文，列出以便追溯。

### 10.1 原文与当前代码事实的偏差（已修订）

| # | 原文 | 现状 / 修订 |
|---|------|------|
| A | 建议新增 `is_steady_state()` 作双判分支 | 经用户指出：LayerReplayPlan 在单 KV-group + budget 不变时跨层恒等（§3.4）。**保留**二分支：稳态走**零开销引用复用**，非稳态消费 spec；稳态信号用 runtime 缓存的 `_last_observe_action`，不改 provider |
| B | §3.5 未说明 imbalance 归属层变化 | 新公式测的是 layer L 的平衡度；`observe_layer_feedback(layer_idx=layer_idx - 1, ...)`（[gpu_model_runner.py:3186-3189](vllm/v1/worker/gpu_model_runner.py#L3186-L3189)）必须改为 `layer_idx=layer_idx`，同步更新 `set_layer_imbalance_ms` 和 profiler 的两处 `set_layer_*` 调用 |
| C | §4 Phase 4 fallback 注释"L=0 pre_hook 没 spec" | 与 Phase 7 bootstrap 预构建 spec(1) 矛盾；改为"防御性回退 + 告警日志" |
| D | §6 文件清单行号与当前代码错位（偏差 10-380 行） | 已逐条核对并改用 markdown 链接，行号基准 = 当前 `runkv` 分支 |
| E | §3.3 后置 spec build 无条件执行 | 改为"上轮 deadband → 跳过 build"，稳态下 CPU 零开销 |
| F | §4 Phase 2 只存 spec，未存 per-layer skip_ids | 稳态复用需要 `runtime.get_layer_skip_ids(L)`；Phase 2 新增 `_layer_skip_ids` 槽，bootstrap 与 Phase 4 都要 `set_layer_skip_ids` |

### 10.2 仍需在实现时验证的点

1. **`observe_layer_feedback` 归属层**：controller 的 Newton-step 基于 (budget, imbalance) pair 推 gain。
   归属层从 L-1 改为 L 后，**Newton-step 首次迭代会用 layer 0 的 imbalance**（新增的 L=0 反馈）。
   需观察 Controller 的 `secant` 是否收敛正常；若出现振荡，可临时保留 `layer_idx > 0` guard。

2. **spec 的跨 step 残留**：`OPTDynamicReplayRuntime` 是每 step 新建的实例（见
   [gpu_model_runner.py:8245-8250](vllm/v1/worker/gpu_model_runner.py#L8245-L8250)），
   天然不跨 step。但 bootstrap 预构建 spec(1) 前要**断言** runtime 的 spec slot 是空的。

3. **CPU 时间窗**：build_speculative 同步插入在 decoder forward 循环中。
   CPU 相对 GPU 快约 7ms，1-2ms 的 CPU 构建会被 CPU-GPU launch slack 完全吸收；
   但若未来 build 膨胀（例如 batch size 扩大），需在 NVTX 中监控
   `runkv:speculative_build:L{X}` 的 CPU 占时是否 > 5ms。

4. **opt.py 对 pre-LN 的假设**：当前 `_forward_dynamic_replay` 已硬要求 PP=1
   （[opt.py:391-394](vllm/model_executor/models/opt.py#L391-L394)）。
   kernel 序列讨论默认 `do_layer_norm_before=True`。OPT-350M 的 post-LN
   路径不在 dynamic replay 支持范围内，无需专门适配，但可加一行 assert 防御。

### 10.3 可行性判定

- **功能正确性**：✅ 归属层变更 + 二分支 + fallback + bootstrap 闭环，逻辑自洽。
  稳态分支**零偏差**（§5.1），非稳态分支偏差 ≤ 1 block（§5.2）。
- **性能收益**：
  - pre_hook 关键路径：稳态分支完全无 CPU 构建（~0ms），非稳态分支消费 spec 也无阻塞
  - IO overlap 窗口扩大 ~1.5ms
  - 总计 per-layer 预期收益 2-3ms，32 层 OPT-2.7B 单 step 减 60-90ms
  - **额外**：稳态下后置 spec build 也跳过，controller 稳定后 CPU 接近 zero-overhead
- **实施风险**：🟡 最大风险是"imbalance 归属层变更"，controller 行为可能改变。
  上线前必须做 imbalance 收敛曲线对比实验（详见 §11 Step 9）。次要风险：
  多 KV-group 模型稳态分支失效，需加 assert 保护。

---

## 11. Step-by-Step 实施步骤

所有步骤在 `runkv` 分支上顺序推进；每个 step 完成后跑 §7.1 的 kernel 序列
对比 + 单元测试，再进下一步。**不要跨 step 合并提交**，便于 bisect。

### Step 1 — OPTDynamicReplayRuntime 扩展（qkv_end + spec + per-layer skip + deadband 缓存）

文件：[opt_dynamic_replay.py](vllm/v1/worker/opt_dynamic_replay.py)

- [ ] 在 `OPTDynamicReplayRuntime` 的字段区追加：
  - `_qkv_end_events: list[torch.cuda.Event | None]`
  - `_speculative_plans: list[LayerReplayPlan | None]`
  - `_speculative_metadata: list[dict[str, Any] | None]`
  - `_speculative_skip_ids: list[set[int] | None]`
  - `_layer_skip_ids: list[set[int] | None]`  **← 新增，稳态复用需要**
  - `_last_observe_action: str | None = None`  **← 新增，pre_hook 稳态判定**
  - `_builder_fn: Callable[...] | None = None`  **← 由 model_runner 注入**
  - `_builder_executor: ThreadPoolExecutor`  **← `__post_init__` 里 `ThreadPoolExecutor(max_workers=1)` 创建一次**
  - `_speculative_futures: dict[int, Future]`
- [ ] `__post_init__` 对应初始化：list 槽 `[None] * num_layers`；
  `_speculative_futures = {}`；executor 带 `thread_name_prefix="runkv-spec-builder"`。
- [ ] 新增方法：
  - `set/get_qkv_end_event`
  - `set/get_layer_skip_ids`
  - `set_speculative` / `pop_speculative` / `clear_speculative`
  - `note_observe_action(action: str)` / `last_observed_in_deadband() -> bool`
  - `bind_speculative_builder(builder_fn)` / `submit_speculative_build(target_layer_idx, current_plan)`
- [ ] **析构/关闭**：给 runtime 加 `close()` 方法 `self._builder_executor.shutdown(wait=False, cancel_futures=True)`，
  在 `_prepare_dynamic_replay_runtime` 释放旧 runtime 时调一次；避免解释器退出时线程挂起。
- [ ] **不修改** `FeedbackReplayPlanProvider`。

**验证**：单测构造 runtime，逐一验证 get/set 往返；初始 `last_observed_in_deadband()==False`；
submit 一个 `builder_fn = lambda *a, **k: runtime.set_speculative(...)` 假实现后，
`pop_speculative` 能拿到写入值。

---

### Step 2 — pre_hook 入口 record qkv_end_event

文件：[gpu_model_runner.py](vllm/v1/worker/gpu_model_runner.py)，
函数 `_runkv_pre_hook` 的 dynamic replay 分支入口（L3091 之后，L3122 之前）

- [ ] 在进入 `sync_load_layer` 前一行：
  ```python
  if self.device.type == "cuda":
      qkv_end_event = torch.cuda.Event(enable_timing=True)
      qkv_end_event.record()  # default = compute stream
      runtime.set_qkv_end_event(layer_idx, qkv_end_event)
  ```
- [ ] 还不改 imbalance 计算（下一步才改），此时 event 已 record 但未使用。
- [ ] 运行一次 E2E 冒烟，确认 event 不泄漏（kernel 序列不变）。

---

### Step 3 — 切换 imbalance 参考点 + 归属层

文件：`gpu_model_runner.py` L3155-3220 的 imbalance 计算段。

- [ ] `prev_end_event = runtime.get_layer_end_event(layer_idx - 1)` →
  `qkv_end_event = runtime.get_qkv_end_event(layer_idx)`
- [ ] guard 从 `layer_idx > 0` 保留（稳妥过渡；见 §3.5 首层说明）
- [ ] `end_ms = step_anchor_event.elapsed_time(prev_end_event)` →
  `qkv_ms = step_anchor_event.elapsed_time(qkv_end_event)`
- [ ] `runtime.set_layer_imbalance_ms(layer_idx - 1, imbalance_ms)` →
  `runtime.set_layer_imbalance_ms(layer_idx, imbalance_ms)`
- [ ] `observe_layer_feedback(layer_idx=layer_idx - 1, ...)` →
  `observe_layer_feedback(layer_idx=layer_idx, ...)`
- [ ] profiler 的 `set_layer_imbalance_ms(layer_idx - 1, ...)` 与
  `set_layer_controller_update(layer_idx - 1, ...)` 同步改为 `layer_idx`
- [ ] **新增**：`observe_layer_feedback(...)` 返回后立即读
  `replay_plan_provider.get_layer_controller_update(layer_idx)` 的 `action` 字段，
  传给 `runtime.note_observe_action(action)`。这是稳态分支判定的唯一来源。

**验证**：
- 打开 `RUNKV_PREHOOK_TIMING_DIR`，确认 imbalance 曲线不再有负突刺（IO 多出 1.5ms overlap，初期 imbalance 更负是预期）
- Controller 仍能收敛（10 步内 |imbalance| < 1ms）
- 打印 `runtime.last_observed_in_deadband()` 的分布：前若干层 False → 稳态后 True

---

### Step 4 — 新增 `_build_speculative_for_layer_impl`

文件：`gpu_model_runner.py`。

- [ ] 新方法：
  ```python
  def _build_speculative_for_layer_impl(
      self, target_layer_idx: int, current_plan: LayerReplayPlan,
      runtime: OPTDynamicReplayRuntime, *, num_reqs, num_scheduled_tokens_np,
  ) -> None:
      # 查 _runkv_layer_info 找 gid + layer_name
      # 依次调用 _build_dynamic_layer_plan / _build_layer_attn_metadata /
      # layer_recompute_manager.compute_skip_block_ids_from_plan
      # 写入 runtime.set_speculative(target_layer_idx, ...)
  ```
- [ ] NVTX 包一层 `runkv:speculative_build:L{target_layer_idx}`
- [ ] 暂不挂到 decoder forward（下一步做），先确保方法本身可被直接调用跑通。

---

### Step 5 — `_prepare_dynamic_replay_runtime` 绑定 callback + bootstrap spec(1)

文件：`gpu_model_runner.py`，`_prepare_dynamic_replay_runtime` (L8225-8355)。

- [ ] 创建 runtime 后立即绑定：
  ```python
  runtime.bind_speculative_builder(functools.partial(
      self._build_speculative_for_layer_impl,
      num_reqs=num_reqs,
      num_scheduled_tokens_np=num_scheduled_tokens_np,
  ))
  ```
  builder_fn 的实际签名：`(target_layer_idx, current_plan, runtime, *, num_reqs, num_scheduled_tokens_np)`。
- [ ] 在构建完 layer0 的 `skip_block_ids` 之后（L8296-8312 周边），调
  `runtime.set_layer_skip_ids(layer0_idx, skip_block_ids)`。这是**稳态分支
  pre_hook(1) 读取的来源**，不可遗漏。
- [ ] 现有 bootstrap 结尾（L8337 之前）追加：若 `len(_runkv_layer_info) >= 2`，
  调用 `runtime.submit_speculative_build(target_layer_idx=layer1_idx, current_plan=layer0_plan)`。
  提交后**不等待 future**；pre_hook(0) 走非稳态分支时 `pop_speculative(1)` 再阻塞
  等结果（预计只需 1-2ms）。
- [ ] 日志里补 `spec1_submitted=True/False`、`layer0_skip_cached=True`。

**验证**：断点 runtime；bootstrap 后 `runtime._speculative_plans[1]` 非空，
`runtime._layer_skip_ids[0]` 非空。

---

### Step 6 — opt.py decoder forward 循环每层后触发 spec(L+2)

文件：[opt.py](vllm/model_executor/models/opt.py) 的 `_forward_dynamic_replay`
（L417-512 循环体）。

- [ ] 在 `runtime.set_layer_end_event(layer_idx, layer_end_event)` 之后、
  `runtime.capture_scheduled_layer_input(...)` 之前（或之后都可以，无强依赖）插入：
  ```python
  if layer_idx + 2 < self.end_layer:
      # 无条件提交到 builder 线程，非阻塞。
      # 稳态与否由下一轮 pre_hook(L+2) 的二分支决定用或丢弃。
      runtime.submit_speculative_build(
          target_layer_idx=layer_idx + 2,
          current_plan=plan,
      )
  ```
- [ ] `plan` 取自循环入口 `plan = runtime.get_layer_plan(layer_idx)`（L421）。
- [ ] **不在这里做 `last_observed_in_deadband()` 预判**：CPU 反正空闲，多做一次
  1-2ms 构建被 GPU 8-10ms 掩盖；预测 pre_hook(L+1) 稳态性不准（下一轮 observe
  还没触发），错判会造成 `pop_speculative` 拿到 None 退化同步 build 的更高代价。

**验证**：
- nsys profile 查看 `runkv:speculative_build:L*` NVTX range 每步都出现，且完整
  落在对应 `runkv:layer_compute:L{X-2}` 到 `runkv:prehook:L{X-1}` 的 GPU 窗口内
- builder 线程的 Python 栈在 `py-spy` 里应处于 `_build_layer_attn_metadata` 附近
  运行，而非始终阻塞
- 稳态阶段：主线程 pre_hook 的 `pop_speculative` 调用量应近 0（全部走稳态复用路径），
  而 builder 仍在持续产出 spec —— 这些 spec 由 `clear_speculative` 丢弃，无泄漏

---

### Step 7 — pre_hook 精简为二分支（稳态复用 / 非稳态消费 spec）

文件：`gpu_model_runner.py`，`_runkv_pre_hook` 的 build_plan / build_metadata /
skip_ids 三段（L3222-3317）。

- [ ] 将三段同步构建**整体替换**为 §4 Phase 4 "替换为" 中的二分支代码：
  - 稳态（`runtime.last_observed_in_deadband()==True`）：读 `get_layer_plan(L)` /
    `get_layer_metadata(L)` / `get_layer_skip_ids(L)`，引用赋值给 L+1 槽。
  - 非稳态：`runtime.pop_speculative(L+1)`。若 None，走一次性 warning 的同步 fallback。
- [ ] **本层 skip_block_ids 也要写**：在当前层原先 `compute_skip_block_ids_*` 产出
  `skip_block_ids` 后（或从 spec/prev 拿到后），调 `runtime.set_layer_skip_ids(layer_idx, skip_block_ids)`。
  稳态下一层会从这里读。
- [ ] schedule_io 段（L3319-3386）的 `next_plan` / `next_skip_block_ids` 直接
  用上面写入 runtime 的结果（通过局部变量传递，避免再读一次）。

**验证**：
- `RUNKV_PREHOOK_TIMING_DIR` 输出的 `build_plan_ms` / `build_meta_ms` /
  `skip_ids_ms` 在稳态阶段降为 ~0，非稳态阶段也 ~0（只是 pop + ref assign）
- kernel 序列比对（§7.1）完全一致
- fallback warning 不出现（若出现说明 bootstrap 或 step 6 有 bug）
- 打印稳态/非稳态分支触发计数；controller 稳定后应几乎全部稳态

---

### Step 8 — （可选）Phase 6 precomputed_segments

- 先跑完 Step 7 的完整 benchmark，若 `segment_build` 仍是瓶颈（> 0.3ms）再做。
- 按 §4 Phase 6 的方案修改 `load_layer_async`，不阻塞主线。

---

### Step 9 — 回归验证与 benchmark

1. **正确性**：
   - `pytest tests/v1/runkv/`（若有相关 case）
   - `extract_nvtx_kernel_map.py --step 5 --detail --layers 0 1 2` 对比改前后
2. **性能**：
   - `benchmark_throughput.py` OPT-2.7B，对比 tokens/s
   - `analyze_per_layer_timing.py` 对比 per-layer imbalance 曲线与收敛速度
3. **Controller 行为**：
   - 确认新归属层 (L 代替 L-1) 下，controller 在 20 步内收敛
   - 若出现振荡，临时把 imbalance guard 恢复为 `layer_idx > 0` 或将 spec 取 L+3（不推荐）
4. **跨 step 稳定性**：至少连续 200 步无 fallback warning、无 assert 失败。

### Step 10 — 提交前检查

- [ ] `git diff` 自审；确认没有遗留注释或 dead code
- [ ] 补一条 commit：`[RunKV] Defer speculative plan building off pre_hook critical path`
- [ ] 如 imbalance 归属层变更导致 benchmark baseline 漂移，单独再提一个 commit 固化新的
  controller 初始参数，便于后续 bisect。
