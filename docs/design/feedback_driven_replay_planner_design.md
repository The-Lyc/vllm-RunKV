## RunKV Feedback-Driven Replay Planner 方案

### Summary
在现有 `prev_layer_output_dynamic` runtime 上新增一个 feedback-driven replay planner。第一版目标不是 ILP，也不是复杂全局优化器，而是一个低维、反馈驱动、可在线更新、可逐步接管执行的 replay 控制器。

这个 planner 的核心思想是：

- 运行时继续沿用现有的 layer-wise dynamic replay 主干；
- planner 不直接输出复杂的 per-request 全局最优解，而只维护一个低维控制量：
  - `global_replay_budget_blocks`
- 每一层执行后，根据当前层观测到的 signed imbalance 反馈，更新下一轮可用的 replay budget；
- batch 内 replay 分配先用一个稳定、可解释、低侵入的规则：
  - contiguous suffix only
  - short-request-first
  - greedy until budget exhausted

第一版保持旧的 static replay provider 为默认行为；feedback planner 先支持 dry-run，再逐步接管实际 replay plan。

---

### Current Repository State

#### 1. Step 边界在哪里
当前 step 边界在 `GpuModelRunner.execute_model()` 这一轮模型执行内定义：

- scheduler 输出在 `scheduler_output`
- step 内 request/token/block 元数据准备发生在：
  - `GpuModelRunner._prepare_inputs()`
  - `GpuModelRunner._prepare_layer_recompute_step_metadata()`
- dynamic replay runtime 构建发生在：
  - `GpuModelRunner._prepare_dynamic_replay_runtime()`
- forward 执行后，step 末尾同步发生在：
  - `GpuModelRunner._sync_runkv_step_end_state()`

因此，一次 step 的自然边界是：

1. `scheduler_output` 已确定本 step active batch
2. `_prepare_inputs()` 已生成 step 内的 scheduled token positions / req_indices / block table 视图
3. `_prepare_dynamic_replay_runtime()` 已生成 layer 0 plan
4. model forward 完成
5. step-end sync 完成

这也是 planner 做 step-level reinitialization 的最佳边界。

#### 2. Layer-wise 执行流程在哪里
当前 dynamic replay 的 layer 主循环在：

- `vllm/model_executor/models/opt.py`
- `OPTDecoder._forward_dynamic_replay()`

这一段逻辑当前已经具备：

- 每层读取 `runtime.get_layer_plan(layer_idx)`
- 按 plan 组装当前层 replay hidden states
- 将 replay token 与 scheduled token pack 成 `combined_hidden_states`
- 使用该层专属 attention metadata 跑一次统一的 layer forward
- 再把输出拆回：
  - `replay_hidden_states`
  - `scheduled_hidden_states`

因此，feedback planner 的 layer-wise update 插入点也应该围绕这条 loop，而不是另起一套执行通路。

#### 3. 当前 replay plan 是怎么生成和消费的
当前 replay plan 生成路径：

- `GpuModelRunner._build_dynamic_layer_plan()`
- `ReplayPlanProvider.get_layer_plan(...)`
- `compute_layer_replay_plan_for_layer(...)`

当前 replay plan 消费路径：

- `OPTDecoder._forward_dynamic_replay()`
- `GpuModelRunner._build_layer_attn_metadata()`
- `GpuModelRunner._runkv_pre_hook()`

现有行为是：

- step 开始前先生成 layer 0 plan
- 在 layer `i` 的 pre-hook 中提前生成 layer `i+1` plan
- 同时提前构建 layer `i+1` metadata
- pre-hook 还会据此启动 layer `i+1` 的 IO prefix KV 加载和 CPU fill H2D

因此，现有系统已经是“layer-wise planning + layer-wise consumption”，只是 planner 还是 static。

#### 4. 当前 replay plan 的表达方式
当前 replay plan 的主数据结构是 `LayerReplayPlan`。

它当前的核心字段包括：

- `kv_replay_start_per_req`
- `computed_lens_per_req`
- `prev_gpu_start_per_req`
- `cpu_fill_positions`
- `cpu_fill_logical_ids`
- `cpu_fill_block_offsets`
- `gpu_reuse_slice_per_req`
- `replay_token_count`
- `scheduled_token_count`
- `num_actual_tokens`
- `query_start_loc`
- `slot_mapping`
- `combined_replay_indices`
- `combined_scheduled_indices`

结论：

- 当前 replay plan 的核心控制语义是**按 request 的 replay frontier 表达**
- 它最终会被下沉成 token 级 pack / unpack 索引
- 它不是显式的 per-block allocator 输出
- 但由于 `kv_replay_start_per_req` 本身是 block 对齐后的 frontier，因此 replay 行为本质上仍是 block 级 suffix replay

#### 5. 当前 request / block / suffix / scheduled token / replay token 的数据结构
当前相关数据结构分层如下：

request / batch state：
- `CachedRequestState`
- `InputBatch`

scheduler step 输出：
- `SchedulerOutput`
- `num_scheduled_tokens: dict[str, int]`

block 视图：
- `BlockTable`
- `MultiGroupBlockTable`
- `num_blocks_per_row`
- `block_table.get_numpy_array()`

step 内 scheduled token 视图：
- `req_indices`
- `positions_np`

replay token 视图：
- `LayerReplayPlan.cpu_fill_positions`
- `LayerReplayPlan.gpu_reuse_slice_per_req`
- `LayerReplayPlan.combined_replay_indices`

结论：

- 当前系统同时有 request 级、token 级、block 级三层视图
- planner 最自然的接入方式不是新造第四套数据结构，而是：
  - planner 内部维护 budget / candidates / alloc
  - 输出仍回写到 `LayerReplayPlan`

#### 6. contiguous suffix 约束当前是否天然满足
当前是天然满足的。

原因是：

- `desired_replay_start_tokens` 最终会向下对齐到 block 边界
- replay 区间总是 `[kv_replay_start, computed_len)`

所以每个 request 的 replay 天然是 contiguous suffix。

但要注意一个结构性问题：

- 当前实际 IO/recompute 路径里，`skip_block_ids` 的来源并不是 `LayerReplayPlan`
- 而是 `layer_recompute_io_prefix_blocks`

所以“contiguous suffix 在 plan 中天然成立”并不等于“执行路径已经完全由 plan 接管”。

#### 7. 当前有哪些 profiling / timing 点
当前已有两类观测：

1. OPT component profiling
- `OPTComponentMFUStepProfiler`
- 用 CUDA event 测 attention / FFN 时间
- 已可拿到 per-layer replay ratio、tokens、MFU

2. RunKV trace / debug
- `record_function_or_nullcontext("runkv_recompute:*")`
- RunKV debug JSONL
- async H2D/D2H ready events
- KV checksum / step metadata

当前缺失的是 planner 语义下的成对 timing：

- `t_i_end`
- `t_{i+1}_ready`

也就是说，已有 trace 足够做 coarse profiling，但不够直接支持 signed imbalance feedback。

#### 8. 当前哪些地方已有 planner / scheduler / memory manager 雏形
已有 planner/provider 雏形：
- `ReplayPlanProvider`
- `StaticReplayPlanProvider`
- `RandomReplayPlanProvider`

已有 runtime 雏形：
- `OPTDynamicReplayRuntime`

已有 memory/materialization 雏形：
- `LayerRecomputeManager`

已有 batch scheduler truth source：
- `SchedulerOutput`
- `InputBatch`

因此，最小侵入方向是：

- 不改 scheduler 职责
- 不把 planner 塞进 `LayerRecomputeManager`
- 不重构 `GpuModelRunner` 主干
- 只把 provider 从 stateless static provider 升级为 stateful feedback provider

---

### Design Goals

#### Feature 1: Layer-wise signed imbalance feedback
反馈信号定义为：

- `d_i = t_{i+1,ready} - t_{i,end}`

语义：

- `d_i > 0`：下一层数据未 ready，IO 更慢，应增加 replay
- `d_i < 0`：下一层数据提前 ready，compute 更慢，应减少 replay
- `d_i ~= 0`：当前较平衡

MVP 目标：

- 先打通可靠 timing 观测
- 先支持 dry-run logging
- 后续再接入控制更新

#### Feature 2: 全局 replay control variable
MVP planner 只维护一个低维控制量：

- `global_replay_budget_blocks`

该 budget 的含义是：

- 当前层可分配的 replay block 总量上限

不做：
- per-request 全局最优解
- 复杂 block heap allocator
- cost-aware param

#### Feature 3: batch 内 replay blocks 的简单选择策略
MVP 固定规则：

- 只允许 contiguous suffix blocks
- 优先 short requests
- 从高优先级 candidate 开始 greedy 分配
- 直到 budget 用完

#### Feature 4: layer-wise feedback update
目标行为：

- 每层执行后更新 controller state
- 不是 step 开始前一次性算完整 static plan

但基于当前架构，第一版需要接受一个现实约束：

- 由于 layer `i+1` plan 通常已经在 layer `i` pre-hook 中生成并发起了预取
- layer `i` 的精确 feedback 很难无损地立即作用于 layer `i+1`
- 因此 MVP 保持 overlap，不强行做 `i -> i+1` 同层闭环
- 第一版采用：
  - 每层都更新 controller state
  - 影响最早尚未 materialize 的下一份 plan
  - 通常表现为一层滞后 actuation

#### Feature 5: step-level batch-aware reinitialization
新 step 开始时 batch 可能变化很大：

- req 退出
- req 加入
- 长度变化
- block 申请变化

因此 planner 不能机械继承上个 step 的最终状态，也不该每次从 0 冷启动。

MVP 需要：

- 感知 batch drift
- 对 budget 做 clamp / shrink / reset
- 对 gain/probe state 做衰减继承或清零

#### Feature 6: 局部线性近似
MVP 假设：

- replay budget -> imbalance 的关系局部单调、局部线性

第一版只做：

- deadband
- damping
- step-size clip
- 一个简单 local gain / secant 风格更新

不做：
- regime split
- per-regime gain cache
- 复杂 cost model

#### Feature 7: 可观测性
MVP 必须打通：

- 每层 replay budget
- 每层实际 replay blocks 总量
- 每层 per-request replay 分配
- 每层 signed imbalance
- 每层 controller update 前后值
- step reinit 行为
- batch drift / reset 决策

#### Feature 8: 渐进式集成
必须按阶段推进：

- A: 反馈观测 + 日志
- B: planner 状态 + dry-run
- C: planner 接管 replay 总量
- D: step reinit
- E: benchmark / validation

---

### Public / Interface Changes
- `RunKVOffloadConfig` 新增：
  - `layer_recompute_planner = "static" | "feedback"`
  - `layer_recompute_planner_dry_run: bool = False`
  - `layer_recompute_planner_debug: bool = False`
  - `layer_recompute_planner_debug_output_path: str | None = None`
- 默认值：
  - `layer_recompute_planner = "static"`
  - `layer_recompute_planner_dry_run = False`
  - `layer_recompute_planner_debug = False`
  - `layer_recompute_planner_debug_output_path = None`
- 不新增新的 replay mode；feedback planner 只在：
  - `layer_recompute_mode == "prev_layer_output_dynamic"`
  下生效
- 继续复用现有 `ObservabilityConfig` 中的 profiling 开关：
  - `enable_opt_component_mfu_profiling: bool = False`
  - `opt_component_mfu_output_path: str | None = None`
  - `opt_component_mfu_peak_tflops: float | None = None`

#### How To Enable
- 只开启 feedback planner、但不接管执行：
  - `layer_recompute_mode="prev_layer_output_dynamic"`
  - `layer_recompute_planner="feedback"`
  - `layer_recompute_planner_dry_run=True`
- 开启 planner debug 输出：
  - 在上述基础上再设置 `layer_recompute_planner_debug=True`
  - 若需要落盘，再设置 `layer_recompute_planner_debug_output_path`
- 开启 profiling 输出：
  - 设置 `enable_opt_component_mfu_profiling=True`
  - 设置 `opt_component_mfu_output_path`
  - 若要计算 MFU，再设置 `opt_component_mfu_peak_tflops`
- 默认行为：
  - 如果只开 planner、不打开 `layer_recompute_planner_debug`
  - 则 Step 7/8 的 imbalance / budget / controller update 只保存在内存态，不写日志或 JSONL

Python 配置示例：

```python
kv_offload_config = {
    "enabled": True,
    "enable_layer_recompute": True,
    "layer_recompute_mode": "prev_layer_output_dynamic",
    "layer_recompute_planner": "feedback",
    "layer_recompute_planner_dry_run": True,
    "layer_recompute_planner_debug": True,
    "layer_recompute_planner_debug_output_path": "/tmp/runkv_planner_debug.jsonl",
}

llm = LLM(
    ...,
    kv_offload_config=kv_offload_config,
    enable_opt_component_mfu_profiling=True,
    opt_component_mfu_output_path="/tmp/opt_component_mfu.jsonl",
    opt_component_mfu_peak_tflops=312.0,
)
```

新增接口：

```python
class FeedbackReplayPlanProvider(ReplayPlanProvider):
    def begin_step(...)
    def get_layer_plan(...)
    def observe_layer_feedback(...)
    def get_debug_snapshot(...)
```

---

### Implementation Changes

#### 1. provider 升级为 stateful feedback planner
新增一个 `FeedbackReplayPlanProvider`，挂在 `GpuModelRunner.replay_plan_provider` 上，跨 step 存活。

它维护的核心状态包括：

- `global_budget_blocks`
- `estimated_local_gain`
- `last_budget_blocks`
- `last_imbalance_ms`
- `probe_state`
- `reinit_generation`
- `step_batch_fingerprint`

职责边界：

- provider 负责：
  - planner state
  - batch drift 判定
  - budget update
  - budget -> per-request alloc
- runtime 负责：
  - 当前 step 内的 layer plans / metadata / timing
- memory manager 不负责 planner 决策

#### 2. 扩展 `LayerReplayPlan`
在现有 plan 基础上新增 block 级字段：

- `replay_blocks_per_req`
- `replay_block_count`
- `skip_logical_block_ids`
- `per_req_replay_block_ranges`

原因：

- 现有执行路径里，真正控制 KV load/recompute 的是 block 集合
- 若 planner 要 live 接管执行，plan 必须显式带出 block-level 结果

#### 3. 在 step 边界新增 planner begin_step hook
在 `_prepare_layer_recompute_step_metadata()` 后、`_prepare_dynamic_replay_runtime()` 前调用：

- `provider.begin_step(...)`

输入：

- active req ids
- computed lengths
- scheduled lengths
- logical block table
- num_blocks_per_row

begin_step 做的事：

- 计算 batch fingerprint
- 计算 replayable block 统计
- 做 drift 判定
- 决定：
  - warm start
  - shrink
  - hard reset
- 重建本 step candidates

#### 4. Signed imbalance instrumentation
新增两个 timing 点：

`t_i_end`
- 接到 `OPTDecoder._forward_dynamic_replay()` 的 layer loop 中
- 每层 `layer(...)` 返回后记录 CUDA event
- 这是 planner 需要的 compute boundary

`t_{i+1}_ready`
- 接到 `GpuModelRunner._runkv_pre_hook()` 中
- 在 `mapper.sync_load_layer(layer_idx)` 与 `manager.sync_cpu_fill_h2d(layer_idx)` 完成后记录 CUDA event
- 这是“下一层数据已经 ready”的精确边界

为什么选这两个点：

- 它们直接对应 planner 关心的 boundary
- 不依赖注意力 kernel 内部近似时间
- 与现有 async H2D / KV prefetch 语义一致

#### 5. feedback 计算与 runtime 缓存
在 runtime 中维护：

- per-layer `layer_end_event`
- per-layer `layer_ready_event`
- per-layer `signed_imbalance_ms`

当事件对齐后，计算：

- `signed_imbalance_ms = ready(i+1) - end(i)`

然后调用：

- `provider.observe_layer_feedback(layer_idx=i, imbalance_ms=...)`

第一版允许：

- 精确观测
- 一层滞后生效

不强行打断当前 overlap。

#### 6. budget -> per-request alloc
MVP 分配器规则固定为：

1. 候选单位是 request 的 suffix blocks
2. 每个 request 只允许分配一个 contiguous suffix
3. request 按可 replay block 数升序排序
4. 稳定 tie-break：
   - `req_idx`
5. greedy 分配直到 budget 用完

输出：

- `allocated_blocks_per_req`

再转换为：

- `kv_replay_start_per_req`

然后继续复用现有 `compute_layer_replay_plan_for_layer()` 生成 token 级执行 plan。

#### 7. dry-run 模式
dry-run 模式下：

- planner 正常维护 state
- 正常计算 next budget
- 正常计算 per-request alloc
- 正常计算 feedback / profiler 所需观测值
- 默认只保留内存态观测；仅在显式开启 debug / profiling 输出时写日志或 JSONL
- 但实际执行仍继续使用 static plan provider

这样可以先验证：

- imbalance 观测是否合理
- update 是否稳定
- budget 是否按预期变化
- step reinit 是否合理

#### 8. live takeover：让 planner 真正接管 replay 总量
这一步是 MVP 真正生效的关键。

当前结构问题是：

- `skip_block_ids` 不是从 `LayerReplayPlan` 来的
- 而是从 `layer_recompute_io_prefix_blocks` 推导来的

因此 live takeover 必须改以下路径：

- step 开始的 layer 0 prefetch
- dynamic pre-hook 中 current layer / next layer 的 skip set 计算
- prefetch inputs for recompute
- recompute KV 的 suffix block 范围

统一改为：

- 由 `LayerReplayPlan.skip_logical_block_ids` 或等价 block-range 结果驱动

只有这一步完成，budget 的变化才会真正影响实际 replay。

#### 9. step-level reinitialization
在 `provider.begin_step(...)` 中实现：

drift metrics：
- req-set overlap
- total replayable blocks delta

默认策略：

- drift 大：
  - hard reset
  - budget clamp
  - gain / probe / last_imbalance 清零
- drift 小：
  - warm start
  - budget clamp
  - gain / probe 衰减继承

reinit 时记录：

- `reset_reason`
- `old_budget`
- `new_budget`
- `old_gain`
- `new_gain`

#### 10. observability
默认运行模式：

- per-layer `signed_imbalance_ms`、budget update、reinit 决策只保存在 runtime / provider 内存态
- 这些观测值用于 live planner 控制闭环，但默认不做 per-layer 文件落盘

显式开启 debug / profiling 输出时，才记录：

- step begin:
  - batch fingerprint
  - drift score
  - reinit decision
- per layer:
  - `budget_before`
  - `budget_after`
  - `selected_replay_blocks`
  - `per_req_allocated_blocks`
  - `signed_imbalance_ms`
  - `controller_before`
  - `controller_after`

profiling / benchmark 输出记录：

- layer index
- replay budget
- actual replay block count
- actual replay token count
- per-request replay allocation
- signed imbalance
- controller update
- step reinit 信息

---

### Risks

#### 1. 当前 prehook 时序与 planner update 时机冲突
当前 layer `i+1` plan 在 layer `i` prehook 就生成并开始预取。

风险：
- layer `i` 的 feedback 来不及改写已经 materialize 的 layer `i+1`

MVP 处理：
- 接受一层滞后 actuation
- 保留当前 overlap
- 不在第一版强行改 prefetch 时序

#### 2. timing 点可能不准确
风险来源：

- 如果把 `t_i_end` 取在 attention profiler 或 FFN profiler 上，只是局部时间，不是整层边界
- 如果把 `t_{i+1}_ready` 取在 async launch 时刻，不代表真正 ready

MVP 处理：
- `t_i_end` 必须取在整层返回后
- `t_{i+1}_ready` 必须取在 sync 之后

#### 3. 当前 replay plan 数据结构对 block-level 执行控制不足
风险：
- 现有 plan 主要是 frontier + token indices
- executor/prehook 真正需要 block set

MVP 处理：
- 扩展 `LayerReplayPlan` 的 block-level 字段
- 不单靠 `kv_replay_start_per_req` 间接还原

#### 4. live takeover 若只改 plan、不改 skip-block 路径，行为不会生效
这是当前最关键的结构性问题。

MVP 处理：
- 明确把“plan-derived skip set 接管执行路径”作为单独阶段
- 在此之前只能叫 dry-run，不算 planner 生效

---

### Test Plan

#### 单元测试
1. budget -> per-request allocation
- budget = 0
- budget 小于最短 request replayable blocks
- budget 足够覆盖多个短 request
- budget 超过总 replayable blocks

2. contiguous suffix correctness
- 分配结果必须总是 suffix
- 不允许中间洞

3. short-request-first priority
- replayable blocks 更短的 request 必须先被满足

4. block-level / token-level plan consistency
- `allocated_blocks_per_req`
- `kv_replay_start_per_req`
- `replay_block_count`
- `replay_token_count`
- 必须一致

5. signed imbalance computation
- mock 事件对
- 验证正负号和数值方向

6. step reinit
- batch 基本不变
- batch 小幅变化
- batch 大幅变化
- 分别验证 warm start / clamp / hard reset

#### 集成测试
1. dry-run 模式
- planner 开启但不改执行
- 输出必须与 static provider 完全一致

2. live takeover 模式
- replay block 数量应随 budget 改变
- correctness 不回退

3. batch drift 场景
- req 加入/退出
- req 长度变化
- planner state 应按预期 shrink / reset

4. benchmark / profiler 输出
- 每层应能看到：
  - budget
  - actual replay blocks
  - per-request alloc
  - imbalance
  - controller update

---

### Assumptions / Defaults
- 第一版只覆盖现有 dynamic replay 支持范围
- 不引入 ILP / MIP / cost-aware allocator
- 不引入 per-regime gain cache
- 不重构 scheduler / memory manager 边界
- planner 默认关闭
- planner dry-run 优先于 live takeover
- 若 feedback planner 与 cudagraph/现有约束冲突，第一版直接禁用该组合

---

## Step-by-Step 实现方案（Agent 拆分）

以下按依赖顺序拆成 12 个 step。每个 step 都应是可独立提交、可独立验证的最小单元。

---

#### Step 1: 配置层 — 新增 planner 开关与 dry-run 开关
**目标**：引入 `layer_recompute_planner` 与 `layer_recompute_planner_dry_run`，不改变行为。  
**实现状态**：已完成  
**改动文件**：
- `vllm/v1/core/kv_cache_offload_config.py`
- `vllm/engine/arg_utils.py`
**实际改动**：
- `RunKVOffloadConfig` 已新增：
  - `layer_recompute_planner: Literal["static", "feedback"] = "static"`
  - `layer_recompute_planner_dry_run: bool = False`
- `EngineArgs` 已新增对应字段，并在 `_build_kv_offload_config()` 中完成下沉
- CLI 已新增：
  - `--runkv-layer-recompute-planner static|feedback`
  - `--runkv-layer-recompute-planner-dry-run`
- 新增 `_validate_runkv_layer_recompute_planner()` 校验
- 默认值保持 static / non-dry-run，当前运行时行为不变
**如何使用**：
- `--runkv-layer-recompute-planner static|feedback`
- `--runkv-layer-recompute-planner-dry-run`
**验证方式**：
- 默认行为完全不变
- feedback + dry-run 可通过配置下沉
- 已执行：
  - `python -m py_compile vllm/v1/core/kv_cache_offload_config.py vllm/engine/arg_utils.py`
**依赖**：无

---

#### Step 2: Provider 骨架 — 新增 `FeedbackReplayPlanProvider`
**目标**：定义 stateful feedback planner 骨架，但先不做任何控制逻辑。  
**实现状态**：已完成  
**改动文件**：
- `vllm/v1/worker/opt_dynamic_replay.py`
 - `vllm/v1/worker/gpu_model_runner.py`
**实际改动**：
- 已新增 class `FeedbackReplayPlanProvider`
- provider 当前维护的 state 已收敛为两层：
  - 跨 step 的 `FeedbackPlannerControllerState`
  - 当前 step 的 `FeedbackPlannerStepSummary`
- `FeedbackPlannerControllerState` 当前包含：
  - `global_budget_blocks`
  - `estimated_local_gain`
  - `last_budget_blocks`
  - `last_imbalance_ms`
  - `probe_state`
  - `reinit_generation`
  - `step_batch_fingerprint`
- `FeedbackPlannerStepSummary` 当前包含：
  - `replayable_blocks_per_req`
  - `total_replayable_blocks`
- 不再保存 raw step metadata；这部分仍由 `LayerRecomputeManager` 持有
- 已补充注释，明确这两个数据结构分别负责跨 step controller state 与当前 step 派生摘要
- `dry_run`
- 已新增方法：
  - `begin_step(...)`
  - `observe_layer_feedback(...)`
  - `get_debug_snapshot(...)`
- `get_layer_plan(...)` 当前直接委托给内部 `StaticReplayPlanProvider`
- `GpuModelRunner` 已按 `layer_recompute_planner` 选择 provider：
  - `static` -> `StaticReplayPlanProvider`
  - `feedback` -> `FeedbackReplayPlanProvider`
- `GpuModelRunner.replay_plan_provider` 类型已放宽为 `ReplayPlanProvider | None`
- 当前不引入任何控制逻辑，执行行为保持不变
**验证方式**：
- provider 可被 `GpuModelRunner` 持有
- 可跨 step 存活
- 已执行：
  - `python -m py_compile vllm/v1/worker/opt_dynamic_replay.py vllm/v1/worker/gpu_model_runner.py`
**依赖**：Step 1

---

#### Step 3: 扩展 `LayerReplayPlan` 为 block-aware
**目标**：让 replay plan 显式带出 block 级决策结果。  
**实现状态**：已完成  
**改动文件**：
- `vllm/v1/worker/opt_dynamic_replay.py`
**实际改动**：
- `LayerReplayPlan` 已新增字段：
  - `replay_blocks_per_req`
  - `replay_block_count`
  - `skip_logical_block_ids`
  - `per_req_replay_block_ranges`
- `compute_layer_replay_plan_for_layer()` 已同步计算并填充这些 block-level 字段
- 当前字段语义为：
  - `per_req_replay_block_ranges`：每个 request 的 replay block 区间 `[start_block, end_block)`
  - `replay_blocks_per_req`：每个 request 的 replay block 数
  - `replay_block_count`：batch 内 replay block 总数
  - `skip_logical_block_ids`：当前 plan 对应的 replay suffix logical block id 集合
- `StaticReplayPlanProvider` 与 `FeedbackReplayPlanProvider` 都通过同一条 plan 构造路径拿到这些字段
**验证方式**：
- static plan 也能正确生成 block-level 结果
- 已执行：
  - `python -m py_compile vllm/v1/worker/opt_dynamic_replay.py`
**依赖**：Step 2

---

#### Step 4: step 边界 hook — planner begin_step 接入
**目标**：在 step 开始时获取完整 active batch truth。  
**实现状态**：已完成  
**改动文件**：
- `vllm/v1/worker/gpu_model_runner.py`
 - `vllm/v1/worker/opt_dynamic_replay.py`
**实际改动**：
- `ReplayPlanProvider` 已扩展出 `begin_step(...)` 接口
- `StaticReplayPlanProvider` 与 `RandomReplayPlanProvider` 已提供 no-op `begin_step(...)`
- `FeedbackReplayPlanProvider.begin_step(...)` 不再保存 raw metadata
- 它现在只基于 step 边界输入派生并更新：
  - `replayable_blocks_per_req`
  - `total_replayable_blocks`
  - `step_batch_fingerprint`
- 在 `_prepare_layer_recompute_step_metadata()` 中，`layer_recompute_manager.begin_step(...)` 之后已调用 `replay_plan_provider.begin_step(...)`
- 当前传入字段包括：
  - `req_ids`
  - `computed_lens`
  - `scheduled_lens`
  - `num_blocks_per_row`
  - `block_size`
**验证方式**：
- 每个 step 都能得到 batch fingerprint
- 不改变 replay 行为
- 已执行：
  - `python -m py_compile vllm/v1/worker/opt_dynamic_replay.py vllm/v1/worker/gpu_model_runner.py`
**依赖**：Step 2

---

#### Step 5: instrumentation A — 接入 `t_i_end`
**目标**：增加可靠的 layer end timing。  
**实现状态**：已完成  
**改动文件**：
- `vllm/model_executor/models/opt.py`
- `vllm/v1/worker/opt_dynamic_replay.py`
**实际改动**：
- `OPTDynamicReplayRuntime` 已新增 per-layer `layer_end_event` 缓存
- 已新增：
  - `set_layer_end_event(...)`
  - `get_layer_end_event(...)`
- 在 `OPTDecoder._forward_dynamic_replay()` 中，每层 layer forward 完成后都会记录一个 CUDA event
- 该事件只在 CUDA 张量路径上创建；非 CUDA 情况下写入 `None`
- 已补充注释，明确这个事件后续会和 next-layer-ready event 配对，用于计算 signed imbalance
**验证方式**：
- event 可成功记录
- 对模型输出无影响
- 已执行：
  - `python -m py_compile vllm/v1/worker/opt_dynamic_replay.py vllm/model_executor/models/opt.py`
**依赖**：Step 2

---

#### Step 6: instrumentation B — 接入 `t_{i+1}_ready`
**目标**：增加可靠的 next-layer-ready timing。  
**实现状态**：已完成  
**改动文件**：
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/opt_dynamic_replay.py`
**实际改动**：
- `OPTDynamicReplayRuntime` 已新增 per-layer `layer_ready_event` 缓存
- 已新增：
  - `set_layer_ready_event(...)`
  - `get_layer_ready_event(...)`
- 在 dynamic replay pre-hook 中，`mapper.sync_load_layer(layer_idx)` 和 `manager.sync_cpu_fill_h2d(layer_idx)` 完成后记录一个 CUDA event
- 该事件只在 CUDA 路径上创建；非 CUDA 情况下写入 `None`
- 已补充注释，明确这个事件表示“当前层 IO prefix load 和可选 CPU-fill H2D 都已 ready”的边界
**验证方式**：
- ready 事件与真实 sync 边界一致
- 已执行：
  - `python -m py_compile vllm/v1/worker/opt_dynamic_replay.py vllm/v1/worker/gpu_model_runner.py`
**依赖**：Step 5

---

#### Step 7: 阶段 A — signed imbalance 观测与日志
**目标**：先只打通 feedback 观测，不改 replay 行为。  
**实现状态**：规划中  
**改动文件**：
- `vllm/v1/worker/opt_dynamic_replay.py`
- `vllm/v1/profiling/opt_component_mfu.py`
- 可选 `vllm/v1/worker/runkv_debug.py`
**实际改动**：
- 根据事件对计算 `signed_imbalance_ms`
- provider 接收 per-layer imbalance feedback，并把结果保存在内存态 runtime/planner state
- 仅在 `layer_recompute_planner_debug=True` 或开启 profiling 输出时，才把每层 imbalance 写到 log / JSONL
- provider 可接收 feedback，但不改变实际 plan
**验证方式**：
- dry-run 情况下输出完全不变
- 默认模式下无额外 per-layer 文件写入
- debug / profiling 模式下日志中能看到每层 imbalance
**依赖**：Step 6

---

#### Step 8: 阶段 B — planner state dry-run 更新
**目标**：引入 budget / gain / probe / reinit 状态，但先只读不生效。  
**实现状态**：规划中  
**改动文件**：
- `vllm/v1/worker/opt_dynamic_replay.py`
**实际改动**：
- 维护：
  - `global_budget_blocks`
  - `estimated_local_gain`
  - `probe_state`
  - `last_imbalance_ms`
- 每层 feedback 后计算“建议 budget”
- 默认只更新内存态 planner state；仅在 `layer_recompute_planner_debug=True` 或开启 profiling 输出时打印或写 profiler，不改变执行 plan
**验证方式**：
- budget 可逐层变化
- 现有执行输出保持不变
**依赖**：Step 7

---

#### Step 9: 阶段 C.1 — budget -> per-request allocation
**目标**：实现 short-request-first + contiguous suffix 的 batch 内分配器。  
**实现状态**：规划中  
**改动文件**：
- `vllm/v1/worker/opt_dynamic_replay.py`
**实际改动**：
- 基于本 step candidates 计算：
  - `allocated_blocks_per_req`
- 转成：
  - `kv_replay_start_per_req`
- 最后构造新的 `LayerReplayPlan`
**验证方式**：
- 分配规则稳定、确定
- contiguous suffix 恒成立
**依赖**：Step 8

---

#### Step 10: 阶段 C.2 — planner 真接管执行路径
**目标**：让 budget 真正影响实际 replay。  
**实现状态**：规划中  
**改动文件**：
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/layer_recompute.py`
**实际改动**：
- 所有 dynamic replay 路径中的 `skip_block_ids` 改为从 `LayerReplayPlan` 派生
- 不再只依赖静态 `layer_recompute_io_prefix_blocks`
**验证方式**：
- replay block 总量会随 planner budget 真正变化
- correctness 不变
**依赖**：Step 9

---

#### Step 11: 阶段 D — step-level reinitialization 生效
**目标**：实现 batch-aware warm start / shrink / reset。  
**实现状态**：规划中  
**改动文件**：
- `vllm/v1/worker/opt_dynamic_replay.py`
- `vllm/v1/worker/gpu_model_runner.py`
**实际改动**：
- 在 `begin_step()` 中加入 drift 判定
- 预算 / gain / probe 按策略重置或衰减继承
- 记录 reinit 决策日志
**验证方式**：
- batch 变化大时状态重置
- batch 变化小时状态平滑继承
**依赖**：Step 10

---

#### Step 12: 阶段 E — benchmark / trace / regression
**目标**：把 planner 接入现有 benchmark 和 regression。  
**实现状态**：规划中  
**改动文件**：
- `examples/offline_inference/opt_replay_component_mfu.py`
- `vllm/v1/profiling/opt_component_mfu.py`
- `tests/v1/kv_offload/test_opt_dynamic_replay_plan.py`
- `tests/v1/kv_offload/test_opt_dynamic_replay_metadata.py`
- `tests/v1/kv_offload/test_opt_dynamic_replay_forward.py`
- `tests/v1/kv_offload/test_dynamic_replay_e2e_concurrent.py`
**实际改动**：
- benchmark 输出 planner 字段
- 新增 dry-run / live planner 测试
- 验证 budget 变化、imbalance 收敛、correctness 不回退
**验证方式**：
- planner on/off 对比可观测
- regression 全过
**依赖**：Step 11

---

### MVP Definition

第一版真正落地后，系统至少应具备以下能力：

必须已经生效：
- 每层 signed imbalance 可观测
- planner 维护全局 replay budget
- planner 能把 budget 转成实际 per-request replay plan
- 实际执行路径由 plan-derived block skip/load/recompute 驱动
- step 边界支持 batch-aware reinit
- planner 指标可通过日志 / profiling 观察

第一版先只记录、不生效：
- 更复杂的 controller 形式
- per-regime gain cache
- source-aware 细粒度策略
- compute-equivalent budget 重参数化
- 任何 ILP / MIP / 全局优化器

---

### 落地顺序建议
推荐真实实施顺序保持为：

1. Step 1
2. Step 2
3. Step 3
4. Step 4
5. Step 5
6. Step 6
7. Step 7
8. Step 8
9. Step 9
10. Step 10
11. Step 11
12. Step 12

原因：

- 先把 feedback 观测打准
- 再做控制状态 dry-run
- 最后才接管实际执行路径
- 把风险最大的“plan 真接管 skip-block 路径”放到观测和 dry-run 之后
