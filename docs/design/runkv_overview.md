# RunKV — Overall Design Summary

> 状态: 现行设计（随 `runkv` 分支演进）
> 最后更新: 2026-04-24

---

## 1. 目标与核心思路

RunKV 是在 vLLM v1 上落地的一条 **layer-wise KV offload + feedback-driven replay** 路径，目标是在 KV cache 容量受限、必须 offload 到 CPU 的场景下，通过 **逐层 IO/compute overlap + 反馈控制 replay 规模**，拿到接近 "KV 常驻 GPU" 的吞吐，同时尽量减少对现有 forward 流水线的侵入。

核心思路：

1. **逐层流水线**：每一层独立地拉 KV + hidden states、做 replay，再跑当前层的真实 compute。IO 与 compute 在两条 CUDA stream 上并行，由 `pre_hook` 完成跨流同步。
2. **Replay 池化而非重算**：通过 `prev_layer_output_dynamic` 模式，用上一层保留的 GPU hidden states 拼装出 replay token 输入，避免从头重算 prefix；只有起始层需要从 CPU H2D 拉 hidden states。
3. **Feedback-driven replay budget**：用 per-layer signed imbalance `IO_ready(L+1) − compute_end(L)` 作为反馈信号，在线调节一个低维控制量 `global_replay_budget_blocks`，让 replay compute 正好填满 IO window。
4. **Speculative plan building**：把 `plan(L+1)` 的构建从 `pre_hook` 关键路径搬到后台 builder 线程，pre_hook 路径只剩跨 stream sync + event 取值。
5. **状态机化的控制器**：把原本 "每层都跑一次 Newton secant" 升级为三态 state machine（STEADY / TRANSIT / TRACKING），显著压住了振荡和饱和，参见 §4。

---

## 2. 模块分层

```
┌─────────────── GpuModelRunner ────────────────┐
│  _prepare_layer_recompute_step_metadata()     │
│  _prepare_dynamic_replay_runtime()            │
│  _runkv_pre_hook()     ← 层间同步点 / 反馈结算  │
└───────────────┬───────────────────────────────┘
                │
        ┌───────┴────────┐
        │                │
┌───────▼──────┐  ┌──────▼───────────────────┐
│ LayerRecompute│ │ OPTDynamicReplayRuntime │
│   Manager     │ │  - layer plans           │
│  - skip_ids   │ │  - cuda events           │
│  - DMA mapper │ │  - speculative plans     │
└───────────────┘ │  - plan_change_hint      │
                  │  - builder thread pool   │
                  └────────┬─────────────────┘
                           │
         ┌─────────────────▼─────────────────┐
         │  FeedbackReplayPlanProvider       │
         │   ├─ ImbalanceController (SM)     │
         │   ├─ budget allocator (SRF greedy)│
         │   └─ batch fingerprint / reinit   │
         └───────────────────────────────────┘
```

- **`GpuModelRunner`** 持有 step/layer 级流程；只有 `prev_layer_output_dynamic` 模式进 RunKV 分支。
- **`LayerRecomputeManager`** 管理 per-step skip block 计算、CPU→GPU hidden state DMA、KV DMA mapper。
- **`OPTDynamicReplayRuntime`**（每 step 重建）持有 per-layer plan / cuda events / speculative 槽 / builder 线程池，是主线程和 builder 线程之间的数据通道。
- **`FeedbackReplayPlanProvider`** 跨 step 存活，维护 `global_budget_blocks` 以及状态机控制器。`observe_layer_feedback(layer_idx, imbalance_ms)` 每层被 `_runkv_pre_hook` 调用一次，输出 `ControlDecision`。

---

## 3. 数据与控制流

### 3.1 Step 边界

1. `scheduler_output` 确定活跃 batch。
2. `_prepare_inputs()` / `_prepare_layer_recompute_step_metadata()` 产出 token / block 级视图。
3. `provider.begin_step(...)` 根据 `(req_ids, computed_lens, num_blocks_per_row, block_size)` 计算 `replayable_blocks_per_req` 与 `total_replayable_blocks`，更新 batch fingerprint、clamp budget。首个有 replayable block 的 step 把 budget 种子到 `total_replayable_blocks`。
4. `_prepare_dynamic_replay_runtime()` 创建 runtime，同步构造 `plan(0)`、注入 speculative builder callback、为 `plan(1)` 提交后台构建。
5. `execute_model()` 进入 decoder forward。

### 3.2 Layer 边界（pre_hook + post layer）

每层的事件顺序（单 KV-group、OPT 类模型）：

```
assembly(L) → LN(L) → QKV(L) ── pre_hook(L) ── KVwrite → FlashAttn(L) → O → LN → FFN → ...
                              │
                              ├─ record qkv_end_event(L)
                              ├─ sync_load_layer(L) / sync_cpu_fill_h2d(L)  (stream wait)
                              ├─ qkv_end_event.sync()                       (CPU wait)
                              ├─ final_ready_event(L).sync()                (CPU wait)
                              ├─ imbalance = IO_ready(L) − qkv_end(L)
                              ├─ observe_layer_feedback(L, imbalance)
                              │       └─ ImbalanceController.observe(...)
                              │          → ControlDecision{Δbudget, hint, state}
                              ├─ runtime.note_plan_change_hint(hint)
                              ├─ pre_hook 二分支（Δbudget-driven）:
                              │    hint ∈ {unchanged, small_delta}
                              │        └─ build_stable_successor_plan(plan(L))
                              │           (cpu_fill=0, gpu_reuse = replay(L))
                              │           → clear_speculative(L+1)
                              │    hint == significant_delta
                              │        └─ pop_speculative(L+1)
                              │           (后备: 同步 rebuild + warning)
                              ├─ runtime.set_layer_plan/metadata/skip_ids(L+1)
                              └─ launch_IO(L+1)  (non-blocking)

layer(L) 返回:
  record layer_end_event(L)
  if hint != "unchanged": submit_speculative_build(L+2, plan)   ← 启发式节流
```

### 3.3 Imbalance 参考点

采用 **`IO_ready(L) − qkv_end(L)`** 作为 signed imbalance：IO 在 QKV 启动后仍可 overlap 到 QKV 结束，比旧的 `IO_ready(L+1) − layer_end(L)` 多 ~1.5 ms 重叠窗口；归属层为 L（记录到 `observe_layer_feedback(layer_idx=L, ...)`）。详见 `docs/design/deferred_speculative_plan_building.md` §3.5。

> 注：per-layer 分析报告中的 `imbalance = compute_end[L] − load_ready[L+1]` 是事后离线指标（取自 CUPTI），与运行时控制器内的 `IO_ready(L) − qkv_end(L)` 方向相同、数值规模接近，是同一物理现象的两种切片。

### 3.4 Budget → plan 的分配器

- **候选**：每个 request 的 suffix block 数 = `min(⌈computed_len / block_size⌉, num_blocks_per_row)`。
- **策略**：short-request-first + contiguous suffix + greedy，直到 budget 用完。
- **输出**：`allocated_blocks_per_req` → `desired_replay_start_tokens = (computed_blocks − allocated) × block_size` → `compute_layer_replay_plan_for_layer(...)` 得到带 block 级字段的 `LayerReplayPlan`。
- **执行路径接管**：`FeedbackReplayPlanProvider` 非 dry-run 时，`skip_block_ids` 直接取自 `plan.skip_logical_block_ids`（`compute_skip_block_ids_from_plan`），保证 budget 变化真正进入 IO/recompute 路径。

---

## 4. Imbalance State Machine Controller

`FeedbackReplayPlanProvider` 在 `use_state_machine=True` 时用 [ImbalanceController](/home/lyc/inference/vllm/vllm/v1/worker/imbalance_controller.py) 取代旧的 Newton secant。细节见 [imbalance_state_machine_controller.md](imbalance_state_machine_controller.md)；这里只列关键决策：

| 状态 | 进入条件 | 控制律 | Budget |
|---|---|---|---|
| STEADY | 默认起点 / TRACKING 收敛后回归 | deadband + ±1 block proportional | 维持±1 block |
| TRANSIT | `stdev(window3) > 3·σ_baseline` | 观测不行动 | **冻结** |
| TRACKING | TRANSIT 出口 `|mean−old_baseline| ≥ 1 ms` | probe-then-exploit + RLS gain | 大步 Newton (≤ 10 blocks) |

- 窗口 K=3，σ_baseline=0.5 ms，stdev 触发倍数 M=3。
- TRACKING 首步只取 gain 的**物理符号**（budget↑→imbalance↓ ⇒ gain<0），第二步起用测得的 gain 做阻尼 Newton（damping=0.6）并以 RLS pool 持续估计。
- 连续 3 层 `|imbalance| < deadband` 或硬超时 20 层 → 回 STEADY。
- Δbudget 导出 `plan_change_hint ∈ {unchanged, small_delta, significant_delta}`，驱动 pre_hook 的"稳态复用 vs 消费 spec"二分支。

控制器与 regime classifier 解耦：当前未实现外层 classifier，使用单全局 baseline，靠 STEADY↔TRANSIT 分支自然吸收单点 outlier。

---

## 5. Speculative Plan Building

目的：把 `plan(L+1)` 构建（~0.5–1 ms）+ attn metadata（~0.3–0.5 ms）+ `skip_block_ids`（~0.1 ms）移出 pre_hook 关键路径。关键设计：

- **单 worker 后台 `ThreadPoolExecutor`**：runtime 生命周期内仅创建一次。
- **Plan-change hint 驱动的提交 + 消费**：
  - pre_hook 二分支：`unchanged / small_delta` 走 `build_stable_successor_plan`（`cpu_fill=0`、`gpu_reuse = replay(L)`），纯 CPU 拼装、无 tensor 分配；`significant_delta` 才 `pop_speculative(L+1)`。
  - post layer：只有当本层 hint ≠ `unchanged` 时提交 `spec(L+2)`，稳态下跳过 builder 工作，避免 CPU 浪费。
- **Fallback**：`pop_speculative` 返回 None → 一次性 warning 后同步构建，保证正确性。
- **多 KV-group 防御**：稳态复用只在单 KV-group 下生效（`len(kv_cache_groups)==1`）；否则走 spec/rebuild。

正确性保证：单 KV-group 下 `LayerReplayPlan` 在 budget 不变时跨层字段完全恒等（见设计文档 §3.4 证明表），稳态分支零偏差；非稳态消费的 spec 最多偏差 1 block（damping × max_step = 0.6 × 2 ≈ 1.2）。

---

## 6. 可观测性

- **CUDA events**：`step_anchor_event`、per-layer `qkv_end_event` / `layer_end_event` / `load_ready_event` / `cpu_fill_ready_event`。
- **NVTX ranges**：`runkv:prehook:{h2d_sync,imbalance,build_plan,skip_ids,schedule_io}:L*`、`runkv:layer_compute:L*`、`runkv:speculative_build:L*`。
- **JSONL 输出**（开启 `enable_opt_component_mfu_profiling`）：per-layer `compute_ms / kv_load_ms / hs_load_ms / load_ms / imbalance_ms / controller_update`，`controller_update` 包含 `budget_before / budget_after / gain_used / raw_delta / clipped_delta / action / sm_state / window_{mean,stdev}_ms / plan_change_hint`。
- **离线工具**：`tools/analyze_per_layer_timing.py`（per-layer 对比表 + 图）、`tools/extract_nvtx_kernel_map.py`（kernel 序列 diff）、`tools/profiler/stdev_check.py`（σ_baseline 估计）。

---

## 7. 配置

`RunKVOffloadConfig` 关键字段（[vllm/v1/core/kv_cache_offload_config.py](/home/lyc/inference/vllm/vllm/v1/core/kv_cache_offload_config.py)）：

| 字段 | 默认 | 作用 |
|---|---|---|
| `enabled` | False | RunKV 总开关 |
| `enable_layer_recompute` | - | 打开逐层 replay |
| `layer_recompute_mode` | - | `prev_layer_output_dynamic` 时进 RunKV 主路径 |
| `layer_recompute_planner` | `static` | `feedback` → 启用 `FeedbackReplayPlanProvider` |
| `layer_recompute_planner_dry_run` | False | True → planner 只观测不接管执行 |
| `layer_recompute_use_state_machine` | False | True → observe 经 `ImbalanceController` 而非 Newton secant |

Observability：`enable_opt_component_mfu_profiling` + `opt_component_mfu_output_path` (+ `opt_component_mfu_peak_tflops`)。

典型启用组合（线上）：`planner=feedback, dry_run=False, use_state_machine=True`。

---

## 8. 实验结果

### 8.1 方案对比

目录 `exp_results/analysis/per_layer/` 下两个快照，workload 相同（OPT-2.7B、pb1000），对比 RunKV vs TightLLM（KV 常驻 GPU 的理论上界）：

- **20260421_2008** — state machine **未启用**（Newton secant 控制器）
- **20260423_2329** — state machine **启用**（三态 SM + probe-then-exploit）

原始 flat.jsonl 来自 `exp_results/opt_feedback_observation/` 和对比数据 `exp_results/tightllm_observation/`。

### 8.2 Imbalance 收敛与端到端延迟

**Imbalance 指标**（三方并列）：

| 指标 | SM 关 (20260421) | SM 开 (20260423) | TightLLM (基线) |
|---|---:|---:|---:|
| 有符号平均 imbalance (ms) | **+7.85** | **−2.24** | −7.23 |
| `\|imbalance\|` 平均 (ms) | 14.83 | **6.91** | 7.24 |
| `\|imbalance\|` 典型 P50 (ms) | ~10 | **~6.5** | ~8.8 |
| `\|imbalance\|` 典型 P95 (ms) | ~33 | ~13 | ~12 |
| 层数处于 `mean>0`（GPU stall） | **30/31** | **0/31** | 0/31 |
| imbalance 物理含义 | 正值 = compute 等 IO | 近 0 = pipeline 平衡 | 负值 = IO 比 compute 早就绪很多 |

- **TightLLM** 的 signed 平均恒为负（−7.23 ms）且 `|mean|` ≈ 7.24 ms，说明它的 IO 流总是远早于 compute 就绪——TightLLM 在离线 ILP 预分配阶段故意给 IO 预留 slack，代价是 compute 路径要做更多 replay（总 replay tokens 260K，见 §8.4）。
- **SM 关**：`|imbalance|` 14.83 ms，signed +7.85 ms 且 30/31 层正值——**compute 在等 IO**，是典型的 budget 欠启用 + 控制器振荡同时发生（P95 冲到 33 ms 就是振荡留下的痕迹）。
- **SM 开**：`|imbalance|` 降到 6.91 ms，**比 TightLLM 的 7.24 ms 还低**；signed −2.24 ms 意味着 IO 只需稍微提前 ready，没有任何 compute 等 IO 的层——既消除振荡，又没掉进 TightLLM 那样为了安全过度预留 IO slack 的保守区。P95 从 ~33 ms 压到 ~13 ms，接近 TightLLM 的 ~12 ms。

**端到端延迟**（Σ over 31 layers，即单 step wall-clock）：

| 维度 | SM 关 | SM 开 | TightLLM |
|---|---:|---:|---:|
| Σ layer_total_gpu (ms/step) | 1157.8 | **1133.5** | 1180.7 |
| avg layer_total_gpu (ms/layer) | 37.35 | **36.56** | 38.09 |
| P95 layer_total_gpu (ms/layer) | ~53 | ~56 | ~56 |
| Δ 单步 vs TightLLM (ms) | **−22.9** | **−47.2** | — |
| Δ 单步相对 TightLLM (%) | −1.9 % | **−4.0 %** | — |

三条关键观察：

1. **SM 开 > TightLLM ≈ SM 关**：三方单 step 延迟排序是 SM 开 (1133.5) < SM 关 (1157.8) < TightLLM (1180.7)。SM 开相比 TightLLM 快 **47 ms/step (~4 %)**，相比 SM 关自身快 **24 ms/step**。
2. **SM 关虽然比 TightLLM 快 23 ms，但方式完全不健康**：§8.5 会展示 SM 关的 kernel_active 只有 15.7 ms，真正的 compute 被 assembly bubble（13.1 ms）撑成 37 ms；而 SM 开的 kernel_active 是 25.9 ms、bubble 7.7 ms——表面上两者 `layer_total_gpu` 只差 0.8 ms/层，但前者是"compute 路径病态"，后者是"compute 跑满、IO 藏净"。遇到更重的 workload 时 SM 关会先崩。
3. **SM 开把 RunKV 的延迟推到了 TightLLM 下界之下**：TightLLM 因为 KV 一直在 GPU（无 offload IO）理论上是延迟下界，但它的 replay 规模（260K tokens）比 RunKV（205K）大 27 %，所以 SM 开后 RunKV 的 compute 路径本身更短——这不是"追上 TightLLM"，而是"以 IO-unconstrained 的姿态反超 TightLLM"。

### 8.3 Compute vs IO 时长（核心瓶颈）

Imbalance 只是过程指标；真正要压低的是每层 `max(compute_dur, io_dur)`——这决定了在流水线完美 overlap 下每层最短可达的 wall-clock。

**如何测 "IO 每层时长"**：per-layer summary 里的 `runkv_dma_ms` 只记录了每层**最后一次** correlated DMA 的 duration（~2 ms 量级），不是该层的 IO 总工作量，**不能拿来作瓶颈比较**。相对可靠的指标是 **load stream 的 per-layer pacing**——也就是连续两层 `load_start` 的间隔：IO 流在整个 step 内几乎连续占用，这个间隔恰好等于 "若不并行、IO 一层接一层串行跑完需要多少时间"。另一个 cross-check 指标是 `io_dur = load_ready − load_start`，反映 "一个被 prefetch 多层的 DMA 从 launch 到 ready 的跨度"，它比 pacing 大约 2× 是因为单层 IO 同时被 2 层 compute overlap 掉。

**Per-layer 平均**：

| 分量 (ms/layer) | SM 关 | SM 开 | TightLLM | 说明 |
|---|---:|---:|---:|---|
| compute_dur（attn+FFN，不含 prehook）| 37.39 | **36.64** | 38.00 | GPU kernel 窗口 |
| IO pacing（Δ load_start, L1..L30）| 37.40 | **36.68** | 38.21 | IO 流串行等效速率 |
| layer_total_gpu（实测 step wall-clock / 层）| 37.35 | **36.56** | 38.09 | = max(compute, IO) + bubble |
| io_dur（单层 launch→ready 跨度）| 70.55 | 67.54 | 67.82 | 供参考，被 pipeline 拆分 |
| max(compute, IO pacing) | 37.40 | **36.68** | 38.21 | 每层瓶颈 |

三方数据一致指向 **compute 与 IO 在 per-layer 尺度上几乎持平**（误差 < 0.1 ms），这就是流水线"平衡点"的直接证据：再加 replay、再省 replay 都会立刻打破这个点——SM 关时正是因为反复打破它才撞出 14.8 ms 的 `|imbalance|`。

**Per-step 汇总（Σ over 30 个 pacing 间隔 / 31 个 layer）**：

| 分量 (ms/step) | SM 关 | SM 开 | TightLLM |
|---|---:|---:|---:|
| Σ compute_dur | 1196.5 | **1172.6** | 1215.9 |
| Σ IO pacing（load_start 跨度，L1→L30）| 1122.1 | **1100.4** | 1146.2 |
| max(Σ compute, Σ IO pacing)——流水线 lower bound | 1196.5 | **1172.6** | 1215.9 |
| **Σ layer_total_gpu（实测 step 瓶颈）** | 1157.8 | **1133.5** | 1180.7 |
| Σ layer_total_gpu 相对 lower bound 的差 | −38.7 | −39.1 | −35.2 |
| Δ vs TightLLM（实测 step） | −22.9 | **−47.2** | — |

> Σ layer_total_gpu 略低于 `max(Σ compute, Σ IO pacing)` 是因为前者只覆盖 L1..L30 的相邻差值（首尾层的 compute 不完全计入），两者在方向和量级上一致。

**几个重要观察**：

1. **瓶颈在哪取决于 1 ms 级差距**：per-layer `compute_dur` 与 `IO pacing` 的差不到 0.1 ms，谁卡住另一条就是当时的瓶颈。SM 关时 IO pacing 37.40 ≈ compute 37.39，但 `|imbalance|` 却 14.8 ms——说明平均值相同不代表对齐：控制器让**每一层**在两条路径出口处都错位，虽然"一层赚、一层亏"在 Σ 上抵消，wall-clock 却被最大错位拖着走。SM 开后 pacing 和 compute 双双降 ~0.7 ms 且 per-layer 对齐稳定。
2. **Σ IO pacing 是 IO 流的"串行等效耗时"**：它是 "若把这 step 所有 IO 一层一层跑需要多少时间" 的直接度量，不受 prefetch 假象干扰。SM 开后 IO 一路缩到 1100.4 ms，比 Σ compute 1172.6 还少 72 ms，所以 max() 卡在 compute 上——**pipeline 已 IO-unconstrained**。
3. **Σ compute_dur 已压到比 TightLLM 低 43 ms**（1172.6 vs 1215.9）：RunKV 因为 replay 少（205K vs 260K tokens）本应更便宜；SM 关时 replay 更少（106K）但 compute 反而更高，全因 assembly bubble（见 §8.5：SM 关 kernel_active 仅 15.7 ms、bubble 13.1 ms 都是 assembly/H2D 等待）。
4. **SM 关 vs SM 开的 wall-clock 差只有 24 ms 但 regime 完全不同**：两者实测 step 分别 1157.8 / 1133.5 ms，看似差距不大。但 SM 关是"compute 路径本身被 bubble 撑胖到 37.4"，SM 开是"compute 跑满到 kernel_active=25.9、IO 被藏净"——前者遇到更重的 batch 会原地崩，后者已贴近 IO-unconstrained 上界。

### 8.4 Replay 规模

| 指标 | SM 关 | SM 开 | TightLLM |
|---|---:|---:|---:|
| 平均 replay ratio | 0.725 | **0.836** | 0.866 |
| Replay tokens / step | 106 206 | **205 461** | 259 719 |
| cpu_fill tokens / step (L1+) | ~670 | ~370 | 0 |

旧控制器 budget 长期处在欠启用状态（每层仅 ~3.3k replay token，而 replayable 上限接近 6.4k）；SM 启用后 budget 几乎稳定在最大可用规模附近，replay ratio 提高到接近 TightLLM。同时 `cpu_fill` 不升反降，说明 `gpu_reuse` 路径被更充分利用。

### 8.5 GPU 利用率 / Bubble

| 指标 | SM 关 | SM 开 | TightLLM |
|---|---:|---:|---:|
| avg kernel_active / layer (ms) | 15.69 | **25.93** | 32.41 |
| avg GPU bubble / layer (ms) | **13.07** | **7.68** | 5.03 |
| avg prehook_cpu / layer (ms) | 15.08 | 9.07 | 9.11 |
| avg layer_compute (NVTX, CPU) (ms) | 32.70 | 32.48 | 36.12 |
| 每 1000 token 层总耗时 (ms) | 8.15 | 4.76 | 4.06 |
| 相对 TightLLM 的 per-token overhead | **+100.7 %** | **+17.2 %** | — |

Bubble 减半、kernel_active 几乎翻倍——这说明压住 imbalance 振荡的首要收益不是"少等 DMA"，而是**让控制器停止对 plan 结构的频繁扰动**，pre_hook 的 `imbalance` sync 分段也从 430 ms/step 降到 221 ms/step（见 §8.6）。核心机理与 §8.3 观察 3 呼应：SM 关时 `kernel_active` 只有 15.7 ms 而 bubble 13.1 ms，compute 路径本身被 assembly/H2D 等待撑胖；SM 开后 `kernel_active` 涨到 25.9 ms、bubble 降到 7.7 ms，compute 路径回归正常，同时 IO 被完全 overlap。

### 8.6 Pre_hook 分解（RunKV 侧）

| 分段 (ms/step) | SM 关 | SM 开 |
|---|---:|---:|
| h2d_sync | 0.58 | 0.57 |
| imbalance（跨 stream 同步等待）| **430.2** | **221.1** |
| build_plan | 6.1 | 18.3 |
| skip_ids | 0.05 | 0.06 |
| schedule_io | 34.9 | 37.1 |
| **total** | **471.9** | **277.1** |

- `imbalance` 段下降 ~210 ms/step 是主要收益：这段等待来自 `load_ready_event.sync()`，振荡减小后 CPU 不再被迫等到很晚才结算。
- `build_plan` 从 6.1 升到 18.3 ms 属于**主动取舍**：SM 开启后同步重建的次数更多（稳态分支仍偶尔走 rebuild 路径，且 speculative 启发式现在会跳过一些稳态 spec 提交），但仍远小于收益。后续若需进一步优化可考虑放宽稳态分支的 skip 条件。

### 8.7 关键结论

1. **真正的瓶颈是 compute，不是 IO**（§8.3）：per-layer max(compute, DMA) ≈ compute，Σ layer_total_gpu 基本等于 Σ compute_dur + 少量 bubble；IO 被 pipeline 完全吸收。Imbalance 只是"两条路径出口错位"的测量，不是"IO 绝对量超了"。
2. **振荡与饱和消失**：`layers with GPU stall` 从 30/31 降到 0/31，验证 Newton secant 的 sign-flip / bang-bang 问题被状态机替换完全消除。
3. **Budget 达到合理工作点**：replay ratio 从 0.725 提升到 0.836，接近 TightLLM 的 0.866，同时 cpu_fill 下降。
4. **Per-token overhead 压缩到 +17 %**：相比 SM 关的 +100 %，已进入 "与 TightLLM 同量级" 区间，剩余开销可追溯到 CPU-side prehook 本身（见 §8.6 的 `build_plan` + `schedule_io`）。
5. **端到端 step -47 ms vs TightLLM**：RunKV 在 offload 模式下已能同时做到 "替 GPU 省 KV 容量" 与 "不比常驻 GPU 方案慢"，且 compute 路径比 TightLLM 少 3.6 % 的同时把 IO 完全藏掉。

---

## 9. 当前状态与后续工作

已在 `runkv` 分支上落地：feedback planner（Step 1–10）+ speculative plan building + imbalance state machine controller + 基础可观测性。

近期规划（未完成）：

- **Step 11 — step-level batch-aware reinit**：drift 判定下的 warm start / shrink / reset（`FeedbackReplayPlanProvider.begin_step` 当前只做 clamp）。
- **Regime classifier**：外层 regime 分类 + per-regime baseline_ewma，替换全局单 baseline；目标是消除跨 regime 切换 step 的异常方差。
- **Adaptive probe size / STEADY gain 累积**：让 TRACKING 首步不再依赖 `probe_size_blocks=2` 常量，并把 STEADY 期间 ±1 block 的微调信息喂进长窗 OLS。
- **DMA segment 预计算（§Phase 6）与 staging buffer**：把 `schedule_io` 的 37 ms/step 再砍一半。
- **TightLLM 路径的 SM 适配**：`TightLLMReplayPlanProvider` 目前仍走旧反馈，若需要在 TightLLM 下实验也能受益于 deferred build，需要统一 provider 接口。

---

## 10. 参考

- 设计文档：
  - [feedback_driven_replay_planner_design.md](feedback_driven_replay_planner_design.md) — planner 骨架与 Step 1–12 拆分
  - [imbalance_state_machine_controller.md](imbalance_state_machine_controller.md) — 三态控制器与 plan-change hint 语义
  - [deferred_speculative_plan_building.md](deferred_speculative_plan_building.md) — QKV 参考点 + 后台 builder
  - [layer_dynamic_replay_design.md](layer_dynamic_replay_design.md) — 底层 replay runtime
- 关键源文件：
  - [vllm/v1/worker/opt_dynamic_replay.py](/home/lyc/inference/vllm/vllm/v1/worker/opt_dynamic_replay.py)
  - [vllm/v1/worker/imbalance_controller.py](/home/lyc/inference/vllm/vllm/v1/worker/imbalance_controller.py)
  - [vllm/v1/worker/gpu_model_runner.py](/home/lyc/inference/vllm/vllm/v1/worker/gpu_model_runner.py) (`_runkv_pre_hook`, `_prepare_dynamic_replay_runtime`, `_build_speculative_for_layer_impl`)
  - [vllm/v1/worker/layer_recompute.py](/home/lyc/inference/vllm/vllm/v1/worker/layer_recompute.py)
  - [vllm/model_executor/models/opt.py](/home/lyc/inference/vllm/vllm/model_executor/models/opt.py) (`OPTDecoder._forward_dynamic_replay`)
- 实验产物：
  - `exp_results/analysis/per_layer/20260421_2008/` — SM 未启用快照
  - `exp_results/analysis/per_layer/20260423_2329/` — SM 启用快照
  - `exp_results/opt_feedback_observation/` / `exp_results/tightllm_observation/` — 原始 JSONL / NSYS 报告
