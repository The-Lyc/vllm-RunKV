# Imbalance State Machine Controller

## 1. 背景与动机

当前 `FeedbackReplayPlanProvider` 在观测到每层的 IO/compute imbalance 之后，使用带阻尼的 Newton secant 更新全局 `global_budget_blocks`。实际运行数据暴露出几个结构性问题：

- **Secant gain 符号不稳**：`gain_used` 分布接近 50/50 正负，意味着接近一半的更新方向是错的。噪声把 secant 的分母拉乱，估出与真实 gain 异号的值。
- **Bang-bang 饱和**：约 54% 的更新命中 `max_step_blocks` 的 clip 边界。控制器持续工作在饱和区，产生 ±10 之间的极限环。
- **Sign flip 高**：相邻观测 imbalance 异号比例接近 40%，连续翻转最长 7 层。
- **双峰 imbalance**：系统在两个工作点之间跳动，不是高斯噪声，是控制器自身的超调行为。

问题的根本在于**控制律一刀切地、每层都动**，既不区分当前是稳态还是过渡，也不识别噪声和真实偏移，于是 Newton 的线性局部近似假设在非稳态下被滥用。

本方案用一个显式的 state machine 替换"每层无条件跑 secant"，把数据驱动的决策与物理模型驱动的更新分离：
- State machine 负责判断"当前处于稳态 / 过渡 / 追踪"，以及"什么时候该用 secant"
- Secant 只在已经进入一个干净的新稳态之后才启用，且喂给它的数据全部来自同一稳态

## 2. 数据观察与关键发现

基于 `DRY_RUN=1` 条件下采集的开环 imbalance 样本（`exp_results/opt_feedback_observation/opt_component_mfu_1000_20260422_174059.flat.jsonl`）：

| 指标 | 值 |
|---|---|
| 稳态内 imbalance 的 per-step P50 σ | **0.24 ms** |
| 稳态内 P95 σ | 2.69 ms |
| Mean σ（跨所有 step） | 2.89 ms（被 regime 切换 step 拉高） |

核心发现三条：

1. **真正的稳态噪声极小**（亚 ms 级）。之前从闭环数据估出的 4–5 ms σ 是控制器自身动作造成的，不是测量噪声。
2. **不同 regime 的 imbalance mean 差距很大**：同样"DRY_RUN 静态 plan"下，`(prefill, 4 reqs)` 的 baseline 是 +9 ms，`(decode, 12 reqs)` 是 +45 ms。**mean 是 regime 的函数，σ 基本 regime 无关**。
3. **跨 regime 切换的 step 有 >100ms 的异常方差**，明显是窗口跨越两个总体的产物，不是真正的单一总体方差。

这决定了控制架构必须 regime-conditioned，而层内控制律必须识别并避开 regime 切换期间的"混合窗口"。

## 3. 整体控制架构

两个正交的层次：

**外层：regime classifier**
- 在 `begin_step` 时根据 batch fingerprint（phase、num_reqs bucket 等）查找当前所属 regime
- 每个 regime 维护独立的 `(baseline_imbalance, steady_budget, σ_estimate)` 状态表
- 跨 step 的 regime 切换由 classifier 直接触发状态表切换，不依赖层内 state machine

**内层：per-regime state machine**
- 在同一 regime 内工作，决定当前该处于"稳态 / 过渡 / 追踪"哪个状态
- 只对**同 regime 内的偶发漂移或未被 classifier 捕获的缓慢变化**作出反应
- 本文档专注描述内层 state machine

两层职责边界清晰：
- 大部分 regime 切换由 batch 组成变化触发，由 classifier 在 step 边界处理
- State machine 只处理同 regime 内的缓慢漂移（thermal、contention 等）以及 classifier 漏检场景

## 4. 层内 state machine 设计

三个状态：

```
STEADY ──(stdev 变大)──> TRANSIT ──(窗口变纯 & stdev 回落)──┬─> STEADY   (mean 回到旧 baseline → 虚警)
                                                            └─> TRACKING (mean 远离旧 baseline → 真实偏移)

TRACKING ──(连续 K 层 residual 回到 deadband)──> STEADY
```

三个状态各自的控制律：

| 状态 | 控制律 | Budget 行为 |
|---|---|---|
| STEADY | Deadband + proportional（步长 ≤ 1 block），不估 gain | 轻微调整维持不漂移 |
| TRANSIT | 观测但不行动 | **冻结**，不变 |
| TRACKING | Newton step + RLS gain 估计，步长 ≤ `max_step` | 大步调向新最优 |

### 4.1 决策窗口

固定 K=3。依据是：
- 稳态 σ ≈ 0.24 ms，3 个样本的均值 std error ≈ 0.14 ms。对于幅度 ≥1 ms 的真实偏移，3 个样本提供接近 7σ 的信噪比，检测功效足够。
- 3 个连续噪声样本恰好拼成"另一个稳态形状"（同号、幅度超阈值、stdev 小）的概率极低，几乎不可能触发虚警。
- 延迟对响应速度关键：K 直接决定 STEADY→TRACKING 的最小延迟（约 2K 层）。

### 4.2 数据不丢弃原则

窗口是连续滑动的 rolling buffer。状态切换不清空窗口，也不丢弃 TRANSIT 期间的样本。理由：
- 如果 TRANSIT 之后确实进入了新稳态，那些"TRANSIT 后半段"的样本本身就是新稳态数据，是有用的
- 让窗口自然演化：触发 TRANSIT 后再经过 K 层，窗口里的旧 regime 样本会自然滑出，剩下的全是新样本
- 把"窗口滑动的物理过程"当成状态切换的计时器，避免显式管理样本生命周期

## 5. Outlier 处理：v1 不做预过滤

早期方案里曾设计过 MAD-based 单点 outlier 预过滤。评估后决定 **v1 不做**，理由：

- State machine 的"虚警分支"天然吸收了单点尖峰。一次 +50 ms 的突发值会让窗口 stdev 暴涨 → 触发 TRANSIT → budget 冻结 K 层等窗口变纯 → 新窗口里没有那个尖峰，stdev 回落、mean ≈ old_baseline → 判定虚警回 STEADY。代价只有 K=3 层的 budget 冻结，没有错误决策。
- 独立的 MAD 预过滤会引入 `K_filter`、`outlier_sigma_multiple` 等新参数，增加调参负担，而它节省的只是那 3 层 TRANSIT 延迟。
- 实际 workload 里单点尖峰的频率未知，可能根本不值得加这层保护。

后续如果实测发现 TRANSIT 被尖峰频繁误触发（表现为高频的短周期虚警），再补回 MAD 作为 v2 的优化。

## 6. 状态转换规则

### 6.1 STEADY → TRANSIT

**触发**：`stdev(window) > M × σ_baseline`，其中 M=3。

只看 stdev，不看 mean 或 sign。因为：
- Mean 在正常稳态下围绕 baseline 抖动，绝对值不稳定
- Stdev 变大是"窗口内包含两个不同总体样本"的直接信号，物理意义明确
- 单信号触发逻辑简单，易调试

进入 TRANSIT 的副作用：
- 冻结 budget，记录 `frozen_budget = current_budget`
- 冻结 `old_baseline = baseline_ewma`（STEADY 期间维护的 EWMA）
- 重置 `layers_in_transit = 0`

### 6.2 TRANSIT 内部：等窗口变纯

每来一个新观测，`layers_in_transit += 1`，窗口继续滑动。Budget 维持冻结。

当 `layers_in_transit ≥ K` 时，窗口里 K 个样本全部产生自 TRANSIT 触发之后——按"稳态段式出现、无连续扰动"的假设，它们要么在新稳态里，要么还在过渡尾部。

### 6.3 TRANSIT → 出口判决

窗口变纯后，在每一层都检查窗口 stdev：

**stdev 已回落到正常水平**（`< M × σ_baseline`）：说明已经进入一个稳定段，用 mean 做最终判决：
- `|mean − old_baseline| < Δ_threshold`（Δ=1 ms） → **虚警，回 STEADY**。`baseline_ewma` 不变，继续原工作点。
- `|mean − old_baseline| ≥ Δ_threshold` → **真实偏移，进 TRACKING**。使用窗口里 K 个样本作为 RLS 的初始种子。

**stdev 仍然大**：继续等。直到 `layers_in_transit > transit_timeout`（经验值 2K=6 层）仍未稳定，强制进 TRACKING（配合按 "稳态段式出现" 的假设，长时间 TRANSIT 不应该发生，超时是防止死锁）。

### 6.4 TRACKING 内部：probe-then-exploit

TRACKING 的核心挑战是：首步时我们**不知道 gain 的数值**。写死一个跨系统的 `default_gain` 经验常数（比如 −0.4 ms/block）是不可靠的——不同硬件、batch 形状、模型规模下 gain 的数量级会差出一两个数量级，错误的种子会让首步严重偏离最优。

但有一件事是跨系统确定的：**gain 的符号**。物理上 budget 增加 → replay compute 增加 → `qkv_end` 右移 → `imbalance = ready − qkv_end` 减小 → 因此 `d(imbalance)/d(budget) < 0`。这个符号是由系统物理结构决定的，不随 workload 变化。

利用这个不变量，TRACKING 采用 **probe-then-exploit** 策略：

**首步（probe）**：只用符号，不用数值
- `Δbudget = −sign(mean_window) × probe_size`，其中 `probe_size = 2 blocks`
- `probe_size` 是一个 system-agnostic 的 safe 幅度：足够小不会造成剧烈扰动，足够大能产生可测的 imbalance 响应

**第二步开始（exploit）**：首步的结果给出 gain 测量值
- 观测 probe 后的新 imbalance，计算 `gain_measured = (new_imbalance − mean_window) / probe_size`
- 从第二步起用 `gain_measured` 做 Newton step：`Δbudget = clip(−current_imbalance / gain × damping, ±max_step)`
- 后续每层把新的 `(budget, imbalance)` pair 加入 RLS 样本池（固定长度 6，FIFO），用 OLS 更新 gain
- **Gain 符号保护**：若 RLS 产出的 gain 与物理要求的符号（负）相反，拒绝更新，保留上一步的 gain 估计

这样整个设计里唯一需要配置的常量是 `probe_size_blocks = 2`，而它是 system-agnostic 的——换硬件、换模型都不用重新调。

### 6.5 TRACKING 期间的 gain 累积（可选增强）

STEADY 期间的 ±1 block proportional 微调其实也携带 gain 信息。可以维护一个长窗口的 `(Δbudget, Δimbalance)` 样本池（比如最近 200 对），后台跑 OLS 得到一个全局 gain 估计。进 TRACKING 时，如果这个累积 gain 的标准误已经足够小（比如 < 50% 相对误差），直接用它作为 RLS 的种子 gain，跳过 probe 步。

这是 v2 的优化方向，v1 只做 probe-then-exploit。Probe 成本只有 1 层的次优 budget 调整，对比错误 default_gain 的风险，这一层的代价是值得的。

### 6.6 TRACKING → STEADY

两种退出路径：

- **正常收敛**：连续 `tracking_settle_layers`（=3）层的 `|imbalance| < deadband`，说明 budget 已经调到新最优。`baseline_ewma = window_mean`，回 STEADY。
- **硬超时**：TRACKING 停留超过 `tracking_timeout_layers`（=20）层仍未稳定，强制回 STEADY。用作失败保险，避免陷在 TRACKING 里无限消耗大步长。

## 7. 判据与参数

### 7.1 t-statistic（variance floor 版本）

用于需要统计显著性判断的地方（目前方案里主要是 debug/观测，实际转换由 stdev 单指标驱动）：

```
t = mean(window) / (max(stdev(window), σ_baseline) / √K)
```

`σ_baseline` 下界的作用是避免稳态下 stdev → 0 使 t 爆炸（稳态下 mean 也接近 0，不加下界时 t 是 0/0 的病态形式）。

### 7.2 参数表

| 参数 | 初始值 | 说明 |
|---|---|---|
| K（决策窗口） | 3 | 固定 |
| σ_baseline | 0.5 ms | variance floor；稳态噪声物理下界 |
| M（stdev 触发倍数） | 3.0 | TRANSIT 触发阈值 = M × σ_baseline |
| Δ_threshold（虚警判定） | 1.0 ms | 区分虚警 vs 真实偏移 |
| deadband | 0.5 ms | STEADY 和 TRACKING 收敛判定的死区 |
| steady_small_step_blocks | 1 | STEADY 下的比例控制步长 |
| probe_size_blocks | 2 | TRACKING 首步探针幅度（system-agnostic） |
| tracking_max_step_blocks | 10 | TRACKING 下的 step 上限 |
| tracking_damping | 0.6 | TRACKING 的 Newton step 阻尼 |
| transit_timeout_layers | 6（2K） | TRANSIT 强制出口 |
| tracking_settle_layers | 3 | TRACKING 正常收敛判据 |
| tracking_timeout_layers | 20 | TRACKING 硬超时 |
| baseline_ewma_alpha | 0.1 | STEADY 的 baseline EWMA 权重 |
| rls_window | 6 | RLS 样本池长度 |

**物理常量（不是可调参数）**：
- `gain_sign = −1`：由系统物理结构决定（budget ↑ → compute ↑ → imbalance ↓），跨系统不变

所有参数作为 config 暴露，初始值基于 DRY_RUN 样本和经验估计。与传统 Newton 控制器不同，本方案**没有需要按系统校准的魔数 gain 值**——TRACKING 的首步通过 probe 自动测出 gain 数值。

## 8. 集成到现有代码

### 8.1 与 FeedbackReplayPlanProvider 的集成

**新文件**：`vllm/v1/worker/imbalance_controller.py`
- `ImbalanceControllerConfig`（dataclass）
- `ImbalanceController` 类
- `SMState` enum
- `ControlDecision`（dataclass，封装 state machine 的输出）

**改动点**：`FeedbackReplayPlanProvider.observe_layer_feedback`
- 根据配置开关选择新旧控制器
- 新路径调用 `ImbalanceController.observe(...)` 获得 `ControlDecision`
- 把 decision 转换为 `global_budget_blocks` 更新，并记录 `FeedbackControllerLayerUpdate`

**配置扩展**：`KVOffloadConfig`
- 新增 `layer_recompute_use_state_machine: bool = False`（默认关，不影响现有行为）
- 新增该状态机的所有 config 参数，全部有 sensible default

**观测性扩展**：`FeedbackControllerLayerUpdate` 增加字段
- `sm_state`、`window_mean_ms`、`window_stdev_ms`、`old_baseline_ms`、`plan_change_hint`
- 下游 JSONL（`opt_component_mfu_*.flat.jsonl`）自动包含这些字段，便于事后分析

关键的 state dispatch 伪代码。`hint` 在最后统一由 Δbudget 导出（不在每个分支里单独写）：

```
observe(imbalance_ms, current_budget):
    window.append((imbalance_ms, current_budget))
    delta = 0

    if state == STEADY:
        update baseline_ewma
        if len(window) == K and stdev(window) > M * σ_baseline:
            enter TRANSIT, freeze old_baseline and budget
        else:
            delta = deadband-or-proportional-nudge  # 0 or ±1

    elif state == TRANSIT:
        layers_in_transit += 1
        if layers_in_transit >= K:
            if stdev(window) < M * σ_baseline:
                if |mean(window) - old_baseline| < Δ:
                    state = STEADY  # false alarm
                else:
                    state = TRACKING  # real shift
                    delta = −sign(mean_window) × probe_size  # probe step
            elif layers_in_transit > transit_timeout:
                state = TRACKING (forced)
                delta = −sign(mean_window) × probe_size

    elif state == TRACKING:
        if first step after entering TRACKING:
            delta = already set at TRANSIT → TRACKING transition
        else:
            append (budget, imbalance) to RLS pool
            update gain with sign-guard (reject if sign flipped)
            if |imbalance| < deadband for settle_layers → state = STEADY
            elif tracking_layer_counter > timeout → state = STEADY (hard abort)
            else: delta = clipped Newton step  # naturally ≈0 when |imb|<deadband

    # hint 由 delta 导出
    if delta == 0:       hint = "unchanged"
    elif |delta| <= 1:   hint = "small_delta"
    else:                hint = "significant_delta"
    return ControlDecision(delta_budget=delta, plan_change_hint=hint, state=state)
```

### 8.2 与 speculative plan building 的集成

Speculative plan building（post_hook 提交 spec(L+2)，pre_hook pop 并组装）在 state machine 架构下**仍然保留**，但它的角色被重新定位：只在控制器确实要动 budget 时才有价值。当系统处于稳态（Δbudget 小且 imbalance 已在 deadband 内），pre_hook 不再需要异步构建好的 spec plan，而是**同步构造一个轻量的 "stable successor plan"**，直接跳过 plan(L+1) 里最昂贵的部分——hidden states 的 H2D。

#### 8.2.1 核心观察：稳态下 plan(L+1) 的 IO 可以砍半

回顾 replay plan 的 IO 组成：每一层的 plan 同时驱动两类异步拷贝

1. **Hidden states H2D**：为下一层 replay 预取输入激活（`cpu_fill_token_count` 条）
2. **KV cache H2D**：`load_layer_async`，把当前层 replay 需要的 KV block 从 CPU 拉到 GPU

当 imbalance 已经进入 deadband 且控制器不动 budget（Δ=0 或 ±1）时，我们相当于在宣告"replay compute 已经和 IO 对齐，不需要再往 replay 池子里加新 token"。此时：

- 上一层 replay 池子里跑过的那些 token，它们的 hidden states 本来就已经落在 GPU 上（被 replay kernel 读过）。plan(L+1) 把它们转成 `gpu_reuse` 段完全够用
- `cpu_fill_token_count` 可以直接置零——稳态下我们不再补新 token 到 replay，H2D hidden states 这一路 IO **整条砍掉**
- KV cache 那一路 IO 不能省：plan(L+1) 对应的是第 L+1 层的 KV，跟上一层的 KV 是不同 block，必须继续 `load_layer_async`

这就是 **stable successor plan** 的构造：`gpu_reuse` 覆盖 prev replay 的全部 token，`cpu_fill = 0`，其他字段按照新 layer 重新生成（attention metadata、skip_block_ids 等）。构造成本纯 CPU、无张量分配，pre_hook 同步做完全可接受。

#### 8.2.2 `plan_change_hint` 的三档语义

`ControlDecision` 里新增 `plan_change_hint` 字段。**判据直接来自控制器当前这一层返回的 `Δbudget` 值**，不直接绑定状态机状态：

| 条件 | `plan_change_hint` | Pre_hook 处理 |
|---|---|---|
| `Δbudget == 0` | `unchanged` | 同步构造 stable successor plan（`gpu_reuse = replay(L)`, `cpu_fill = 0`），丢弃 spec |
| `0 < \|Δbudget\| ≤ 1` | `small_delta` | 同步构造 stable successor plan（±1 block 的差异通过 budget 下一层再慢慢吸收，不走 spec 路径）|
| `\|Δbudget\| ≥ 2` | `significant_delta` | 正常 pop spec，走 rebuild 流程 |

这套 Δbudget-驱动的判据自然覆盖所有状态：

- **STEADY deadband** → Δ=0 → `unchanged`
- **STEADY nudge** → Δ=±1 → `small_delta`
- **TRANSIT**（冻结） → Δ=0 → `unchanged`
- **TRACKING 的 probe 步** → |Δ|=probe_size=2 → `significant_delta`
- **TRACKING 正常步** → 根据 Newton step 幅度决定
- **TRACKING 收敛阶段**（`|imbalance| < deadband`） → Newton step 自然算出 Δ≈0 → `unchanged`

最后这条尤其重要：**TRACKING 正式退出条件是连续 `tracking_settle_layers` 层 `|imbalance| < deadband`**，但在那几层里 Newton step 本身就会产出 Δ=0（因为 residual 在死区里）。用 Δ 驱动 hint 意味着这几层**自动走 stable successor 分支**，省掉 hidden states H2D，不会再白白消耗 speculative build 的产物。这解决了"imbalance 已经降到 deadband 还用 spec 是浪费"的问题。

这套判据比旧版 `last_observed_stable()` 更强的原因：
- 旧版基于连续 deadband 计数，是行为启发式，没有明确物理含义。而且旧实现里是尝试**直接复用 plan(L) 作为 plan(L+1)**，这在层间 `replay_token_count` 对不上时会触发 `Expected N replay tokens but assembled M` 的 assertion（session 早期踩过的 bug）。新设计不做 plan 复用，而是**重建一个 cpu_fill=0 的新 plan**，从源头上避开了 token 数对不齐的风险
- 新版直接用控制器这一层实际要做的 Δbudget 做判据，语义清晰，任何"不动 budget"的路径都会自动归为 `unchanged`

#### 8.2.3 Speculative build 的提交时机

提交时机无法直接用 Δbudget 驱动，因为 post_hook 里我们要预判**将来 L+2 的 pre_hook 会不会需要 spec**，而那时的 Δbudget 还没算出来。只能用状态机当前状态做启发式预测：

- **STEADY 或 TRANSIT**：下一层 Δbudget 大概率是 0 或 ±1，pre_hook 会走 stable successor 分支，**不需要提交 speculative build**。省掉 builder 线程的 CPU 工作，也省掉 builder 提前为 `cpu_fill` 做的预计算
- **TRACKING**：budget 大概率会动，提交 spec(L+2) 让 builder 线程并行准备

这让 speculative building 的工作量大致匹配控制器动作频率——系统稳定时省 CPU，系统变化时才并行准备。

注意这是启发式：TRACKING 末尾的收敛层其实不需要 spec（§8.2.2 最后一条），但这些层仍会提交 spec（builder 还是做了无用功）。这是启发式的代价；相比"每层都提交 spec"仍是改进，v1 可接受。

#### 8.2.4 边界情况：TRANSIT → TRACKING 切换那一层

状态机在第 N 层的 `observe()` 里判定进 TRACKING 并返回 probe 动作。但 N+1 层的 pre_hook 到来时，N 层的 post_hook 可能没提交 speculative（因为 N 层前还是 TRANSIT）。这意味着 N+1 层的 pre_hook 会发现 `spec_plan is None`，走**同步 rebuild** fallback（完整 rebuild，不是轻量的 stable successor）。

代价：1 层的 pre_hook 同步开销，在 TRACKING 入口这种"大调整时刻"可接受。

如果想消除这一层的同步开销，可以在 TRANSIT 的判决点（`layers_in_transit = K` 那一刻）**预测下一层会切 TRACKING** 时提前提交 spec。这是优化，v1 不做。

#### 8.2.5 具体 pre_hook 改动

[gpu_model_runner.py 的 pre_hook 分支](vllm/v1/worker/gpu_model_runner.py#L3268-L3283) 当前逻辑（runkv 分支已实现）：
```
if runtime.last_observed_stable() and invariant holds:
    next_plan = build_stable_successor_plan(current_plan)
    同步 build attention metadata + compute skip_block_ids
    clear_speculative(L+1)
else:
    pop_speculative → fallback to sync rebuild
```

state machine 启用后改为：
```
hint = runtime.get_last_plan_change_hint(layer_idx)
if hint in ("unchanged", "small_delta"):
    next_plan = build_stable_successor_plan(current_plan)
    同步 build attention metadata + compute skip_block_ids
    runtime.clear_speculative(next_layer_idx)
else:  # "significant_delta"
    pop_speculative → fallback to sync rebuild
```

核心仅是把触发条件从启发式 `last_observed_stable()` 换成 Δbudget-driven 的 `hint`；stable 分支的动作（`build_stable_successor_plan`）保持不变。

`runtime` 里维护 `_last_plan_change_hint`，由 `FeedbackReplayPlanProvider` 在每次控制器 observe 之后调用 `runtime.note_plan_change_hint(hint)` 更新。这取代旧的 `note_observe_action`。

旧的 `note_observe_action` + `last_observed_stable` 机制在 state machine 启用时退路保留（通过配置开关切换），完全替换后删除。

## 9. 验证方案

### 9.1 单元测试

`tests/v1/kv_offload/test_imbalance_controller.py` 覆盖：

| 场景 | 输入序列特征 | 预期行为 |
|---|---|---|
| 纯稳态 | 长序列噪声，`σ ≈ 0.24 ms` | 始终 STEADY，不触发 TRANSIT |
| 单点 outlier | 稳态中夹一个 +50 ms | STEADY → TRANSIT → 虚警回 STEADY（代价 K 层冻结） |
| 真实偏移 | 稳态 → 跳到 +15 ms 稳态 | STEADY → TRANSIT → TRACKING → 回 STEADY |
| 虚警短扰动 | 稳态 → 短时 +5 ms 3 层 → 恢复 | STEADY → TRANSIT → 回 STEADY |
| TRANSIT 超时 | 持续高 stdev | 超时后强制 TRACKING |
| Probe 首步 | 进 TRACKING 后观察首次 Δbudget | 幅度 = probe_size，符号 = −sign(mean_window) |
| RLS gain 保护 | 喂异号 `(b,y)` 对 | Gain 拒绝，保持上一步估计 |

### 9.2 Offline replay

新建 `tools/profiler/sm_controller_replay.py`：
- 读 DRY_RUN JSONL
- 把 `imbalance_ms` 序列喂给 `ImbalanceController`
- 输出每层的 `(state, action, delta)` 轨迹
- 汇总：TRANSIT 触发次数和时刻、虚警率、TRACKING 收敛层数

关键验证：TRANSIT 触发点应精确对应 `174059.flat.jsonl` 里已知的 regime 切换点（比如 step 101 的 stdev=141 ms 跳变）。不该在稳态 step 内随机触发。

### 9.3 Online A/B

完成 offline 验证后，在真实 workload 下 A/B 对比新老控制器：

| 指标 | 当前 Newton | 目标 |
|---|---|---|
| `sign_flip_rate` | 38% | < 10% |
| `saturation_rate`（`|Δ|==max_step`） | 54% | < 20% |
| `bad_gain_rate` | 45% | < 5% |
| P95 `|imbalance|` | 16.4 ms | 显著下降 |

关注 tail latency：状态机的延迟检测换来稳定性，总平均 imbalance 不应上升。

## 10. 未解决的问题与风险

### 10.1 Regime classifier 与 state machine 的接口

本文档专注内层 state machine，假设外层 regime classifier 会在 regime 切换时重置 state machine 到 STEADY 并换用该 regime 的 `baseline_ewma`、`σ_baseline` 等参数。但 classifier 本身尚未实现，接口仍需明确：

- Classifier 在 `begin_step` 判定 regime change 时，是否调用 `controller.reset_for_regime(new_regime_state)`？
- 新 regime 第一次出现时（冷启动），`baseline_ewma` 从 0 开始还是用 ILP seed？

短期方案：先不实现 classifier，用单一全局 baseline_ewma。观察真实 workload 里的 regime 切换有多少能被层内 state machine 正确处理（走 TRANSIT 分支），有多少导致虚警/漏检——据此决定 classifier 是否必要，以及它应该做到什么粒度。

### 10.2 Probe size 的选择与 gain 预学习

`probe_size_blocks = 2` 的选择是经验值，权衡：
- 太小（=1）：probe 产生的 imbalance 变化可能被噪声淹没，gain 测量不准
- 太大（≥5）：probe 本身造成的过冲明显，首步代价过高

2 blocks 在多数场景下同时满足"响应可测"和"幅度 safe"。但严格讲最优 probe 幅度应该正比于 `σ_baseline / |gain|`——系统越嘈杂或 gain 越小，probe 需要越大。

v2 可以考虑：
- **Adaptive probe size**：记录几次 probe 的 `(probe_size, Δimbalance)` 对，动态调整
- **STEADY 期间持续 gain 累积**（§6.5 描述的可选增强）：从大量 ±1 block 微调中积累样本，进 TRACKING 时直接用这个累积 gain 做 RLS 种子，完全跳过 probe 步

### 10.3 Regime 内非平稳因素

即使在同一 regime 内，长期运行也可能有缓慢漂移（GPU thermal、内存碎片化）。目前 baseline_ewma 的 α=0.1 提供一定的自适应性，但如果漂移速率大于 EWMA 跟踪速率，state machine 可能频繁误触发 TRANSIT。需要实测后决定是否加更严格的漂移补偿。

### 10.4 参数调整责任

参数虽然都有默认值，但最优值依赖具体 workload 和硬件。需要建立一套**离线调参工具链**：
- 用 DRY_RUN 样本回归出 `σ_baseline`
- 用历史真实运行日志跑 offline replay，调 `M`、`Δ_threshold` 等决策阈值
- 生成推荐参数配置

Gain 自学习（§6.4 probe-then-exploit）意味着**不再需要离线拟合 default_gain**。这是本方案相对传统 Newton 的关键简化——减少了一个跨系统魔数的校准负担。

### 10.5 Small_delta 复用的正确性

§8.2.1 提出 `plan_change_hint == "small_delta"` 时可以选择复用 plan(L)。这依赖一个假设：±1 block 的 budget 变化对 plan 结构的影响可忽略。

严格讲，增加 1 block 意味着多一个 block 进入 replay 集合，对应的 `kv_replay_start`、`gpu_reuse_slice` 等字段会变化。复用旧 plan 等于让这一层少 replay 一个 block（或多 replay 一个 block），实际上就是把 budget 变化延迟一层生效。

在 STEADY 期间 ±1 block 只是为了防漂移，延迟一层生效没有语义差别。但需要 code review 确认跨层 plan 的 data structure 允许这种"budget 差 1 但 plan 一致"的复用不引入任何正确性问题（特别是 replay_token_count 和 metadata 的一致性）。如果有风险，v1 直接把 `small_delta` 当成 `significant_delta` 处理（走 rebuild），牺牲一点 CPU 换安全。

## 11. 落地顺序

推荐分 5 步推进，前两步纯 offline，不碰运行时代码：

1. **写 `imbalance_controller.py` + 单元测试** — 独立模块，单测全绿
2. **写 offline replay 工具，在 DRY_RUN 样本上验证** — 触发轨迹合理
3. **接入 `FeedbackReplayPlanProvider`，加配置开关** — 默认关，不破坏现有行为
4. **加 JSONL 观测字段，跑真实 workload** — 实测对比
5. **A/B 决策** — 稳定后作为默认，保留旧 Newton 作为 fallback 开关

前两步完成后的产物可以独立验证、独立迭代，大幅降低集成风险。
