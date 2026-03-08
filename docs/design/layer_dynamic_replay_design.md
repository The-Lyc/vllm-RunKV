## RunKV OPT 主干动态 Replay 方案

### Summary
新增一种显式的 layer recompute 模式：`prev_layer_output_dynamic`。该模式只用于 `OPT`，但它不是和现有 `io_hidden_states` 完全独立的另一套机制。

正确的理解是：

- 每一层仍然有自己的 `IO prefix + replay suffix` 划分。
- 该层 replay suffix 的 K/V 仍然需要被重建，只是 replay 输入 hidden states 的来源变成了混合来源：
  - 能直接复用上一层 GPU replay 输出覆盖到的那部分 suffix，就直接复用；
  - 上一层没有覆盖到的那一段，由当前层自己的 CPU hidden-state store 补到 GPU。
- 因此，CPU hidden-state IO 不会只发生在 layer 0；它可能发生在任意层，但只发生在当前层 replay 窗口中、上一层 GPU 输出没有覆盖到的前缀缺口上。

这个方案的目标不是“消灭 CPU hidden-state IO”，而是“在保持每层 IO prefix / replay suffix 语义不变的前提下，尽量把 replay 输入从 CPU IO 迁移到上一层 GPU 输出复用”。

第一版保持旧模式为默认值，新模式只在 `OPT + RunKV + layer recompute` 下启用。第一版直接强制 `eager`，仅支持单机单卡；不兼容 full/piecewise cudagraph、DCP、TP、PP、cascade attention、ubatching、多 KV group。

### Public / Interface Changes
- 在 `RunKVOffloadConfig` 和 CLI 新增模式开关：
  - `layer_recompute_mode = "io_hidden_states" | "prev_layer_output_dynamic"`
  - CLI: `--runkv-layer-recompute-mode`
- 默认值保持 `"io_hidden_states"`，即现有实现完全不变。
- `prev_layer_output_dynamic` 的 phase-1 运行时校验：
  - 仅允许 `OPT`
  - 仅允许单机单卡
  - 仅允许单 KV cache group
  - 禁用 cudagraph、DCP、TP、PP、cascade attention、ubatching
  - 不满足时直接报错，不做 silent fallback
- 在 `ForwardContext` 新增可选字段：
  - `layer_recompute_runtime: OPTDynamicReplayRuntime | None`

### Implementation Changes
#### 1. Worker 侧改成"逐层获取 replay plan + 逐层构造 metadata"
在 `gpu_model_runner` 新增一条专用准备路径。与旧模式（step 级共享 metadata）不同，新模式按 **流水线方式逐层获取** replay plan。

每层的 replay plan 由独立的 `ReplayPlanProvider` 生成（phase-1 用静态或随机 provider，后续替换为动态决策模块）。

核心规则是每层独立定义自己的 replay 窗口：

- 对每个 request，先算本 step 前的历史长度 `computed_len`
- 决策模块可以先在 token 粒度产生一个“期望的 replay 起点” `desired_replay_start_token[layer]`
- 在真正构造 `LayerReplayPlan` 时，将这个 token 级起点向下对齐到 block 边界：
  - `kv_replay_start[layer] = floor(desired_replay_start_token[layer] / block_size) * block_size`
  - 并裁剪到合法区间 `[0, computed_len]`
- 对每层按 `io_prefix_blocks[layer]` 或 provider 的决策结果得到：
  - `kv_replay_start[layer]`
  - 含义：该层为了补本层缺失 KV，真正需要 replay 的历史 suffix 起点
  - 注意：这里的 `kv_replay_start` 是 block 对齐后的有效起点
- 这样做的目的：
  - 计划语义仍然可以是 token 级的，便于后续动态决策模块细粒度调整
  - 真正执行时回落到 block 边界，与现有 recompute / skip-block 机制保持一致
  - 相邻层若只存在几个 token 的小波动，通常会被同一个 block 边界吸收掉
  - 这样可以减少 layer 间 metadata、slot_mapping、pack / unpack 索引的抖动，让系统更稳定
- 对 layer 0：
  - `prev_gpu_start = computed_len`
  - 表示没有上一层 GPU replay 输出可复用
- 对 layer `l > 0`：
  - `prev_gpu_start = kv_replay_start[layer - 1]`
  - 表示上一层 GPU replay 输出覆盖的是区间 `[kv_replay_start[layer - 1], computed_len)`

因此，当前层 replay 输入被拆成两段：

- `cpu_fill_range = [kv_replay_start[layer], min(prev_gpu_start, computed_len))`
- `gpu_reuse_range = [max(kv_replay_start[layer], prev_gpu_start), computed_len)`

当前层最终 replay hidden states 输入是：

- `replay_input = concat(cpu_fill_hidden_states, gpu_reuse_hidden_states)`

这意味着有三种局部关系：

- 当前层 replay 更短：
  - `kv_replay_start[layer] > prev_gpu_start`
  - 当前层只需要上一层 GPU replay 输出的后缀切片
  - 当前层不做 CPU hidden-state IO
- 当前层 replay 一样长：
  - `kv_replay_start[layer] == prev_gpu_start`
  - 当前层完全复用上一层 GPU replay 输出
- 当前层 replay 更长：
  - `kv_replay_start[layer] < prev_gpu_start`
  - 当前层先从 CPU hidden-state store 补 `[kv_replay_start[layer], prev_gpu_start)`，
  - 再拼接上一层 GPU 输出 `[prev_gpu_start, computed_len)`

每层生成一个 `LayerReplayPlan`，包含：

- `kv_replay_start_per_req`
- `computed_lens_per_req`
- `prev_gpu_start_per_req`
- `cpu_fill_positions`
- `gpu_reuse_start_offset` 或等价的 GPU suffix slice 信息
- `scheduled_lens_per_req`
- 该层合并后 query 的 `query_start_loc`
- `num_actual_tokens` / `max_query_len`
- 该层合并 query 的 `slot_mapping`
- `combined_replay_indices` / `combined_scheduled_indices`
  - 用于 layer 内 pack / unpack
- 该层对应的 attention metadata

metadata 构造方式：

- 复用现有 attention builder，但对每层单独构造 `CommonAttentionMetadata`
- `seq_lens`、`max_seq_len`、`block_table_tensor` 在各层共享
- 每层变化的只有：
  - `query_start_loc`
  - `num_actual_tokens`
  - `max_query_len`
  - `slot_mapping`
- 动态模式下关闭现有 `cached_attn_metadata + update_block_table` 复用逻辑，直接逐层 `build()`，避免把 step 级 metadata 误复用于不同层

**slot_mapping 组装规则**：

对于每层的 `slot_mapping`，需要按 "per-request: replay first, scheduled second" 的顺序组装：

- 先遍历每个 request，收集该 request 在当前层 replay 窗口 `[kv_replay_start[layer], computed_len)` 中所有 token 的 slot
  - slot 计算：`slot = mapper.mapping[logical_block_table[req, pos // block_size]] * block_size + pos % block_size`
- 再收集该 request 的 scheduled tokens 的 slot（与旧模式一致）
- 最终 `slot_mapping` 形状为 `[num_actual_tokens]`，按 `query_start_loc` 指定的 per-request 偏移排布
- replay tokens 和 scheduled tokens 在同一 request 内连续排列，replay 在前、scheduled 在后，position 均按升序

**逐层 metadata build 开销分析与优化**：

当前每个 attention group 只 build 一次 metadata，所有层共享。新方案要求 **每层单独 build**。对 OPT-175B（96 层），build 次数从 1 → 96。需要控制开销：

- Phase-1 直接逐层调用 `builder.build()`，profile 单次耗时
  - 若单次 build ~10μs 级别（96 层 < 1ms），直接逐层 build
  - 若单次 build ~100μs 级别（96 层 ~10ms），需引入优化
- 优化方向：**metadata patch 复用**
  - 首层正常 `build()` 产出完整 metadata
  - 后续层如果 `query_start_loc` / `num_actual_tokens` / `max_query_len` / `slot_mapping` 与上一层相同（replay 窗口长度没变），直接复用上一层 metadata
  - 如果有变化，构造新的 `CommonAttentionMetadata`（仅替换变化字段，`seq_lens` / `block_table_tensor` / `max_seq_len` 复用），再调用 `build()`
  - 这样只有 replay 窗口发生跳变的层才需要真正 rebuild

#### 2. Hidden-state CPU store 继续保留，但定义为“目标层输入”
旧实现按层抓 LN 后 hidden states，用于直接投影 K/V。新实现仍然需要 CPU hidden-state store，但它的职责变成“为当前层 replay 窗口中上一层 GPU 没覆盖到的部分补洞”。

统一定义：

- `cpu_layer_inputs_by_layer[0]`：layer 0 的输入，来自 `OPTDecoder` 的 `embed + pos + project_in`
- `cpu_layer_inputs_by_layer[l]`：layer `l` 的输入，来自 layer `l-1` 的输出

因此：

- 新模式不再使用当前的 layernorm hook 抓取输入
- 改为在 `OPTDecoder.forward` 里显式捕获：
  - 进入 layer loop 前，捕获 layer 0 的 scheduled token 输入
  - 每过一层，捕获该层 scheduled token 输出，写入下一层输入 store
- 历史 replay token 的输出不再重复 D2H 写回，因为它们本来就是历史 token，CPU store 中应已存在
- 现有 `sync_hs_d2h()` 的 block/offset/materialize 逻辑保留，但写入对象从“当前层 LN HS store”改成“目标层输入 store”

#### 3. 主干执行改成“每层统一 forward，一次 attention 覆盖 replay + scheduled”
在 `OPTDecoder.forward` / `OPTDecoderLayer.forward` 增加动态路径，逻辑如下：

step 开始前：

- worker 不再假设“只有 layer 0 需要 CPU->GPU replay hidden-state 拷贝”
- 而是对每层只准备它自己的 `LayerReplayPlan`
- 实际执行到 layer `l` 时，再按该层 plan 判断：
  - 是否需要 CPU hidden-state IO 补 `cpu_fill_range`
  - 是否需要从上一层 GPU replay 输出切出 `gpu_reuse_range`

**IO prefix block loading + replay H2D 流水线时序**：

- IO prefix blocks 的 KV 加载和 replay hidden states 的 H2D 都在 pre-hook 中流水线式完成
- 新方案移除 pre-hook 中的 `recompute_kv_for_layer()` 调用（replay suffix 的 KV 改由主干 forward 的 `reshape_and_cache` 产出）
- 新 pre-hook 执行流程（layer i 的 pre-hook）：
  1. `mapper.sync_load_layer(i)` — 等待 layer i 的 IO prefix KV 加载完成（由上一层 pre-hook 发起的异步加载）
  2. 等待 layer i 的 replay hidden states H2D 完成（如果有 cpu_fill）
  3. 调用 `provider.get_layer_plan(i+1, ...)` — 获取 layer i+1 的 replay 计划
  4. `compute_skip_block_ids_for_layer(i+1)` — 根据 layer i+1 的 plan 计算 suffix blocks
  5. `mapper.load_layer_async(i+1, skip_block_ids)` — 异步启动 layer i+1 的 IO prefix KV 加载
  6. `load_cpu_fill_h2d(i+1, plan)` — 异步启动 layer i+1 的 replay hidden states H2D（如果 layer i+1 有 cpu_fill）
  7. 构造 layer i+1 的 attention metadata（或复用上一层的，如果 replay 窗口相同）
- 特殊处理：
  - Step 开始前（layer 0 进入 forward 之前）：获取 layer 0 的 plan，启动 layer 0 的 IO prefix 加载和 H2D
  - 最后一层 pre-hook：只做 sync，不再预取下一层
- 移除的旧逻辑：
  - `prefetch_recompute_inputs_for_layer()` — 不再在 pre-hook 中做旧模式的 H2D 拷贝
  - `recompute_kv_for_layer()` — 不再在 pre-hook 中做 QKV projection
  - 这两步被主干 forward 中的 "CPU 补洞 + 统一 layer forward" 替代

每层执行：

- 维护两份 GPU 张量：
  - `scheduled_hidden_states`
  - `replay_hidden_states`
- `replay_hidden_states` 的语义固定为：
  - 当前层自己的 replay 窗口输出，即区间 `[kv_replay_start[layer], computed_len)` 对应的 hidden states
  - 不再为更深层提前扩展成更长的 supersets
- 若该层 `kv_replay_start == computed_len`：
  - 该层无 replay，直接走原 layer forward
- 否则：
  - 按当前层 `LayerReplayPlan` 先构造 replay 输入：
    - CPU 补 `cpu_fill_range`
    - GPU 复用 `gpu_reuse_range`
    - 拼成当前层完整的 replay hidden states
  - 再用该层 `combined_*_indices` 把 `replay_hidden_states` 和 `scheduled_hidden_states` pack 成 `combined_hidden_states`
  - query 顺序固定为“每个 request 内 replay 在前，scheduled 在后”，且都按绝对 position 升序
  - 直接把 `combined_hidden_states` 送进现有 `OPTDecoderLayer.forward`
  - 该层 attention 自动使用该层专属 `attn_metadata`
  - `reshape_and_cache` 自动为 replay token 和 scheduled token 都写 KV
  - layer 输出后，再按预计算 indices 拆回：
    - `replay_hidden_states = combined_out[combined_replay_indices]`
    - `scheduled_hidden_states = combined_out[combined_scheduled_indices]`

最终只有 `scheduled_hidden_states` 继续进入 logits / sampling；`replay_hidden_states` 只在层间内部流动。

这条路径的关键点：

- 每层只做一次 attention，不存在旁路再跑一次 layer 的重复大 kernel
- 历史 suffix token 的 replay 完全并入主干 attention
- 当前层只对“本层自己的 replay 窗口”负责
- 若下一层需要更短 suffix，下一层直接从当前层 GPU 输出切后缀
- 若下一层需要更长 suffix，下一层自己从 CPU hidden-state store 补前缀缺口
- 因此不存在全局“从后往前传播 replay_start / replay_len”的需求

#### 4. Replay 决策模块接口（phase-1 留接口，不实现决策逻辑）

per-layer 的 replay 计划（每层的 `io_prefix_blocks` / `kv_replay_start`）最终将由一个独立的决策模块来生成。该模块在运行时根据当前 batch 的 request 分布、GPU 显存余量、IO 带宽等信息动态决定每层的 replay 窗口大小。

决策模块可以先给出 token 级的 `desired_replay_start_token`，但在真正生成 `LayerReplayPlan` 时，必须先向下对齐到 block 边界，得到有效的 `kv_replay_start`。phase-1 的所有 metadata 构造、skip block 计算、CPU fill / GPU reuse 边界都只使用这个 block-aligned 的 `kv_replay_start`。

**phase-1 不实现决策逻辑**，只定义接口：

```python
class ReplayPlanProvider(Protocol):
    def get_layer_plan(
        self,
        layer_idx: int,
        num_reqs: int,
        computed_lens: np.ndarray,       # [num_reqs]
        scheduled_lens: np.ndarray,      # [num_reqs]
        prev_layer_plan: LayerReplayPlan | None,
    ) -> LayerReplayPlan:
        """返回 layer_idx 层的 replay 计划。
        
        在 layer (layer_idx - 1) 的 pre-hook 中被调用，
        以便异步预取 layer_idx 的数据。
        """
        ...
```

phase-1 提供两个默认实现：
- `StaticReplayPlanProvider`：按固定 `io_prefix_blocks` 列表计算（与现有逻辑等价）
- `RandomReplayPlanProvider`：每层随机生成 token 级 `desired_replay_start_token`，再向下对齐到 block 边界（仅用于测试）

**调用时序**：

replay plan 的获取不是在 step 开始前一次性全部计算完成，而是 **流水线式逐层获取**：

1. Step 开始前：调用 `provider.get_layer_plan(0, ...)` 获取 layer 0 的 plan
   - 根据 layer 0 的 plan，构造 layer 0 的 attention metadata
   - 如果 layer 0 有 cpu_fill，启动 layer 0 的 hidden states H2D
2. Layer 0 的 pre-hook 中：
   - 等待 layer 0 的 IO prefix KV 加载完成
   - 调用 `provider.get_layer_plan(1, ...)` 获取 layer 1 的 plan
   - 根据 layer 1 的 plan：
     - 异步启动 layer 1 的 IO prefix KV 加载（`mapper.load_layer_async`）
     - 如果 layer 1 有 cpu_fill，异步启动 layer 1 的 hidden states H2D（`load_cpu_fill_h2d`）
     - 构造 layer 1 的 attention metadata
3. Layer i 的 pre-hook 中（通用模式）：
   - 等待 layer i 的 IO prefix KV 加载完成（`mapper.sync_load_layer(i)`）
   - 等待 layer i 的 hidden states H2D 完成（如果有）
   - 调用 `provider.get_layer_plan(i+1, ...)` 获取 layer i+1 的 plan
   - 异步启动 layer i+1 的 IO prefix KV 加载
   - 异步启动 layer i+1 的 hidden states H2D（如果 layer i+1 有 cpu_fill）
   - 构造 layer i+1 的 attention metadata
4. 最后一层的 pre-hook：
   - 只等待自己的数据加载完成，不再预取

这样每层的 **IO prefix KV 加载 + replay hidden states H2D** 都与上一层的 GPU 计算重叠，最大化 IO/计算并行度。

**与 OPTDecoder.forward 的交互**：

pre-hook 准备好的 layer i+1 plan 和 metadata 需要传递给 `_forward_dynamic_replay` 使用。统一采用逐层存储方式：
- `OPTDynamicReplayRuntime` 中维护 `_layer_plans[layer_idx]`
- `OPTDynamicReplayRuntime` 中维护 `_per_layer_attn_metadata[layer_idx]`
- pre-hook 负责写入 layer i+1 的 plan / metadata
- forward loop 按当前 `layer_idx` 读取
- 不再使用单槽位 `next_layer_plan`，也不通过 `ForwardContext.additional_kwargs` 传递

#### 5. 旧实现保留，但新模式与旧模式不是完全隔离的两套数据路径
- `io_hidden_states` 保持现状：
  - 当前层 replay 输入完全来自 CPU hidden-state IO
- `prev_layer_output_dynamic` 走专用 OPT 路径：
  - 当前层 replay 输入优先复用上一层 GPU 输出
  - CPU hidden-state store 只补当前层未覆盖的前缀缺口
- 两种模式共享：
  - RunKV block mapping
  - CPU block ownership / valid_len / positions 元数据
  - CPU hidden-state store 的物理存储
- 两种模式不共享：
  - 当前层 replay 输入的组装逻辑
  - attention metadata 生成方式
  - 层间 replay hidden-state 的 GPU 传递方式

### Test Plan
- 单元测试：当前层 replay 更短
  - `kv_replay_start[layer] > kv_replay_start[layer - 1]`
  - 当前层只做 GPU suffix slice
  - 不发生 CPU hidden-state IO
- 单元测试：当前层 replay 相等
  - 当前层完全复用上一层 GPU replay 输出
- 单元测试：当前层 replay 更长
  - `kv_replay_start[layer] < kv_replay_start[layer - 1]`
  - 当前层做“CPU 补前缀 + GPU 复用后缀”
- 单元测试：多层交替变长 / 变短
  - 验证每层只按自己的 replay 窗口工作
  - 验证不存在全局反向传播 replay 长度
- 单元测试：token 级小抖动被 block 边界吸收
  - 相邻层的 `desired_replay_start_token` 只差几个 token，但落在同一 block
  - 验证它们得到相同的 `kv_replay_start`
  - 验证 metadata / slot_mapping / combined indices 复用条件成立
- 单元测试：每层 pack / unpack 索引
  - replay 长度为 0
  - replay / scheduled 都非 0
  - 多 request 长度不均匀
  - pack 后再 unpack，scheduled / replay 恢复完全一致
- 单元测试：layer metadata
  - `query_start_loc`
  - `num_actual_tokens`
  - `max_query_len`
  - `slot_mapping`
  - 与每层 request 内 token 顺序一致
- 集成测试：小 OPT 模型 correctness
  - 新模式 vs 旧模式 / baseline 输出一致
  - 并发多 request
  - layer-wise replay 窗口交替变长 / 变短
  - 纯 decode、extend、长上下文 partial block
- 负向测试：
  - 非 OPT
  - cudagraph 打开
  - DCP / TP / PP / cascade / ubatching 打开
  - 多 KV group
  - 都应在初始化时明确报错

### Assumptions / Defaults
- 默认模式仍是 `io_hidden_states`
- phase-1 新模式只对 `OPT` 打开
- phase-1 只支持单机单卡
- phase-1 直接强制 eager，不兼容任何 cudagraph
- phase-1 correctness 目标先按 pre-LN OPT 做验证
- phase-1 不支持 post-LN OPT（如 OPT-350M）：post-LN 的 residual 路径不同，hidden-state store 的捕获点需要调整，延期到 phase-2
- hidden-state store 按"目标层输入"定义
- 决策模块可以在 token 粒度思考，但真正落地执行时一律使用 block-aligned `kv_replay_start`
- 当前层 replay 的目标始终是“补本层缺失 KV”，不是“替更深层提前生产更长 hidden states”
---

### Step-by-Step 实现方案（Agent 拆分）

以下按依赖顺序拆分为 12 个实现 step，每个 step 是一个可独立提交、可独立验证的最小单元。

约定：
- 每个 step 完成后，都要在本节回填 `实现状态`
- 同时补充该 step 的 `实际改动` 和 `如何使用`
- 写单元测试，并填写单元测试的测试内容和使用单元测试的方法
- 若设计文档中的计划与仓库实际结构不完全一致，需要在对应 step 下明确说明差异

#### Step 1: 配置层 — 新增 `layer_recompute_mode` 字段与 CLI 参数

**目标**：让用户可以通过配置切换到新模式，但不改变任何运行时行为。

**实现状态**：已完成

**改动文件**：
- `vllm/v1/core/kv_cache_offload_config.py`
  - `RunKVOffloadConfig` 新增字段 `layer_recompute_mode: str = "io_hidden_states"`
  - 可选值：`"io_hidden_states"` | `"prev_layer_output_dynamic"`
- `vllm/engine/arg_utils.py`
  - 新增 CLI 参数 `--runkv-layer-recompute-mode`
  - 解析并写入 `RunKVOffloadConfig`

**实际改动**：
- 已在 `vllm/v1/core/kv_cache_offload_config.py` 中新增 `layer_recompute_mode`
- 已在 `vllm/engine/arg_utils.py` 中新增：
  - `EngineArgs.runkv_layer_recompute_mode`
  - CLI 参数 `--runkv-layer-recompute-mode`
  - `_build_kv_offload_config()` 中的 mode 下沉与校验逻辑
- 已补测试：
  - `tests/engine/test_arg_utils.py`
  - `tests/v1/kv_offload/test_runkv_offload.py`
  - 覆盖默认值、CLI round-trip、非法值校验、`kv_offload_config` dict/object 路径
- 仓库中不存在 `vllm/v1/engine/arg_utils.py`，因此本 step 只需要修改统一的 `vllm/engine/arg_utils.py`

**如何使用**：

CLI:

```bash
vllm serve MODEL_PATH \
  --enable-runkv \
  --runkv-enable-layer-recompute \
  --runkv-layer-recompute-mode io_hidden_states
```

切换到新模式：

```bash
vllm serve MODEL_PATH \
  --enable-runkv \
  --runkv-enable-layer-recompute \
  --runkv-layer-recompute-mode prev_layer_output_dynamic
```

Python:

```python
from vllm.v1.core.kv_cache_offload_config import RunKVOffloadConfig

cfg = RunKVOffloadConfig(
    enabled=True,
    enable_layer_recompute=True,
    layer_recompute_mode="prev_layer_output_dynamic",
)
```

Step 1 单元测试运行方式：

```bash
./.venv/bin/python -m pytest -q tests/engine/test_arg_utils.py -k 'runkv_layer_recompute_mode or runkv_fields_are_preserved_by_from_cli_args'
./.venv/bin/python -m pytest -q tests/v1/kv_offload/test_runkv_offload.py -k 'TestRunKVOffloadConfig'
```

**验证方式**：
- 启动时传 `--runkv-layer-recompute-mode io_hidden_states` → 行为完全不变
- 传 `--runkv-layer-recompute-mode prev_layer_output_dynamic` → 配置可读但运行时无变化（还没有使用方）
- 传非法值 → 报错

**依赖**：无

---

#### Step 2: 运行时校验 — `prev_layer_output_dynamic` 的前置条件检查

**目标**：在 model runner 初始化时，对新模式做 phase-1 约束检查。

**改动文件**：
- `vllm/v1/worker/gpu_model_runner.py`
  - 在 RunKV 初始化路径中（现有 `_init_runkv_*` 系列方法附近），新增校验逻辑
  - 校验项：
    1. 模型必须是 OPT（检查 `model.__class__.__name__` 包含 "OPT"）
    2. 必须是 pre-LN（`config.do_layer_norm_before == True`）
    3. 只能单机单卡运行（TP / PP / DCP 全部为 1 或关闭）
    4. 只有单 KV cache group（`len(self.kv_cache_config.kv_cache_groups) == 1`）
    5. cudagraph 模式为 NONE
    6. cascade attention、ubatching 全部关闭
  - 不满足任一条件 → `raise ValueError` 带明确错误信息

**验证方式**：
- 传 `prev_layer_output_dynamic` + 非 OPT 模型 → 报错且消息明确
- 传 `prev_layer_output_dynamic` + cudagraph 打开 → 报错
- 传 `prev_layer_output_dynamic` + TP / PP 打开 → 报错
- 传 `prev_layer_output_dynamic` + OPT pre-LN + eager → 通过校验

**依赖**：Step 1

---

#### Step 3: 数据结构 — `LayerReplayPlan` 与 `OPTDynamicReplayRuntime`

**目标**：定义核心数据结构，不含任何构造逻辑。

**改动文件**：
- `vllm/v1/worker/opt_dynamic_replay.py`（新文件）
  - 定义 `LayerReplayPlan` dataclass：
    ```python
    @dataclass
    class LayerReplayPlan:
        kv_replay_start_per_req: np.ndarray       # [num_reqs] int32，block-aligned effective start
        computed_lens_per_req: np.ndarray          # [num_reqs] int32
        prev_gpu_start_per_req: np.ndarray         # [num_reqs] int32
        cpu_fill_token_count: int                  # 该层需要 CPU H2D 的 token 总数
        gpu_reuse_token_count: int                 # 该层复用上层 GPU 输出的 token 总数
        replay_token_count: int                    # = cpu_fill + gpu_reuse
        scheduled_token_count: int                 # scheduled tokens 总数
        num_actual_tokens: int                     # = replay + scheduled
        max_query_len: int
        query_start_loc: torch.Tensor              # [num_reqs + 1] int32
        slot_mapping: torch.Tensor                 # [num_actual_tokens] int64
        combined_replay_indices: torch.Tensor      # [replay_token_count] int64
        combined_scheduled_indices: torch.Tensor   # [scheduled_token_count] int64
        cpu_fill_positions: np.ndarray             # [cpu_fill_token_count] int32
        cpu_fill_logical_ids: np.ndarray           # [cpu_fill_token_count] int32
        cpu_fill_block_offsets: np.ndarray          # [cpu_fill_token_count] int32
        gpu_reuse_slice_per_req: list[tuple[int,int]]  # 每 req 在上层 replay_out 中的 [start, end)
    ```
  - 定义 `OPTDynamicReplayRuntime` dataclass：
    ```python
    @dataclass
    class OPTDynamicReplayRuntime:
        num_layers: int
        _layer_plans: list[LayerReplayPlan | None]         # [num_layers], 逐层填充
        _per_layer_attn_metadata: list[dict[str, Any] | None]  # [num_layers], 逐层填充
        cpu_hs_store: torch.Tensor                 # 引用 LayerRecomputeManager 的 store
        replay_plan_provider: ReplayPlanProvider   # 决策模块引用
        
        def get_layer_plan(self, layer_idx: int) -> LayerReplayPlan:
            """获取已由 pre-hook 填充的 layer plan。"""
            assert self._layer_plans[layer_idx] is not None
            return self._layer_plans[layer_idx]
        
        def set_layer_plan(self, layer_idx: int, plan: LayerReplayPlan):
            """pre-hook 调用，填充下一层 plan。"""
            self._layer_plans[layer_idx] = plan
        
        def set_layer_metadata(self, layer_idx: int, metadata: dict):
            self._per_layer_attn_metadata[layer_idx] = metadata
        
        def current_layer_plan(self, layer_idx: int) -> LayerReplayPlan | None:
            return self._layer_plans[layer_idx]
    ```
- `vllm/forward_context.py`
  - `ForwardContext` 新增可选字段：`layer_recompute_runtime: OPTDynamicReplayRuntime | None = None`

**验证方式**：
- 导入不报错
- dataclass 可正常实例化

**依赖**：无（可与 Step 1/2 并行）

---

#### Step 4: replay plan 接口与默认实现 — `ReplayPlanProvider` + 静态/随机 provider

**目标**：定义决策模块接口，提供 phase-1 默认实现。replay plan 的**决策逻辑**由独立模块负责，本 step 只定义接口和测试用的 stub。

**额外约束**：
- provider 可以先产生 token 级的 `desired_replay_start_token`
- 但在返回最终 `LayerReplayPlan` 之前，必须向下对齐到 block 边界，得到有效的 `kv_replay_start_per_req`
- phase-1 的静态 / 随机 provider 都遵守这个约束

**改动文件**：
- `vllm/v1/worker/opt_dynamic_replay.py`
  - 定义 `ReplayPlanProvider` Protocol：
    ```python
    class ReplayPlanProvider(Protocol):
        def get_layer_plan(
            self,
            layer_idx: int,
            num_reqs: int,
            computed_lens: np.ndarray,
            scheduled_lens: np.ndarray,
            logical_block_tables: np.ndarray,
            block_size: int,
            mapper_mapping: dict[int, int],
            prev_layer_plan: LayerReplayPlan | None,
        ) -> LayerReplayPlan:
            ...
    ```
  - 实现 `StaticReplayPlanProvider`：
    ```python
    class StaticReplayPlanProvider:
        """按固定 io_prefix_blocks 列表计算每层 plan。"""
        def __init__(self, io_prefix_blocks: list[int]):
            self.io_prefix_blocks = io_prefix_blocks
        
        def get_layer_plan(self, layer_idx, ...):
            # desired_replay_start_token = io_prefix_blocks[layer_idx] * block_size
            # kv_replay_start = floor(desired_replay_start_token / block_size) * block_size
            # 按 Section 1 的公式计算 cpu_fill / gpu_reuse / indices
            ...
    ```
  - 实现 `RandomReplayPlanProvider`（仅测试用）：
    ```python
    class RandomReplayPlanProvider:
        """每层随机生成 desired_replay_start_token，再对齐到 block。"""
        def __init__(self, num_layers: int, max_tokens: int, seed: int = 42):
            rng = np.random.default_rng(seed)
            self._random_start_tokens = [
                rng.integers(0, max_tokens + 1)
                for _ in range(num_layers)
            ]
        
        def get_layer_plan(self, layer_idx, ...):
            # 使用 self._random_start_tokens[layer_idx]，并向下对齐到 block 边界
            ...
    ```
  - `compute_layer_replay_plan_for_layer()` 核心函数（被两个 provider 共用）：
    ```python
    def compute_layer_replay_plan_for_layer(
        layer_idx: int,
        io_prefix_blocks_for_layer: int,
        computed_lens: np.ndarray,
        scheduled_lens: np.ndarray,
        logical_block_tables: np.ndarray,
        block_size: int,
        mapper_mapping: dict[int, int],
        prev_layer_plan: LayerReplayPlan | None,
    ) -> LayerReplayPlan:
    ```
    - 实现逻辑：
      1. 先得到 token 级 `desired_replay_start_token`
      2. `kv_replay_start = floor(desired_replay_start_token / block_size) * block_size`（per-req clip to computed_len）
      3. `prev_gpu_start`：无 prev_plan 则 = computed_len，否则 = prev_plan.kv_replay_start
      4. 计算 `cpu_fill_range` 和 `gpu_reuse_range`
      5. 组装 `query_start_loc`、`slot_mapping`
      6. 生成 `combined_replay_indices` / `combined_scheduled_indices`
      7. 记录 `cpu_fill_positions` / `cpu_fill_logical_ids` / `cpu_fill_block_offsets`
      8. 记录 `gpu_reuse_slice_per_req`

**验证方式**：（见 Step 9 单元测试，使用 RandomReplayPlanProvider 生成随机 plan 验证算法）

**依赖**：Step 3

---

#### Step 5: per-layer attention metadata 构造

**目标**：为每层构造独立的 `CommonAttentionMetadata` 并 build 成 backend-specific metadata。

**改动文件**：
- `vllm/v1/worker/gpu_model_runner.py`
  - 新增方法 `_build_per_layer_attn_metadata()`：
    ```python
    def _build_per_layer_attn_metadata(
        self,
        layer_plans: list[LayerReplayPlan],
        base_seq_lens: torch.Tensor,
        base_max_seq_len: int,
        block_table_tensor: torch.Tensor,
        num_reqs: int,
    ) -> list[dict[str, AttentionMetadata]]:
    ```
  - 实现逻辑：
    1. 遍历每层 plan
    2. 构造该层的 `CommonAttentionMetadata`：
       - `query_start_loc` = plan.query_start_loc
       - `num_actual_tokens` = plan.num_actual_tokens
       - `max_query_len` = plan.max_query_len
       - `slot_mapping` = plan.slot_mapping
       - `seq_lens` / `max_seq_len` / `block_table_tensor` = 共享的 base 值
    3. 检查是否与上一层 metadata 参数相同：
       - 相同 → 直接复用上一层 metadata dict
       - 不同 → 调用 `builder.build()` 生成新 metadata
    4. 返回 `list[dict[str, metadata]]`，索引 = layer_idx

**验证方式**：
- 构造两层plan，第二层与第一层replay窗口相同 → 验证复用同一metadata对象
- 构造两层plan，第二层replay窗口不同 → 验证生成不同metadata

**依赖**：Step 4

---

#### Step 6: hidden-state store 语义适配 — "目标层输入" 捕获

**目标**：在新模式下，将 hidden-state 捕获从 "layernorm hook" 改为 "layer 输入/输出显式捕获"。

**改动文件**：
- `vllm/v1/worker/layer_recompute.py` / `LayerRecomputeManager`
  - 新增方法 `capture_layer_input_d2h()`：
    ```python
    def capture_layer_input_d2h(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,       # 只含 scheduled tokens
        req_indices: np.ndarray,
        positions: np.ndarray,
    ):
    ```
    - 与现有 `_layernorm_hook` 的 D2H 逻辑相同（async copy + event）
    - 但写入 `cpu_layer_inputs_by_layer[layer_idx]` 而非 `cpu_attn_inputs_by_layer[layer_idx]`
    - 只对 scheduled tokens 执行（replay tokens 的历史数据已在 store 中）
  - 新增方法 `load_cpu_fill_h2d()`：
    ```python
    def load_cpu_fill_h2d(
        self,
        layer_idx: int,
        cpu_fill_positions: np.ndarray,
        cpu_fill_logical_ids: np.ndarray,
        cpu_fill_block_offsets: np.ndarray,
    ) -> torch.Tensor:
    ```
    - 从 `cpu_layer_inputs_by_layer[layer_idx]` 中按 logical_id + offset 取出 hidden states
    - 拼成连续 tensor，async H2D 到 GPU
    - 返回 GPU tensor `[cpu_fill_token_count, hidden_size]`
  - 在新模式下，跳过 layernorm hook 注册（不调用 `register_layernorm_hooks`）

**验证方式**：
- 手动调用 capture + load，验证 D2H → H2D 的数据完整性
- 验证 layernorm hook 在新模式下不触发

**依赖**：Step 3

---

#### Step 7: OPT model forward 动态路径 — 核心执行逻辑

**目标**：在 `OPTDecoder.forward` 中增加动态 replay 路径。这是最核心的 step。

**改动文件**：
- `vllm/model_executor/models/opt.py`
  - `OPTDecoder.forward` 修改：
    ```python
    def forward(self, ...):
        # ... existing embed + pos + project_in ...
        
        runtime = get_forward_context().layer_recompute_runtime
        if runtime is not None:
            return self._forward_dynamic_replay(hidden_states, runtime)
        
        # ... existing layer loop (unchanged) ...
    ```
  - 新增 `OPTDecoder._forward_dynamic_replay()`：
    ```python
    def _forward_dynamic_replay(self, hidden_states, runtime):
        scheduled_hs = hidden_states  # [scheduled_tokens, hidden_size]
        replay_hs = None              # 上一层的 replay 输出（layer 0 无）
        
        for layer_idx, layer in enumerate(self.layers[start:end]):
            # plan 和 metadata 由 pre-hook 逐层填充到 runtime 中
            # layer 0 的 plan 在 step 开始前已准备好
            # layer i>0 的 plan 在 layer i-1 的 pre-hook 中准备好
            plan = runtime.get_layer_plan(layer_idx)
            
            if plan.replay_token_count == 0:
                # 该层无 replay，直接走原 layer forward
                # 设置该层 attn_metadata 到 forward context
                _set_layer_attn_metadata(runtime, layer_idx)
                scheduled_hs = layer(scheduled_hs)
                # 捕获 scheduled_hs → CPU store for next layer
                runtime.capture_layer_output(layer_idx, scheduled_hs)
                replay_hs = None
                continue
            
            # 1. 构造当前层 replay 输入
            cpu_fill_hs = None
            if plan.cpu_fill_token_count > 0:
                cpu_fill_hs = runtime.load_cpu_fill(layer_idx, plan)
            
            gpu_reuse_hs = None
            if plan.gpu_reuse_token_count > 0 and replay_hs is not None:
                gpu_reuse_hs = _slice_gpu_reuse(replay_hs, plan)
            
            current_replay_input = _concat_replay_input(cpu_fill_hs, gpu_reuse_hs)
            
            # 2. Pack: replay + scheduled → combined
            combined_hs = torch.empty(
                plan.num_actual_tokens, hidden_size, ...)
            combined_hs[plan.combined_replay_indices] = current_replay_input
            combined_hs[plan.combined_scheduled_indices] = scheduled_hs
            
            # 3. 设置该层 attn_metadata，执行 layer forward
            _set_layer_attn_metadata(runtime, layer_idx)
            combined_out = layer(combined_hs)
            
            # 4. Unpack: combined → replay + scheduled
            replay_hs = combined_out[plan.combined_replay_indices]
            scheduled_hs = combined_out[plan.combined_scheduled_indices]
            
            # 5. 捕获 scheduled → CPU store
            runtime.capture_layer_output(layer_idx, scheduled_hs)
        
        # final_layer_norm + project_out（只对 scheduled_hs）
        if self.final_layer_norm is not None:
            scheduled_hs = self.final_layer_norm(scheduled_hs)
        if self.project_out is not None:
            scheduled_hs = self.project_out(scheduled_hs)
        return scheduled_hs
    ```

**关键实现细节**：
- `_set_layer_attn_metadata()`：临时替换 `ForwardContext.attn_metadata` 为当前层的 metadata dict
- `_slice_gpu_reuse()`：对每个 req，从 `replay_hs` 中按 `gpu_reuse_slice_per_req` 切片并拼接
- `_concat_replay_input()`：`torch.cat([cpu_fill_hs, gpu_reuse_hs], dim=0)` 或处理其中一个为 None 的情况

**验证方式**：（见 Step 10 集成测试）

**依赖**：Step 4, 5, 6

---

#### Step 8: gpu_model_runner 集成 — 流水线式 pre-hook + step 执行流串联

**目标**：在 `gpu_model_runner` 的 step 执行流中，实现流水线式的 per-layer plan 获取、异步预取和 runtime 注入。

**改动文件**：
- `vllm/v1/worker/gpu_model_runner.py`
  - 新增成员 `replay_plan_provider: ReplayPlanProvider`
    - 初始化时根据 config 选择 `StaticReplayPlanProvider`（默认）或将来的自定义 provider
  - 在 `_execute_model_step()` 或等价方法中，新模式下：
    1. 调用 `provider.get_layer_plan(0, ...)` 获取 layer 0 的 plan
    2. 启动 layer 0 的 IO prefix KV 异步加载
    3. 启动 layer 0 的 replay hidden states H2D（如果有 cpu_fill）
    4. 构造 layer 0 的 attention metadata
    5. 构造 `OPTDynamicReplayRuntime`（初始只含 layer 0 的 plan/metadata）并赋值到 `ForwardContext`
  - 重写 `_runkv_pre_hook()` 的新模式分支：
    ```python
    if self.kv_offload_config.layer_recompute_mode == "prev_layer_output_dynamic":
        runtime = get_forward_context().layer_recompute_runtime
        
        # 1. 等待当前层(i)的数据就绪
        mapper.sync_load_layer(layer_idx)                    # IO prefix KV
        manager.sync_cpu_fill_h2d(layer_idx)                 # replay HS（如果有）
        
        # 2. 获取下一层(i+1)的 replay plan
        if layer_idx + 1 < num_layers:
            next_plan = self.replay_plan_provider.get_layer_plan(
                layer_idx=layer_idx + 1,
                ...,
                prev_layer_plan=runtime.current_layer_plan(layer_idx),
            )
            runtime.set_layer_plan(layer_idx + 1, next_plan)
            
            # 3. 异步启动下一层(i+1)的数据预取
            next_skip_blocks = manager.compute_skip_block_ids_for_layer(
                layer_idx + 1, next_plan)
            mapper.load_layer_async(
                next_layer_name, layer_idx + 1,
                skip_block_ids=next_skip_blocks)
            
            if next_plan.cpu_fill_token_count > 0:
                manager.load_cpu_fill_h2d_async(
                    layer_idx + 1, next_plan)
            
            # 4. 构造下一层(i+1)的 attention metadata
            next_metadata = self._build_layer_attn_metadata(
                next_plan, base_seq_lens, ...)
            runtime.set_layer_metadata(layer_idx + 1, next_metadata)
        
        return  # 不做 recompute_kv_for_layer
    ```

**流水线重叠效果**：
```
Layer i pre-hook:  sync(i) → get_plan(i+1) → async_load(i+1) → build_meta(i+1)
Layer i forward:   [GPU compute layer i]  ← 与 layer i+1 的 IO/H2D 并行
Layer i+1 pre-hook: sync(i+1) → get_plan(i+2) → async_load(i+2) → ...
```

**验证方式**：
- 启动新模式，验证 pre-hook 不再调用 recompute_kv_for_layer
- 验证 layer i+1 的 IO 加载与 layer i 的 GPU 计算并行（通过 nsys profile 可视化验证）
- 验证 ForwardContext 中 runtime 被正确设置，每层 plan/metadata 按需填充

**依赖**：Step 4, 5, 7

---

#### Step 9: 单元测试 — replay plan 计算正确性（含随机 plan 测试）

**目标**：验证 `compute_layer_replay_plan_for_layer()` 的核心算法正确性，以及 `RandomReplayPlanProvider` 生成的随机 plan 在各种边界条件下的正确性。

**新建文件**：`tests/v1/kv_offload/test_opt_dynamic_replay_plan.py`

**测试用例**：

```
Test 1: 当前层 replay 更短（静态 provider）
  io_prefix_blocks = [2, 4]  # layer 0: 2 blocks, layer 1: 4 blocks
  computed_len = 128, block_size = 16
  layer 0: replay [32, 128), prev_gpu = 128 → cpu_fill [32, 128), gpu_reuse empty
  layer 1: replay [64, 128), prev_gpu = 32  → cpu_fill empty, gpu_reuse = layer 0 output 的后缀 [64, 128)
  验证：layer 1 的 cpu_fill_token_count == 0

Test 2: 当前层 replay 相等（静态 provider）
  io_prefix_blocks = [2, 2]
  验证：layer 1 完全复用，cpu_fill_token_count == 0, gpu_reuse == full

Test 3: 当前层 replay 更长（静态 provider）
  io_prefix_blocks = [4, 2]
  验证：layer 1 cpu_fill_token_count > 0 且 gpu_reuse_token_count > 0

Test 4: 随机 plan 多层交替（随机 provider）
  使用 RandomReplayPlanProvider(num_layers=8, max_blocks=8, seed=42)
  验证每层只按自己窗口工作，无全局传播
  验证所有层的 plan 满足不变量：
    replay_token_count == cpu_fill_token_count + gpu_reuse_token_count
    num_actual_tokens == replay_token_count + scheduled_token_count

Test 5: 随机 plan 多 request 长度不均（随机 provider）
  req 0: computed_len = 128, req 1: computed_len = 64
  使用 RandomReplayPlanProvider 生成 plan
  验证 query_start_loc / slot_mapping 正确排布

Test 6: replay 长度为 0
  io_prefix_blocks[layer] 足够大 → 该层无 replay
  验证 plan.replay_token_count == 0

Test 7: pack/unpack 对称性（随机 provider）
  用 RandomReplayPlanProvider 生成多层 plan
  对每层：构造 combined → unpack → 验证与原始 replay/scheduled 完全一致

Test 8: slot_mapping 正确性
  验证 slot = mapper_mapping[logical_id] * block_size + pos % block_size
  验证 per-req 排列顺序：replay 在前、scheduled 在后

Test 9: provider 接口合规性
  StaticReplayPlanProvider 和 RandomReplayPlanProvider 都满足 ReplayPlanProvider Protocol
  get_layer_plan 可以逐层调用，prev_layer_plan 正确传递
```

**依赖**：Step 4

---

#### Step 10: 单元测试 — per-layer metadata 构造

**目标**：验证 per-layer metadata 构造逻辑。

**新建文件**：`tests/v1/kv_offload/test_opt_dynamic_replay_metadata.py`

**测试用例**：

```
Test 1: 两层 replay 窗口相同 → metadata 对象复用
  验证 layer_0_metadata is layer_1_metadata

Test 2: 两层 replay 窗口不同 → metadata 不同
  验证 layer_0_metadata is not layer_1_metadata
  验证各自的 query_start_loc / num_actual_tokens / slot_mapping 正确

Test 3: query_start_loc 正确性
  2 reqs, layer 有 replay + scheduled
  验证 query_start_loc[0] == 0
  验证 query_start_loc[1] == req0_replay + req0_scheduled
  验证 query_start_loc[2] == total tokens

Test 4: slot_mapping 与 block_table 一致性
  构造已知 block_table + mapper_mapping
  验证 slot_mapping 中每个 slot 的计算正确
```

**依赖**：Step 5

---

#### Step 11: 集成测试 — 小 OPT 模型端到端 correctness

**目标**：验证新模式在真实 OPT 模型上的输出与 baseline 一致。

**新建文件**：`tests/v1/kv_offload/test_opt_dynamic_replay_e2e.py`

**测试用例**：

```
Test 1: 基础 correctness（静态 provider）
  模型：facebook/opt-125m (pre-LN)
  配置：runkv + layer_recompute + prev_layer_output_dynamic
  io_prefix_blocks: [2]（单值广播到所有层）
  单 request，prompt 128 tokens，生成 32 tokens
  对比：新模式 output vs 旧模式 output → token 完全一致

Test 2: 交替 io_prefix_blocks（静态 provider）
  io_prefix_blocks: 逐层不同（如 [2,4,2,4,...] 按层数展开）
  验证 output 与 baseline 一致

Test 3: 随机 plan（随机 provider）
  使用 RandomReplayPlanProvider(seed=42) 生成完全随机的 per-layer plan
  验证 output 与 baseline（无 replay，全 IO）一致
  用多个 seed 重复验证

Test 4: 并发多 request（随机 provider）
  3 个不同长度 request 并发
  使用 RandomReplayPlanProvider
  验证每个 request output 与单独运行一致

Test 5: 纯 decode step
  prompt 已跑完，每步只 decode 1 token
  验证 replay 窗口正确（decode step 中 computed_len 持续增长）

Test 6: extend（multi-token scheduled）
  一次 schedule 多个新 token
  验证 pack/unpack 在 scheduled > 1 时正确

Test 7: replay 窗口从短变长
  先运行几步（KV cache 积累），再触发 replay 窗口变长
  验证 CPU fill 补洞正确

Test 8: 流水线预取验证
  验证 layer i 计算时 layer i+1 的 IO/H2D 已异步启动
  通过 nsys profile 或手动 event 计时验证重叠
```

**依赖**：Step 8（所有代码就绪）

---

#### Step 12: 负向测试 — 配置不满足时的报错

**新建文件**：`tests/v1/kv_offload/test_opt_dynamic_replay_negative.py`

**测试用例**：

```
Test 1: 非 OPT 模型 → ValueError
Test 2: post-LN OPT (opt-350m) → ValueError（若方便获取模型配置）
Test 3: cudagraph 打开 → ValueError
Test 4: DCP 打开 → ValueError
Test 5: cascade attention 打开 → ValueError
Test 6: ubatching 打开 → ValueError
Test 7: 多 KV group → ValueError
```

所有 test 验证错误消息包含有意义的描述（不只是 generic error）。

**依赖**：Step 2

---

### Step-by-Step 测试方案（验证顺序）

测试按 **从底向上** 的顺序执行，确保每层都在下层已验证的基础上运行。

| 阶段 | 测试类型 | 对应 Step | 运行命令 | 通过标准 |
|------|---------|-----------|---------|---------|
| T1 | 配置解析 | Step 1 | `pytest tests/v1/kv_offload/test_opt_dynamic_replay_negative.py -k "config"` | CLI arg 正确解析，非法值报错 |
| T2 | 前置校验 | Step 2, 12 | `pytest tests/v1/kv_offload/test_opt_dynamic_replay_negative.py` | 所有非法组合报 ValueError |
| T3 | 数据结构 | Step 3 | `python -c "from vllm.v1.worker.opt_dynamic_replay import LayerReplayPlan, OPTDynamicReplayRuntime, ReplayPlanProvider"` | 导入成功 |
| T4 | Plan 算法 | Step 4, 9 | `pytest tests/v1/kv_offload/test_opt_dynamic_replay_plan.py -v` | 9 个 test 全过（含随机 plan） |
| T5 | Metadata 构造 | Step 5, 10 | `pytest tests/v1/kv_offload/test_opt_dynamic_replay_metadata.py -v` | 4 个 test 全过 |
| T6 | HS store 适配 | Step 6 | `pytest tests/v1/kv_offload/test_opt_dynamic_replay_plan.py -k "h2d or d2h"` | capture + load 数据一致 |
| T7 | 端到端 correctness | Step 11 | `pytest tests/v1/kv_offload/test_opt_dynamic_replay_e2e.py -v --timeout 300` | 所有 output 与 baseline 一致（含随机 plan） |
| T8 | metadata build 性能 | Step 5 | 手动 profile：`python -c "..."` 打印每次 build 耗时 | 单次 build < 100μs |

**阶段间依赖关系**：

```
T1 → T2 → T3 → T4 → T5 → T6 → T7
                                 ↑
                            T8（可并行）
```

**CI 集成建议**：
- T1-T6 加入常规 CI（无需 GPU 或只需小 GPU）
- T7 加入 GPU CI（需要 OPT-125m 模型权重，~500MB）
- T8 为手动 profiling，不加入 CI

**回归保护**：
- 旧模式 `io_hidden_states` 的所有现有测试保持不变
- 新模式测试在 `tests/v1/kv_offload/test_opt_dynamic_replay_*.py` 下独立组织
- 任何对 `layer_recompute.py` / `gpu_model_runner.py` / `opt.py` 的修改都会触发两套测试
