# Layer-wise KV Cache Recompute + RunKV IO Hybrid（V1 Llama/OPT / 单机单卡）

本文档是 **decision-complete** 的实现规格：agent 只要照此文档修改/新增对应代码，即可完成第一版功能（能跑、正确、可测）。

---

## 0. V1 范围与硬约束（必须满足，否则直接报错/禁用）

### 0.1 仅支持的运行形态

- **单机单卡**：仅支持 `PP=1, TP=1`，且不启用任何 DCP/PCP 等 CP 相关 interleave（即 `total_cp_world_size==1`）。
- **必须启用 RunKV**：本功能只在 `kv_offload_config.enabled=True` 时可用。
- **仅 Llama/OPT 系模型**：支持 `llama` 与 `opt` 的 decoder-only 结构。
- **仅 decoder self-attention（Llama/OPT 结构）**：V1 只做“能工作且对齐代码事实”的实现，不做跨模型泛化、不做 MLA、不做 Q-recompute。
- **仅单一 KV cache group（gid=0）**：V1 直接 `assert len(kv_cache_config.kv_cache_groups) == 1`，避免多 group（例如 cross-attn）带来的索引与生命周期复杂度。

### 0.2 V1 不做的事

- 不支持 MLA（DeepSeek V2/V3）、不支持 encoder/ cross-attention。
- 不做 Q-recompute，不做 per-step 自适应策略。
- 不做多机/多卡。

---

## 1. 背景：RunKV 现状（代码事实）

### 1.1 RunKV staging 的核心对象

- `vllm/v1/worker/gpu_model_runner.py::PagedBlockMapper`
  - `prepare_step()`：为当前 step 的 active requests 收集 **全历史** blocks（attention 需要完整 history），并为这些 logical blocks 分配 staging slot（`mapping: logical_id -> staging_slot`）。
  - `load_layer_async()` / `sync_load_layer()`：在 `load_stream` 上做 CPU→GPU staging copy，并用 event 让 default stream wait。
  - `flush_layer_async()` / `sync_all_offloads()`：把 dirty blocks（本 step 写入的新 KV）从 staging 回写到 CPU KV backing store。

### 1.2 RunKV hooks 的挂载点

- RunKV hooks 挂在 **vLLM 的 `Attention` module** 上（不是 decoder layer）。
- 现有 `_runkv_pre_hook` 时序是：layer0 启动 load → sync 当前层 → prefetch 下一层。

### 1.3 `reshape_and_cache_flash`

- Python wrapper：`vllm/_custom_ops.py::reshape_and_cache_flash`
- CUDA kernel：`csrc/cache_kernels.cu::reshape_and_cache_flash`
- 功能：根据 `slot_mapping[token_i] = staging_slot*block_size + offset`，把 K/V 写到 staging KV cache 的对应 slot。

---

## 2. 目标：每层 IO prefix blocks + recompute suffix blocks（V1 定义）

### 2.1 IO/Recompute 的语义（V1 固定为“最早前缀 IO”）

对每个 attention layer（按 layer_idx）给定 `io_prefix_blocks[layer_idx]`：

- **IO blocks（最早前缀）**：该请求中所有 token position 满足：
  - `position // block_size < io_prefix_blocks[layer_idx]`
  - 这些 blocks **永远走 RunKV 的 CPU→GPU KV load**。
- **Recompute blocks（历史后缀）**：该请求中所有 token position 满足：
  - `position // block_size >= io_prefix_blocks[layer_idx]`
  - 对这些 blocks：本 layer 进入时 **跳过 KV load**，改为从 CPU 保存的 attention 输入 hidden states 重算 K/V 并写入 staging。

> 注意：本设计的 N（IO prefix）按 **block** 为单位配置；V1 强制 block 对齐，不做 block 内混合。

### 2.2 scheduled / dirty 的定义（与 RunKV 保持一致）

- **scheduled tokens**：本 step 实际 forward 的 tokens（`GPUModelRunner._prepare_inputs()` 生成的 `positions_np`/`req_indices` 对应的 tokens）。
- **dirty blocks**：包含本 step scheduled tokens 的 blocks（`PagedBlockMapper.prepare_step()` 已计算并返回），需要 flush。
- **recompute blocks 必须永远不包含 dirty blocks**：因为 dirty blocks 的 KV 将由本 step 正常 attention forward 写入 staging 并 flush 回 CPU。

---

## 3. V1 关键思想：把 recompute 并入 `_runkv_pre_hook` 实现 overlap

### 3.1 为什么不挂 DecoderLayer pre/post hook（V1 选择）

`positions` 参数只包含 scheduled tokens；而 recompute 的对象是 **历史 blocks**。把 recompute 做成 DecoderLayer hook 容易误用 `positions` 并造成实现与代码事实不一致。

V1 直接修改 `GPUModelRunner._runkv_pre_hook`：
- 先把“当前层需要的两类输入”异步搬运到 GPU：`prefix KV`（RunKV load_stream）+ `suffix hidden states`（manager H2D stream）。
- 当前层进入 attention 前只做一次就绪栅栏；随后立即启动“下一层 prefix KV + suffix HS 预取”，让它与**当前层 recompute + forward** 重叠。

### 3.2 `_runkv_pre_hook` 的新时序（必须实现）

对 layer N：

1) 计算 `skip_block_ids`（本 layer 要 recompute 的 logical block id 集合）
2) 若 `layer_idx==0`：`mapper.load_layer_async(layer_name, layer_idx, skip_block_ids=skip_block_ids)`
3) `manager.prefetch_recompute_inputs_for_layer(...)`（把当前层 suffix HS/pos/slot 先搬到 GPU）
4) `mapper.sync_load_layer(layer_idx)`（确保当前层 IO prefix blocks ready）
5) prefetch 下一层（在本层 recompute 前发起）：
   - 先计算 `skip_next`（下一层要 recompute 的 blocks；仅用于 IO skip，不在本层 pre_hook 做下一层的 recompute）
   - `next_mapper.load_layer_async(next_layer_name, next_layer_idx, skip_block_ids=skip_next)`
   - `manager.prefetch_recompute_inputs_for_layer(..., layer_idx=next_layer_idx, skip_block_ids=skip_next)`
6) `manager.recompute_kv_for_layer(...)`（消费当前层预取好的 HS，重算并写入 skip blocks）

> 关键正确性约束：`load_layer_async(..., skip_block_ids)` 必须确保 IO copy 不写入 skip blocks 对应的 staging slots；recompute 只写入 skip blocks 对应 slots。两者写入区域必须互斥，否则有数据竞争。

推荐实现伪代码（必须保持顺序语义）：

```python
def _runkv_pre_hook(..., layer_name: str, layer_idx: int, gid: int):
    mapper = self.paged_block_mappers[gid]
    dirty = self.paged_dirty_blocks[gid]

    if self.layer_recompute_enabled:
        skip = self.layer_recompute_manager.compute_skip_block_ids_for_layer(
            layer_idx=layer_idx,
            gid=gid,
            mapper=mapper,
            dirty_blocks=dirty,
            io_prefix_blocks=self.kv_offload_config.layer_recompute_io_prefix_blocks,
        )
    else:
        skip = None

    if layer_idx == 0:
        mapper.load_layer_async(layer_name, layer_idx, skip_block_ids=skip)

    if self.layer_recompute_enabled and skip:
        self.layer_recompute_manager.prefetch_recompute_inputs_for_layer(
            layer_idx=layer_idx,
            layer_name=layer_name,
            gid=gid,
            mapper=mapper,
            skip_block_ids=skip,
        )

    mapper.sync_load_layer(layer_idx)

    next_info = self._get_next_layer_info(layer_idx)
    if next_info is not None:
        next_layer_name, next_layer_idx, next_gid = next_info
        next_mapper = self.paged_block_mappers[next_gid]
        if self.layer_recompute_enabled:
            skip_next = self.layer_recompute_manager.compute_skip_block_ids_for_layer(
                layer_idx=next_layer_idx,
                gid=next_gid,
                mapper=next_mapper,
                dirty_blocks=self.paged_dirty_blocks[next_gid],
                io_prefix_blocks=self.kv_offload_config.layer_recompute_io_prefix_blocks,
            )
        else:
            skip_next = None
        next_mapper.load_layer_async(
            next_layer_name, next_layer_idx, skip_block_ids=skip_next
        )
        if self.layer_recompute_enabled and skip_next:
            self.layer_recompute_manager.prefetch_recompute_inputs_for_layer(
                layer_idx=next_layer_idx,
                layer_name=next_layer_name,
                gid=next_gid,
                mapper=next_mapper,
                skip_block_ids=skip_next,
            )

    if self.layer_recompute_enabled and skip:
        self.layer_recompute_manager.recompute_kv_for_layer(
            layer_idx=layer_idx,
            layer_name=layer_name,
            gid=gid,
            mapper=mapper,
            attn_module=module,  # vLLM Attention
            skip_block_ids=skip,
        )
```

---

## 4. CPU 侧需要新增的数据（V1 存什么、为何足够）

### 4.1 V1 只存 “Attention 输入”（即 `input_layernorm` 输出）

对 decoder layer（Llama/OPT），attention 的输入是：

- `hidden_states_normed`：`LlamaDecoderLayer.input_layernorm(...)` 的输出 hidden states（注意 residual 存在时返回 tuple）

V1 选择只存这一份张量的原因：

- recompute 的目标是重建 **K/V**，在 Llama/OPT 中都可从 attention 前 layernorm 输出通过 `qkv_proj` 重建（Llama 额外需要 rotary）。
- 不需要保存 `residual`，也不需要保存 decoder layer 的输出 hidden states。

### 4.2 CPU 存储结构（必须实现）

在 `GPUModelRunner.initialize_kv_cache_tensors()`（RunKV 初始化完成后）分配以下 CPU pinned buffers：

1) **每层的 attention 输入 hidden states**

- `cpu_attn_inputs_by_layer: list[torch.Tensor]`
- `cpu_attn_inputs_by_layer[layer_idx]` shape：
  - `[num_cpu_blocks, block_size, hidden_size]`
- dtype：
  - 与模型 dtype 一致（例如 fp16/bf16），用来给 `LlamaAttention.qkv_proj` 作为输入。
- device/pin：
  - `device="cpu", pin_memory=True`

2) **每个 logical block 的 token position 元数据**

- `cpu_block_positions: torch.Tensor`
- shape：`[num_cpu_blocks, block_size]`
- dtype：`torch.int32`
- 初始化为 `-1`（表示该 offset 尚无 token）

3) **每个 logical block 的有效长度**

- `cpu_block_valid_lens: torch.Tensor`
- shape：`[num_cpu_blocks]`
- dtype：`torch.int16` 或 `torch.int32`
- 初始化为 `0`

4) **logical block 的 owner（防止 block 被 scheduler 复用导致的脏读）**

- `logical_id_owner_req_id: dict[int, str]`
- 含义：logical_id 当前属于哪个 `req_id`（来自 `InputBatch.req_ids`）
- 每 step 更新；当发现 owner 变化时必须 reset 对应 block 的元数据（见 4.3）。

> 说明：这里只需要 reset `cpu_block_positions` 与 `cpu_block_valid_lens`。`cpu_attn_inputs_by_layer` 不强制清零，因为只要 valid_lens/positions 正确，就不会读取到旧数据。

### 4.3 owner 变化时的 reset 规则（必须实现）

每个 step（在 `GPUModelRunner._prepare_inputs()` 内）构造当前 batch 的 `logical_id -> req_id` 映射（最多 `capacity` 个 blocks，成本可接受）：

- 输入：
  - `logical_block_table_np = input_batch.block_table.block_tables[gid].get_numpy_array()`
  - `num_blocks_per_row = input_batch.block_table.block_tables[gid].num_blocks_per_row`
  - `req_ids = input_batch.req_ids`（`list[str]`，row index 与 block_table 行对应）
- 遍历 `row in [0..num_reqs)` 和 `col in [0..num_blocks_per_row[row])`：
  - `logical_id = int(logical_block_table_np[row, col])`
  - `req_id = req_ids[row]`

对每个出现的 `logical_id`：

- 若 `logical_id_owner_req_id.get(logical_id) != req_id`：
  - `logical_id_owner_req_id[logical_id] = req_id`
  - `cpu_block_positions[logical_id, :].fill_(-1)`
  - `cpu_block_valid_lens[logical_id] = 0`

> 该规则能避免因为 InputBatch 行交换/compact 等导致的误判：owner 用 req_id（稳定标识），不是用 row index。

---

## 5. Step 级别数据准备（让 hooks 无需猜测）

### 5.1 在 `_prepare_inputs()` 中缓存 scheduled tokens 的 block 索引（必须实现）

位置：`vllm/v1/worker/gpu_model_runner.py::GPUModelRunner._prepare_inputs()`，在计算出：

- `req_indices: np.ndarray`（shape `[total_num_scheduled_tokens]`）
- `positions_np: np.ndarray`（shape `[total_num_scheduled_tokens]`，int32）

之后，新增一段仅对 recompute 使用的 **向量化** 计算（不允许 per-token Python for-loop）：

对每个 KV cache group `gid`（V1 可先只做 `gid=0`，并 assert 只有一个 group）：

- `block_size = self.input_batch.block_table.block_tables[gid].block_size`
- `logical_table = self.input_batch.block_table.block_tables[gid].get_numpy_array()`
- `block_indices = positions_np // block_size`
- `block_offsets = positions_np % block_size`
- `logical_ids = logical_table[req_indices, block_indices]`（shape `[num_tokens]`，int32）

把这些 per-token 数组缓存到 runner（供 layernorm hook 使用）：

- `self._lr_req_indices_np = req_indices`
- `self._lr_positions_np = positions_np`
- `self._lr_logical_ids_np = logical_ids`
- `self._lr_block_offsets_np = block_offsets`
- `self._lr_block_size = block_size`
- `self._lr_num_reqs = num_reqs`

> 这些缓存只在当前 step 有效，step 结束后可清空或覆盖。

---

## 6. 保存 scheduled tokens 的 hidden states（V1 hook：挂 `input_layernorm`）

### 6.1 hook 挂载点与如何定位 layer_idx（必须实现）

V1 支持 Llama/OPT，通过模型结构定位每层：

- Llama：`decoder_layers = self.model.model.layers`，hook 到 `decoder_layer.input_layernorm`
- OPT：`decoder_layers = self.model.model.decoder.layers`，hook 到 `decoder_layer.self_attn_layer_norm`

该 hook 的职责是：把本 step 的 scheduled tokens 的 `hidden_states_normed` 写入 CPU store。

### 6.2 hook 输入输出的准确处理（必须实现）

`RMSNorm.forward`（`vllm/model_executor/layers/layernorm.py::RMSNorm`）在 residual 存在时会返回 `(hidden_states_normed, residual_out)`。

因此 hook 必须这样取 norm 输出：

- `out = output`
- `hidden_states_normed = out if isinstance(out, torch.Tensor) else out[0]`

### 6.3 scheduled tokens 全量保存（必须实现）

hook 内必须保存本 step 的全部 scheduled tokens（包含 prefix 与 suffix）：

- 后续每个 step 的 recompute 策略可能变化，不能假设 prefix 永远不参与 recompute。
- 因此“KV block 可用”必须等价于“HS block 可用”，两者容量与内容覆盖范围保持一致。

### 6.4 hook 写入 CPU store 的精确算法（必须实现）

在 hook 内（伪代码，必须按语义实现；允许实现细节不同）：

1) 直接使用本 step 全量数组（不做 prefix 过滤）：
   - `logical_ids = self._lr_logical_ids_np`
   - `offsets = self._lr_block_offsets_np`
   - `positions = self._lr_positions_np`
   - `hs = hidden_states_normed`
3) **D2H**：
   - 把 `hs` non_blocking copy 到 CPU pinned 的临时 buffer（或直接 copy 到目标切片，二者择一）
   - 必须在 `LayerRecomputeManager` 内部使用专用 `hs_d2h_stream`（`torch.cuda.Stream`）来做 D2H，并在 stream 末尾 record event：
     - `self._hs_d2h_event_by_layer[layer_idx] = torch.cuda.Event(); event.record(hs_d2h_stream)`
4) scatter 写入：
   - `cpu_attn_inputs_by_layer[layer_idx][logical_ids, offsets, :] = hs_cpu`
   - `cpu_block_positions[logical_ids, offsets] = positions`
5) 更新 `cpu_block_valid_lens`：
   - 对本次写入涉及到的每个 `logical_id`：
     - `cpu_block_valid_lens[logical_id] = max(cpu_block_valid_lens[logical_id], max(offsets_for_block)+1)`

> 注意：更新 valid_lens 时必须以 owner/reset 规则（4.3）为前提，否则 block 复用会导致 max 永远不降，产生脏读。V1 要求严格执行 4.3 的 reset。

### 6.5 step 结束时的同步（必须实现，否则可能读到未完成的 D2H）

为了保证“下一 step 的 recompute 能读到上一 step 保存的 hidden states”，在 `GPUModelRunner.execute_model()` 中：

- 在 step 末尾调用统一同步逻辑（例如 `_sync_runkv_step_end_state()`）：
  - `self._sync_all_runkv_offloads()`
  - 若启用 layer recompute，再调用 `LayerRecomputeManager.sync_hs_d2h()`：
    - 对本 step 所有 layers 的 `hs_d2h_event_by_layer` 执行 `event.synchronize()` 并清理

该同步是正确性要求（否则下一 step 可能在 D2H 未完成时就开始 recompute）。

---

### 6.6 `LayerRecomputeManager` 的最小 API（必须实现）

新增文件：`vllm/v1/worker/layer_recompute.py`

V1 要求 manager 至少提供以下接口（名字可调整，但调用点与语义必须一致）：

- `__init__(...)`
  - 输入：`device`、`kv_offload_config`、`kv_cache_config`（用于 `num_cpu_blocks`）、`hidden_size`、`block_size`、`num_layers`、`model`（LlamaForCausalLM 或 OPTForCausalLM）等
  - 分配 CPU pinned buffers：`cpu_attn_inputs_by_layer/cpu_block_positions/cpu_block_valid_lens`
  - 初始化：`logical_id_owner_req_id: dict[int,str]`
  - 初始化 D2H stream 与 event 容器

- `register_layernorm_hooks(gpu_model_runner: GPUModelRunner) -> None`
  - 给每个 decoder layer 的 attention 前 layernorm 注册 hook（Llama: `input_layernorm`，OPT: `self_attn_layer_norm`）
  - hook 内使用 `gpu_model_runner` 暴露的 step 缓存（5.1）

- `begin_step(...) -> None`
  - 在 `_prepare_inputs()` 内调用（或由 runner 写入 manager 字段）
  - 功能：
    - 保存本 step 的 `req_ids/req_indices/positions/logical_ids/offsets`
    - 更新 owner 并按 4.3 reset 元数据

- `recompute_kv_for_layer(*, layer_idx: int, layer_name: str, gid: int, mapper: PagedBlockMapper, attn_module: Attention, skip_block_ids: set[int]) -> None`
  - 在 `_runkv_pre_hook` 内调用
  - 消费 `prefetch_recompute_inputs_for_layer` 产物并执行 7.5 的 recompute 写 staging（不再在内部重复计算 skip）

- `prefetch_recompute_inputs_for_layer(*, layer_idx: int, layer_name: str, gid: int, mapper: PagedBlockMapper, skip_block_ids: set[int]) -> None`
  - 在 `_runkv_pre_hook` 内调用
  - 负责把 suffix tokens 的 `hidden_states/positions/slots` 预取到 GPU（异步 H2D）
  - 允许重复调用（同一层同一步只保留一份缓存）

- `compute_skip_block_ids_for_layer(*, layer_idx: int, gid: int, mapper: PagedBlockMapper, dirty_blocks: set[int], io_prefix_blocks: list[int]) -> set[int]`
  - 在 `_runkv_pre_hook` 内调用，用于：
    - 当前层：传给 `load_layer_async(..., skip_block_ids=...)`
    - 下一层 prefetch：传给 `next_mapper.load_layer_async(..., skip_block_ids=skip_next)`
  - **只计算 skip 集合，不做任何 GPU compute**（不写 staging）

- `sync_hs_d2h() -> None`
  - 在 step 结束时调用（`execute_model()` 内）
  - 同步所有 D2H events，确保下一 step recompute 读取的是已完成的数据

> 重要：V1 不允许 manager 在 hook 内依赖 “下一层/上一层”的数据流转，也不允许把 recompute tokens 注入主 forward。manager 只做存储（layernorm hook）与填 staging（pre_hook）。

## 7. recompute K/V：在 `_runkv_pre_hook` 中执行（V1 核心）

### 7.1 如何拿到 qkv_proj / rotary_emb / 参数（必须实现）

在 `_runkv_pre_hook` 中，`module` 是 vLLM 的 `Attention`（`vllm/attention/layer.py::Attention`），它没有 `qkv_proj`。

对 Llama，必须通过 layer_idx 找到 `LlamaAttention` wrapper：

- `llama_attn = self.model.model.layers[layer_idx].self_attn`  （类型 `LlamaAttention`）
- 使用：
  - `llama_attn.qkv_proj`
  - `llama_attn.q_size / llama_attn.kv_size`
  - `llama_attn.rotary_emb`

KV cache quant scales 必须来自 vLLM Attention（也就是 `_runkv_pre_hook` 的 `module`）：

- `k_scale = module._k_scale`
- `v_scale = module._v_scale`
- `kv_cache_dtype_str = module.kv_cache_dtype`

### 7.2 recompute 写入的 staging buffer 获取方式（必须实现）

在 pre_hook 内：

- `forward_context = vllm.forward_context.get_forward_context()`
- `kv_cache = module.kv_cache[forward_context.virtual_engine]`
- `key_cache, value_cache = kv_cache.unbind(0)`

### 7.3 计算 skip_block_ids（必须实现）

输入：

- `mapper = self.paged_block_mappers[gid]`
- `dirty_blocks = self.paged_dirty_blocks[gid]`（由 `prepare_step` 缓存到 runner）
- `all_blocks = set(mapper.mapping.keys())`（本 step staged 的 logical blocks）
- `block_size = mapper.block_size`
- `prefix_blocks = io_prefix_blocks[layer_idx]`

对每个 logical_id in `all_blocks`，若满足以下所有条件，则加入 `skip_block_ids`：

1) `logical_id not in dirty_blocks`
2) 该 block 的最早 token position 满足 “后缀”：
   - `first_pos = int(cpu_block_positions[logical_id, 0])`
   - 要求 `first_pos >= 0` 且 `first_pos // block_size >= prefix_blocks`
3) hidden states 元数据完整可用：
   - `valid_len = int(cpu_block_valid_lens[logical_id])`
   - `valid_len > 0`
   - 对 `j in [0..valid_len)`：`cpu_block_positions[logical_id, j] >= 0`

若任一条件不满足，该 block 不能 skip，必须走 IO（保证正确性）。

### 7.4 PagedBlockMapper 的 IO load 必须支持 skip（必须实现）

修改 `vllm/v1/worker/gpu_model_runner.py::PagedBlockMapper.load_layer_async` 签名为：

```python
def load_layer_async(
    self,
    layer_name: str,
    layer_idx: int,
    skip_block_ids: set[int] | None = None,
) -> int:
    ...
```

实现要求：

- fallback per-block copy：`for logical_id, slot in self.mapping.items(): if logical_id in skip: continue`
- UVA batch copy（`self.use_batch_copy==True`）时：
  - 必须构建过滤后的 `(src_indices, dst_indices)`（logical_ids 与 slots），然后以 `filtered_num_blocks` 调用 batch copy kernel
  - 不能直接复用 `_assign_slots()` 预填的 indices（因为它包含全部 blocks，无法表达 skip）

### 7.5 recompute KV 的精确算法（必须实现）

在 `_runkv_pre_hook` 中（且在 `sync_load_layer` 之前）执行：

1) 构造 tokens 扁平化输入（CPU→GPU）：
   - 对每个 `logical_id in skip_block_ids`：
     - `valid_len = cpu_block_valid_lens[logical_id]`
     - `hs_cpu = cpu_attn_inputs_by_layer[layer_idx][logical_id, :valid_len, :]`
     - `pos_cpu = cpu_block_positions[logical_id, :valid_len]`
     - `slot_cpu = mapper.mapping[logical_id] * block_size + arange(valid_len)`
   - 按 block 拼接成：
     - `hs_cat_cpu: [T, hidden_size]`（CPU pinned）
     - `pos_cat_cpu: [T]`（CPU）
     - `slot_cat_cpu: [T]`（CPU）
2) H2D：
   - `hs_gpu = hs_cat_cpu.to(self.device, non_blocking=True)`
   - `pos_gpu = pos_cat_cpu.to(self.device, non_blocking=True)`
   - `slot_gpu = slot_cat_cpu.to(self.device, non_blocking=True)`
3) K/V 计算（Llama/OPT attention）：
   - `qkv, _ = llama_attn.qkv_proj(hs_gpu)`
   - `q, k, v = qkv.split([llama_attn.q_size, llama_attn.kv_size, llama_attn.kv_size], dim=-1)`
   - `q, k = llama_attn.rotary_emb(pos_gpu, q, k)`
   - 只用 `k, v`
   - reshape：
     - `k = k.view(-1, llama_attn.num_kv_heads, llama_attn.head_dim)`
     - `v = v.view(-1, llama_attn.num_kv_heads, llama_attn.head_dim)`
4) 写 staging：
   - 调用 `vllm._custom_ops.reshape_and_cache_flash(k, v, key_cache, value_cache, slot_gpu, kv_cache_dtype_str, k_scale, v_scale)`

### 7.6 flush 逻辑不变（V1）

- recompute blocks 不属于 dirty blocks，因此不会被 flush（正确，且避免无谓 PCIe 写回）。
- dirty blocks（本 step 新 KV）继续由现有 `_runkv_post_hook -> flush_layer_async -> sync_all_offloads` 处理。

---

## 8. 配置与 CLI（V1 必须能打开/关闭且可按层配置）

### 8.1 扩展 `RunKVOffloadConfig`

文件：`vllm/v1/core/kv_cache_offload_config.py::RunKVOffloadConfig`

新增字段：

- `enable_layer_recompute: bool = False`
- `layer_recompute_io_prefix_blocks: list[int] = field(default_factory=list)`
- `layer_recompute_measure_overhead: bool = False`（可选，但推荐实现）

### 8.2 CLI 参数（`vllm/engine/arg_utils.py`）

在 RunKV argument group 中新增：

- `--runkv-enable-layer-recompute`（bool）
- `--runkv-layer-recompute-io-prefix-blocks`（str）
  - 支持：
    - 单值：`"8"`（广播到所有层）
    - 列表：`"8,8,8,12,..."`（长度必须等于 num_layers）
- `--runkv-layer-recompute-measure-overhead`（bool，可选）

并在 `_build_kv_offload_config()` 把这些参数写入 `RunKVOffloadConfig`。

### 8.3 per-layer 参数的校验规则（必须实现）

在 `GPUModelRunner.initialize_kv_cache_tensors()`（已知 num_layers）做校验：

- 若 `enable_layer_recompute`：
  - `layer_recompute_io_prefix_blocks` 不能为空
  - 若长度为 1：广播到 `num_layers`
  - 否则长度必须等于 `num_layers`
  - 每个值必须 `>=0`

---

## 9. 测量与日志（可选但推荐）

当 `layer_recompute_measure_overhead=True` 时输出（按 layer/step 聚合均可）：

- `recompute_gpu_ms`：`qkv_proj + rope + reshape_and_cache_flash` 的 GPU event 时间
- `io_sync_wait_ms`：`mapper.sync_load_layer(layer_idx)` 的 wall time（越小代表 overlap 越好）
- `skip_blocks` / `skip_tokens`：本层 skip 的 blocks/tokens 数量

---

## 10. 测试（必须新增/修改）

### 10.1 单元测试：`PagedBlockMapper.load_layer_async` 支持 skip

文件：`tests/v1/kv_offload/test_runkv_offload.py`

新增用例（至少覆盖两条路径）：

1) fallback copy：
   - 构造 `mapper.mapping={10:0,20:1}`
   - `skip_block_ids={20}`
   - `load_layer_async(..., skip_block_ids=skip_block_ids)` 后：
     - slot0 被拷贝，slot1 不拷贝（保持初始值）

2) batch_copy path：
   - 需要在测试里让 `mapper.use_batch_copy=True` 且提供 `mapper._batch_copy_fn`（可用一个简单的 python/torch 实现替代，或 mock）
   - 同样验证 skip 的 slot 不被写

### 10.2（可选）单元测试：skip_block_ids 的计算正确

新增 `tests/v1/kv_offload/test_layer_recompute_skip.py`（或同目录其他文件）：

- 给定：
  - `cpu_block_positions/cpu_block_valid_lens`
  - `dirty_blocks`
  - `mapper.mapping`
  - `io_prefix_blocks[layer_idx]`
- 断言：
  - dirty blocks 不会被 skip
  - prefix blocks 不会被 skip
  - 元数据不完整的 blocks 不会被 skip

---

## 11. 实现步骤清单（按顺序执行）

1) 扩展 `RunKVOffloadConfig` + CLI 参数（`vllm/v1/core/kv_cache_offload_config.py`、`vllm/engine/arg_utils.py`）
2) 新增 `vllm/v1/worker/layer_recompute.py`：
   - `LayerRecomputeManager`（封装 CPU store、hook 写入、recompute 执行、D2H sync）
3) 修改 `GPUModelRunner.initialize_kv_cache_tensors()`：
   - 初始化 manager、分配 CPU store、注册 layernorm hooks
4) 修改 `GPUModelRunner._prepare_inputs()`：
   - 缓存 scheduled tokens 的 `logical_ids/offsets/positions`
   - 维护 `logical_id_owner_req_id` 并 reset 元数据
5) 修改 `PagedBlockMapper.load_layer_async()`：
   - 增加 `skip_block_ids`，支持 batch_copy 与 fallback
6) 修改 `GPUModelRunner._runkv_pre_hook()`：
   - 新时序：prefetch_current(KV+HS) → sync_current_load → prefetch_next(KV+HS) → recompute_current
7) 修改 `GPUModelRunner.execute_model()`：
   - 在 step 末尾增加 `LayerRecomputeManager.sync_hs_d2h()`
8) 跑测试：
   - `pytest tests/v1/kv_offload/test_runkv_offload.py -k skip -v`

---

## 12. 实施进度（逐步更新）

### Step 1（已完成）：扩展 `RunKVOffloadConfig` + CLI 参数

- 完成时间：2026-03-02
- 已修改：
  - `vllm/v1/core/kv_cache_offload_config.py`
    - 新增字段：
      - `enable_layer_recompute`
      - `layer_recompute_io_prefix_blocks`
      - `layer_recompute_measure_overhead`
  - `vllm/engine/arg_utils.py`
    - `EngineArgs` 新增 RunKV CLI 对应字段，确保 `from_cli_args()` 不丢失 RunKV 参数
    - 新增 CLI 参数：
      - `--runkv-enable-layer-recompute`
      - `--runkv-layer-recompute-io-prefix-blocks`
      - `--runkv-layer-recompute-measure-overhead`
    - `_build_kv_offload_config()` 已写入上述参数到 `RunKVOffloadConfig`
    - 新增 `layer_recompute_io_prefix_blocks` 字符串解析逻辑（支持单值 / 逗号列表）
- 已新增测试：
  - `tests/engine/test_arg_utils.py`
    - `test_runkv_layer_recompute_prefix_blocks_parser`
    - `test_runkv_layer_recompute_cli_to_config`
    - `test_runkv_layer_recompute_prefix_blocks_parser_rejects_empty_segment`
    - `test_runkv_fields_are_preserved_by_from_cli_args`

### Step 2（已完成）：新增 `LayerRecomputeManager`

- 完成时间：2026-03-02
- 已新增：
  - `vllm/v1/worker/layer_recompute.py`
    - 新增 `LayerRecomputeManager`，包含以下最小 API：
      - `register_layernorm_hooks()`
      - `begin_step()`
      - `compute_skip_block_ids_for_layer()`
      - `recompute_kv_for_layer()`
      - `sync_hs_d2h()`
    - 已实现 CPU store 分配与 owner/reset 逻辑：
      - `cpu_attn_inputs_by_layer`
      - `cpu_block_positions`
      - `cpu_block_valid_lens`
      - `logical_id_owner_req_id`
    - 已实现 layernorm hook 的 hidden states 捕获、D2H pending 写入与 step-end 同步落盘
    - 已实现 recompute 路径（Llama：`qkv_proj + rotary_emb + reshape_and_cache_flash`；OPT：`qkv_proj + reshape_and_cache_flash`）
    - 2026-03-03 策略更新（按最新设计）：
      - `cpu_attn_inputs_by_layer` 改为按 `[num_layers, num_cpu_blocks, block_size, hidden_size]` 全量预分配
      - hook 保存所有 scheduled tokens（prefix + suffix），不再只保存 suffix
      - 目标：保证 KV/HS block 数与覆盖范围严格一致，支持后续动态 recompute 配置
      - 新增初始化日志：输出 HS store 的 `num_blocks`、`hs_bytes_per_block`、`total_size`、`meta_size`
- 已新增测试：
  - `tests/v1/kv_offload/test_layer_recompute_manager.py`
    - `test_begin_step_resets_owner_changed_blocks`
    - `test_layernorm_hook_and_sync_store_all_tokens`
    - `test_compute_skip_block_ids_is_suffix_only_by_block_index`
- 测试结果：
  - `pytest -q tests/v1/kv_offload/test_layer_recompute_manager.py`
  - `3 passed`

### Step 3（已完成）：在 `initialize_kv_cache_tensors()` 初始化 manager

- 完成时间：2026-03-02
- 已修改：
  - `vllm/v1/worker/gpu_model_runner.py`
    - 新增字段：
      - `layer_recompute_manager`
      - `layer_recompute_enabled`
    - 新增 `_normalize_layer_recompute_io_prefix_blocks(num_layers)`：
      - 校验 `layer_recompute_io_prefix_blocks` 不能为空
      - 长度为 1 时自动广播到 `num_layers`
      - 否则必须严格等于 `num_layers`
      - 所有值必须 `>=0`
    - 新增 `_maybe_init_layer_recompute_manager(...)`：
    - 在 recompute 启用时创建并注册 `LayerRecomputeManager`
    - 在 recompute 关闭时移除 hooks 并清理 manager
    - 额外约束：
        - 仅支持单 KV cache group
        - 仅支持 `model_type in {\"llama\", \"opt\"}`
    - 在 `initialize_kv_cache_tensors()` 的 RunKV 路径接入上述初始化逻辑
- 已新增测试：
  - `tests/v1/kv_offload/test_layer_recompute_manager.py`
    - `test_normalize_io_prefix_blocks_broadcasts_single_value`
    - `test_normalize_io_prefix_blocks_rejects_length_mismatch`
    - `test_normalize_io_prefix_blocks_rejects_negative_values`
- 测试结果：
  - `pytest -q tests/v1/kv_offload/test_layer_recompute_manager.py`
  - `6 passed`

### Step 4（已完成）：在 `_prepare_inputs()` 缓存 scheduled token 元数据并执行 owner/reset

- 完成时间：2026-03-02
- 已修改：
  - `vllm/v1/worker/gpu_model_runner.py`
    - 新增 step 缓存字段：
      - `_lr_req_indices_np`
      - `_lr_positions_np`
      - `_lr_logical_ids_np`
      - `_lr_block_offsets_np`
      - `_lr_block_size`
      - `_lr_num_reqs`
    - 新增 `_prepare_layer_recompute_step_metadata(...)`：
      - 在 recompute 启用时，向量化计算 `block_indices/block_offsets/logical_ids`
      - 缓存上述 `_lr_*` 字段供 hooks 使用
      - 调用 `layer_recompute_manager.begin_step(...)` 执行 owner/reset 元数据维护
      - 当前强约束仅支持单 block table group（V1）
      - 在 recompute 关闭时自动清空 `_lr_*` 缓存
    - 在 `_prepare_inputs()` 中接入 `_prepare_layer_recompute_step_metadata(...)` 调用
- 已新增测试：
  - `tests/v1/kv_offload/test_layer_recompute_manager.py`
    - `test_prepare_layer_recompute_step_metadata_caches_arrays_and_calls_manager`
    - `test_prepare_layer_recompute_step_metadata_clears_cache_when_disabled`
- 测试结果：
  - `pytest -q tests/v1/kv_offload/test_layer_recompute_manager.py`
  - `8 passed`

### Step 5（已完成）：`PagedBlockMapper.load_layer_async()` 支持 skip

- 完成时间：2026-03-02
- 已修改：
  - `vllm/v1/worker/gpu_model_runner.py`
    - `PagedBlockMapper.load_layer_async(...)` 增加参数：
      - `skip_block_ids: set[int] | None = None`
    - `PagedBlockMapper.load_layer(...)` 同步支持 `skip_block_ids`
    - fallback copy 路径：
      - 遍历 `mapping` 时跳过 `skip_block_ids`
    - batch copy 路径：
      - 无 skip 时继续走预填索引快路径
      - 有 skip 时构建过滤后的 `(src_indices, dst_indices)` 并调用 batch copy kernel
- 已新增测试：
  - `tests/v1/kv_offload/test_runkv_offload.py`
    - `test_load_layer_async_skip_blocks_fallback_copy`
    - `test_load_layer_async_skip_blocks_batch_copy`
- 本地测试结果：
  - `pytest -q tests/v1/kv_offload/test_runkv_offload.py -k skip_blocks`
  - 当前环境无 CUDA，新增 2 个用例被 `skipif` 跳过（`2 skipped`）

### Step 6（已完成）：修改 `_runkv_pre_hook()` 新时序并接入 recompute

- 完成时间：2026-03-03
- 已修改：
  - `vllm/v1/worker/gpu_model_runner.py`
    - `_runkv_pre_hook()` 已接入：
      - 当前层 `skip_block_ids` 计算
      - 当前层 `load_layer_async(..., skip_block_ids=...)`（layer0 首次）
      - 当前层 `prefetch_recompute_inputs_for_layer(...)`（HS/pos/slot 预取）
      - `sync_load_layer(layer_idx)` 作为当前层 IO prefix 就绪栅栏
      - 下一层 `skip_next` 计算 + `load_layer_async(..., skip_block_ids=skip_next)` + `prefetch_recompute_inputs_for_layer(...)`
      - 当前层 `recompute_kv_for_layer(...)`（消费预取输入）
  - `vllm/v1/worker/layer_recompute.py`
    - 新增 `prefetch_recompute_inputs_for_layer(...)`
    - `recompute_kv_for_layer(...)` 改为优先消费预取缓存，无缓存时兜底构建
- 已新增测试：
  - `tests/v1/kv_offload/test_layer_recompute_manager.py`
    - `test_runkv_pre_hook_layer_recompute_pipeline_order_and_skip_forwarding`
    - `test_runkv_pre_hook_without_layer_recompute_keeps_original_behavior`
- 测试结果：
  - `pytest -q tests/v1/kv_offload/test_layer_recompute_manager.py -k 'runkv_pre_hook or prepare_layer_recompute_step_metadata or compute_skip_block_ids or normalize_io_prefix'`
  - `8 passed`

### Step 7（已完成）：在 step 末尾同步 layer recompute D2H

- 完成时间：2026-03-03
- 已修改：
  - `vllm/v1/worker/gpu_model_runner.py`
    - 新增 `_sync_runkv_step_end_state()`：
      - 统一执行 `self._sync_all_runkv_offloads()`
      - 当 `layer_recompute_enabled` 时执行 `self.layer_recompute_manager.sync_hs_d2h()`
    - `execute_model()` 与同类 forward 路径中的 step-end 同步调用点统一改为 `_sync_runkv_step_end_state()`
- 已新增测试：
  - `tests/v1/kv_offload/test_layer_recompute_manager.py`
    - `test_sync_runkv_step_end_state_syncs_offload_and_hs_when_enabled`
    - `test_sync_runkv_step_end_state_skips_hs_sync_when_recompute_disabled`
    - `test_sync_runkv_step_end_state_is_noop_when_runkv_disabled`
- 测试结果：
  - `pytest -q tests/v1/kv_offload/test_layer_recompute_manager.py`
  - `13 passed`

### Step 8（部分完成）：端到端正确性测试（vanilla vs RunKV vs recompute）

- 完成时间：2026-03-03
- 已修改：
  - 新增 `tests/v1/kv_offload/test_layer_recompute_e2e.py`
    - 新增端到端三方正确性对比：
      - `test_layer_recompute_single_request_matches_vanilla_and_runkv_baseline`
      - `test_layer_recompute_concurrent_requests_match_vanilla_and_runkv_baseline`
    - 两组测试都比较：
      - A: `vanilla`（`RunKV(enabled=False)`）
      - B: `RunKV(enabled=True, enable_layer_recompute=False)`
      - C: `RunKV(enabled=True, enable_layer_recompute=True, layer_recompute_io_prefix_blocks=[4])`
      - 断言同一请求集最终 `token_ids`：`A==B` 且 `A==C`
    - 增加环境自适应：
      - 无 CUDA 自动 skip
      - 无本地模型时自动 skip（可通过 `VLLM_RUNKV_E2E_MODEL` 指定模型目录）
    - 增加内存安全保护：
      - 每次 `_run_case()` 构造 engine 前记录 host available memory
      - `shutdown` 后显式释放 `engine` 引用并触发 `gc/empty_cache`
      - 等待 available memory 回升到接近基线，再构造下一 engine
      - 避免连续构造三组引擎时 pinned memory 尚未回收导致 OOM
    - 调整 e2e 负载以降低环境敏感性：
      - `max_model_len` 提升到 `2048`
      - 缩短测试 prompts，避免超过模型上下文上限
- 本地测试结果：
  - `pytest -q tests/v1/kv_offload/test_layer_recompute_e2e.py`
  - 当前环境无 CUDA，`2 skipped`
- 待完成（Step 8 剩余）：
  - 增加性能观测脚本与指标输出（`io_sync_wait_ms / recompute_ms / step_latency / tokens_per_s`）

### Step 9（已完成）：`cpu_memory_fraction` 对 `KV + HS` 总预算生效

- 完成时间：2026-03-03
- 背景与问题：
  - 之前 `cpu_memory_fraction/cpu_memory_limit` 只约束 KV backing store，
    layer recompute 的 CPU hidden states 存储不在预算内，容易出现 host 内存持续上涨。
- 已修改：
  - `vllm/v1/core/kv_cache_utils.py`
    - 在 `get_kv_cache_config_from_groups(...)` 中新增 recompute 预算折算逻辑：
      - 当 `kv_offload_config.enabled=True` 且 `enable_layer_recompute=True` 时，
        `available_memory` 视为 `KV + hidden states` 的总预算。
      - 先计算每个 logical block 的总开销：
        - `kv_bytes_per_block_total`
        - `hs_bytes_per_block_total = num_layers * block_size * hidden_size * dtype_size`
      - 直接按每 block 总成本计算 block 数：
        - `num_blocks = floor(total_budget / (kv_bytes_per_block_total + hs_bytes_per_block_total))`
      - 保证 KV blocks 与 HS blocks 一一对应（能存 KV 的 token 必须也能存其 HS）。
  - `vllm/v1/core/kv_cache_offload_config.py`
    - `cpu_memory_fraction` 注释更新为“在 layer recompute 下由 KV 与 HS 共享”。
  - 已新增测试：
  - `tests/v1/kv_offload/test_runkv_offload.py`
    - `test_layer_recompute_budget_includes_hidden_state_store`
    - 验证同一 `available_memory` 下：
      - recompute 关闭时 `num_blocks` 更大（仅 KV）
      - recompute 开启时 `num_blocks` 按 `KV+HS` 联合预算收缩

### Step 10（已完成）：补齐 RunKV + recompute 的 NVTX 观测点

- 完成时间：2026-03-04
- 已修改：
  - `vllm/v1/worker/layer_recompute.py`
    - 为以下关键阶段新增 NVTX scope（通过 `record_function_or_nullcontext`）：
      - `runkv_recompute:prefetch_inputs:L{layer_idx}`
      - `runkv_recompute:prefetch_pack:L{layer_idx}`
      - `runkv_recompute:hs_h2d_copy:L{layer_idx}`
      - `runkv_recompute:recompute_kv:L{layer_idx}`
      - `runkv_recompute:project_kv:L{layer_idx}`
      - `runkv_recompute:cache_kv:L{layer_idx}`
      - `runkv_recompute:capture_hs:L{layer_idx}`
      - `runkv_recompute:hs_d2h_copy:L{layer_idx}`
      - `runkv_recompute:sync_hs_d2h`
  - `vllm/v1/worker/gpu_model_runner.py`
    - 新增 step/调度层 NVTX scope：
      - `runkv_recompute:prepare_step_metadata`
      - `runkv_recompute:pre_hook:{layer_name}:L{layer_idx}`
      - `runkv_recompute:post_hook:{layer_name}:L{layer_idx}`
      - `runkv_recompute:step_end_sync`
- 测试结果：
  - `python -m py_compile vllm/v1/worker/layer_recompute.py vllm/v1/worker/gpu_model_runner.py`
  - `./.venv/bin/pytest -q tests/v1/kv_offload/test_layer_recompute_manager.py -k "prepare_layer_recompute_step_metadata or runkv_pre_hook or sync_runkv_step_end_state"`
  - `7 passed`
