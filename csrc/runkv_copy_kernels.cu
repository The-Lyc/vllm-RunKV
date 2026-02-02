// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/*
 * RunKV Batch Copy Kernels
 *
 * Uses UVA (Unified Virtual Addressing) to enable single-kernel-launch
 * batch copy of KV cache blocks between CPU (pinned) and GPU memory.
 *
 * This eliminates Python for-loop overhead which was causing memcpy
 * dispatch to block compute dispatch.
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

// Kernel for batch copy using UVA
// Works for both H2D and D2H directions since UVA allows GPU to access pinned
// CPU memory
//
// For tensor shape [d0, d1, ..., blocks_dim, ..., dn]:
// - outer_size = product of d0, d1, ..., d_{blocks_dim-1}
// - inner_size = product of d_{blocks_dim+1}, ..., dn = stride[blocks_dim]
//
// We iterate over outer_idx in [0, outer_size) and copy inner_size elements
// each time.
template <typename T>
__global__ void batch_copy_blocks_kernel(
    T* __restrict__ dst, const T* __restrict__ src,
    const int64_t* __restrict__ dst_indices,
    const int64_t* __restrict__ src_indices, const int64_t num_copies,
    const int64_t outer_size, const int64_t inner_size,
    const int64_t dst_block_stride, const int64_t src_block_stride,
    const int64_t num_dst_blocks, const int64_t num_src_blocks) {
  const int64_t copy_idx = blockIdx.x;
  if (copy_idx >= num_copies) return;

  const int64_t dst_block = dst_indices[copy_idx];
  const int64_t src_block = src_indices[copy_idx];

  // Bounds check: skip if indices are out of range
  if (dst_block < 0 || dst_block >= num_dst_blocks || src_block < 0 ||
      src_block >= num_src_blocks) {
    // Invalid index - skip this copy to avoid illegal memory access
    // This shouldn't happen if caller provides valid indices
    return;
  }

  // Total elements to copy for this block
  const int64_t total_elements = outer_size * inner_size;

  // Each CUDA thread handles multiple elements
  for (int64_t i = threadIdx.x; i < total_elements; i += blockDim.x) {
    // Decompose i into (outer_idx, inner_idx)
    const int64_t outer_idx = i / inner_size;
    const int64_t inner_idx = i % inner_size;

    // Calculate linear offsets
    // For shape [d0, d1, blocks_dim, d3, d4], blocks_dim=2:
    // offset = outer_idx * (num_blocks * inner_size) + block * inner_size +
    // inner_idx But we need to be careful about the stride calculation
    //
    // Actually for contiguous:
    // offset = outer_idx * (shape[blocks_dim] * stride[blocks_dim]) + block *
    // stride[blocks_dim] + inner_idx
    const int64_t dst_offset = outer_idx * (num_dst_blocks * dst_block_stride) +
                               dst_block * dst_block_stride + inner_idx;
    const int64_t src_offset = outer_idx * (num_src_blocks * src_block_stride) +
                               src_block * src_block_stride + inner_idx;

    dst[dst_offset] = src[src_offset];
  }
}

}  // namespace

void runkv_batch_copy_blocks(torch::Tensor dst, torch::Tensor src,
                             torch::Tensor dst_indices,
                             torch::Tensor src_indices, int64_t blocks_dim) {
  // Get the current CUDA stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t num_copies = dst_indices.size(0);
  if (num_copies == 0) return;

  // For a tensor with shape [d0, d1, ..., blocks_dim, ..., dn]
  // We want to copy ALL elements in the slice tensor[:, :, ..., block_idx, :,
  // ...]
  //
  // The stride at blocks_dim gives us the step between consecutive blocks.
  // But we need to copy:
  //   - For dims BEFORE blocks_dim: need to iterate over all values
  //   - For dims AFTER blocks_dim: covered by the stride
  //
  // Total elements per block = product of all dims except blocks_dim
  // = (stride[blocks_dim] * shape[0...blocks_dim-1])
  //
  // Actually, for contiguous layout:
  // stride[blocks_dim] = product of shape[blocks_dim+1 ... n]
  // So total elements per block = stride[blocks_dim] * product of shape[0 ...
  // blocks_dim-1]
  //
  // But it's simpler: total elements / shape[blocks_dim]

  int64_t total_elements = dst.numel() / dst.size(blocks_dim);
  int64_t block_stride = dst.stride(blocks_dim);
  int64_t outer_size = 1;
  for (int64_t i = 0; i < blocks_dim; i++) {
    outer_size *= dst.size(i);
  }
  int64_t inner_size = block_stride;  // elements per block per outer iteration

  // Launch kernel
  const int threads = 256;
  const int blocks = num_copies;

  // Ensure indices are on the right device (can be CPU pinned or GPU)
  // UVA allows GPU to read from pinned CPU memory
  const int64_t* dst_idx_ptr = dst_indices.data_ptr<int64_t>();
  const int64_t* src_idx_ptr = src_indices.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, dst.scalar_type(),
      "batch_copy_blocks", [&] {
        // Use generic kernel that handles outer loop
        batch_copy_blocks_kernel<<<blocks, threads, 0, stream>>>(
            dst.data_ptr<scalar_t>(), src.data_ptr<scalar_t>(), dst_idx_ptr,
            src_idx_ptr, num_copies, outer_size, inner_size,
            dst.stride(blocks_dim), src.stride(blocks_dim),
            dst.size(blocks_dim),  // num_dst_blocks (for bounds checking)
            src.size(blocks_dim)   // num_src_blocks (for bounds checking)
        );
      });
}

// Standalone module for use without rebuilding vLLM
// Compile with: python setup_runkv.py build_ext --inplace
#ifdef RUNKV_STANDALONE_MODULE
  #include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batch_copy_blocks", &runkv_batch_copy_blocks,
        "Batch copy KV cache blocks using UVA", py::arg("dst"), py::arg("src"),
        py::arg("dst_indices"), py::arg("src_indices"),
        py::arg("blocks_dim") = 0);
}
#endif
