// Public API for the Q8_0 repacked-weight matmul path .
#pragma once

#include "../common.cuh"

bool ggml_backend_buft_is_cuda_repack(ggml_backend_buffer_type_t buft);

ggml_backend_buffer_type_t ggml_backend_cuda_repack_buffer_type(int device);

bool ggml_cuda_repack_tensor_supported(const ggml_tensor * t);

bool ggml_cuda_repack_mul_mat_should_fire(const ggml_tensor * src0);

// Async-upload path: canonical chunks stage into per-device scratch and the
// device-side repack kernel runs when the tensor completes. The scratch is
// released on the first graph compute after a load.
void ggml_cuda_repack_set_tensor_async(int device, cudaStream_t stream,
    ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_cuda_repack_async_release(int device);

// Fused MMV is Q8_0-only (it uses the mat-vec kernel, which has no MXFP4 port).
bool ggml_cuda_repack_mmv_fusion_supported(const ggml_tensor * src0);

void ggml_cuda_mul_mat_repacked(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_cuda_mul_mat_id_repacked(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
    ggml_tensor * dst);

// True when a batch of n_tokens is narrow enough that the repacked path runs
// the multi-column mat-vec, which is the only shape the fused FFN entry points
// below can serve. Wider batches fall through to the tiled GEMM.
bool ggml_cuda_repack_mmv_fusion_width_ok(int64_t n_tokens, bool has_ids, ggml_type wt);

void ggml_cuda_mul_mat_vec_repacked_fused(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion);

void ggml_cuda_mul_mat_id_vec_repacked_fused(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
    ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion);
