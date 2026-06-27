#pragma once

#include "common.cuh"

// GCN (gfx906) weight-repacking buffer type.
//
// Q4_K weights are transformed at upload into a three-plane layout that
// the repacked matvec kernel streams fully coalesced — the on-disk
// superblock layout only sustains ~58% of HBM bandwidth on wave64
// because nibbles, scales and superblock constants interleave every 144
// bytes. Repacked, the same matvec sustains ~89% (measured 470 -> 740
// GB/s on a 21504x5376 FFN shape, gfx906).
//
// Layout (per 2D tensor [ne0 = K, ne1 = M], row-major rows of K):
//   nibble plane:     M * nsp * 16 bytes — one 16-byte chunk per
//                     32-weight sub-block; uint32 j of a chunk holds
//                     weights 4j..4j+3 in its low nibbles and weights
//                     16+4j..16+4j+3 in its high nibbles (dp4a-ready).
//   scale plane:      M * nsp * 2 bytes — per sub-block the 6-bit
//                     scale then min as two u8 (get_scale_min_k4 output).
//   superblock plane: M * (K/256) * 4 bytes — fp16 d then dmin.
// nsp = K/32, plus one padding sub-block when K/32 is a power of two
// (a power-of-two row stride aliases every row onto the same HBM
// channel: ~3x matvec slowdown).
//
// Enabled with GGML_CUDA_REPACK=1 on GCN devices only. Weights placed
// in this buffer type cannot be read back (get_tensor aborts).

bool ggml_backend_buft_is_cuda_repack(ggml_backend_buffer_type_t buft);

// Returns the repack buffer type for `device`, or nullptr when disabled
// (env GGML_CUDA_REPACK unset/0) or the device is not GCN.
ggml_backend_buffer_type_t ggml_backend_cuda_repack_buffer_type(int device);

// True if `t` can live in the repack buffer type: 2D Q4_K with
// ne0 % 256 == 0.
bool ggml_cuda_repack_tensor_supported(const ggml_tensor * t);

// MUL_MAT with src0 in the repack buffer type. ne11 == 1 runs the
// repacked dp4a matvec; larger ne11 runs the repacked int8 MMQ GEMM.
// 3D/4D src1 broadcasts the 2D weight per slice.
void ggml_cuda_mul_mat_repacked(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// MUL_MAT_ID with 3D src0 (per-expert repacked slabs) in the repack
// buffer type. Decode (one token) runs one matvec per assignment;
// batches run a grouped tile GEMM over expert-sorted assignments.
void ggml_cuda_mul_mat_id_repacked(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
    ggml_tensor * dst);

// Fused gate+up GLU decode path (Q4_K, dense and MoE).
bool ggml_cuda_repack_should_fuse_glu(const ggml_tensor * up, const ggml_tensor * gate,
    const ggml_tensor * glu);
void ggml_cuda_mul_mat_repacked_fused_glu(ggml_backend_cuda_context & ctx,
    const ggml_tensor * up_w, const ggml_tensor * gate_w,
    const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, int glu_op);
