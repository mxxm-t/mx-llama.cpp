#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
    const int32_t * snap_rows       = nullptr; // ring: per-(slot, seq) cache row, data = cache base
    int64_t         snap_row_stride = 0;       // ring: cache row stride in floats
};

// input state read in place from cache rows instead of a gathered copy (elided GET_ROWS)
struct ggml_cuda_gdn_state_src {
    const float *   base;       // GET_ROWS source rows
    const int32_t * rows;       // per-sequence row index
    int64_t         row_stride; // in floats
};
void ggml_cuda_gdn_put_state_src(const ggml_tensor * gdn, const ggml_cuda_gdn_state_src & src);
bool ggml_cuda_gdn_take_state_src(const ggml_tensor * gdn, ggml_cuda_gdn_state_src & src);

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);
