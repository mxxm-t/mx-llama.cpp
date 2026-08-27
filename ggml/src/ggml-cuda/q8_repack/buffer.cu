// Repack buffer type: on upload, supported weights (Q8_0, MXFP4) are converted to the
// gfx906 two-plane layout; everything else is copied through unchanged. Enabled
// only on gfx906 (the arch the kernels compile for); which tensors qualify is
// decided by ggml_cuda_repack_tensor_supported().
#include "repack.cuh"
#include "repack-common.cuh"
#include "ggml-cuda.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

struct ggml_backend_cuda_repack_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_repack_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buft->context;
    return ctx->name.c_str();
}

bool ggml_backend_buft_is_cuda_repack(ggml_backend_buffer_type_t buft) {
    bool is_repack = buft->iface.get_name == ggml_backend_cuda_repack_buffer_type_get_name;
    return is_repack;
}

// Pinned upload slots: two per device, alternating, event-guarded.
struct upl_slot {
    uint8_t *   p       = nullptr;
    size_t      cap     = 0;
    cudaEvent_t ev      = nullptr;
    bool        ev_live = false;
};
static upl_slot     s_upl[GGML_CUDA_MAX_DEVICES][2];
static int          s_upl_turn[GGML_CUDA_MAX_DEVICES] = {};
static cudaStream_t s_upl_stream[GGML_CUDA_MAX_DEVICES] = {};
static std::mutex   s_upl_mutex;

// Repack one tensor into the GCN layout (per-expert stride repack_gcn_nbytes) and
// upload it; the source may be the host upload buffer or an accumulated staging copy.
static void repack_and_upload(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
        const uint8_t * src) {
    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buffer->buft->context;
    ggml_cuda_set_device(ctx->device);

    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t ne2 = tensor->ne[2];

    const size_t src_stride = ggml_nbytes(tensor) / ne2;
    const size_t dst_stride = repack_gcn_nbytes(tensor->type, ne0, ne1);
    const size_t need       = dst_stride * ne2;

    // Pinned double-buffered upload scratch per device: repack into one slot,
    // launch the H2D asynchronously on a dedicated stream, and only wait when
    // the slot comes around again. Hides the upload behind the next tensor's
    // read and splice; a pageable synchronous H2D here measured ~22 s of the
    // -sm tensor MoE load. Completion before compute is enforced by
    // ggml_cuda_repack_async_release() at the first graph compute.
    std::lock_guard<std::mutex> upl_lock(s_upl_mutex);
    const int dev = ctx->device;
    if (s_upl_stream[dev] == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&s_upl_stream[dev], cudaStreamNonBlocking));
    }
    upl_slot & u = s_upl[dev][s_upl_turn[dev]];
    s_upl_turn[dev] ^= 1;
    if (u.ev_live) {
        CUDA_CHECK(cudaEventSynchronize(u.ev));
        u.ev_live = false;
    }
    if (u.cap < need) {
        if (u.p != nullptr) {
            CUDA_CHECK(cudaFreeHost(u.p));
        }
        CUDA_CHECK(cudaMallocHost((void **) &u.p, need));
        u.cap = need;
    }
    if (u.ev == nullptr) {
        CUDA_CHECK(cudaEventCreateWithFlags(&u.ev, cudaEventDisableTiming));
    }
    uint8_t * repacked = u.p;

    for (int64_t e = 0; e < ne2; e++) {
        repack_host(tensor->type, src + e * src_stride,
            repacked + e * dst_stride, ne0, ne1);
    }

    CUDA_CHECK(cudaMemcpyAsync(tensor->data, repacked, need,
        cudaMemcpyHostToDevice, s_upl_stream[dev]));
    CUDA_CHECK(cudaEventRecord(u.ev, s_upl_stream[dev]));
    u.ev_live = true;
}

// ---- async upload path ------------------------------------------------------
// The loader's async path delivers CANONICAL bytes in file-order chunks whose
// boundaries ignore the 34-byte block structure, so chunks stage into a
// per-device scratch at their canonical offsets and the repack kernel runs on
// the SAME stream once the tensor is complete. Stream ordering makes one
// scratch per device safe: the next tensor's H2D writes are queued after this
// tensor's repack kernel has read the bytes.

// Device-side repack: one thread per destination sub-block. Reads canonical
// block_q8_0 (34 B: half d, then 32 int8) and writes the two planes. Padding
// sub-blocks (blk >= n_blocks) are zeroed, matching repack_q8_0_host().
static __global__ void repack_q8_0_kernel(
        const uint8_t * __restrict__ src, uint8_t * __restrict__ dst,
        const int64_t ne1, const int64_t n_blocks, const int64_t qs_str,
        const int64_t qs_len, const int64_t src_stride, const int64_t dst_stride,
        const int64_t total) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t blk = i % n_blocks;
        const int64_t row = (i / n_blocks) % ne1;
        const int64_t e   = i / (n_blocks * ne1);

        uint8_t * d_qs = dst + e * dst_stride + row * qs_str + blk * 32;
        uint8_t * d_d  = dst + e * dst_stride + qs_len + (row * n_blocks + blk) * 2;

        const uint8_t * sb = src + e * src_stride + (row * n_blocks + blk) * 34;
#pragma unroll
        for (int k = 0; k < 32; k++) {
            d_qs[k] = sb[2 + k];
        }
        d_d[0] = sb[0];
        d_d[1] = sb[1];
    }
}

// Device-side MXFP4 repack: canonical block_mxfp4 (17 B: e8m0 byte, 16 nibble
// bytes) to de-aliased nibble rows + a 1-byte e plane. The de-alias gap bytes
// are never read (the kernels guard with sb < n_sub), matching the Q8_0 kernel.
static __global__ void repack_mxfp4_kernel(
        const uint8_t * __restrict__ src, uint8_t * __restrict__ dst,
        const int64_t ne1, const int64_t n_blocks, const int64_t qs_str,
        const int64_t qs_len, const int64_t src_stride, const int64_t dst_stride,
        const int64_t total) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t blk = i % n_blocks;
        const int64_t row = (i / n_blocks) % ne1;
        const int64_t e   = i / (n_blocks * ne1);

        uint8_t * d_qs = dst + e * dst_stride + row * qs_str + blk * 16;
        uint8_t * d_e  = dst + e * dst_stride + qs_len + row * n_blocks + blk;

        const uint8_t * sb = src + e * src_stride + (row * n_blocks + blk) * 17;
#pragma unroll
        for (int k = 0; k < 16; k++) {
            d_qs[k] = sb[1 + k];
        }
        d_e[0] = sb[0];
    }
}

struct repack_async_state {
    uint8_t *           scratch  = nullptr;
    size_t              cap      = 0;
    const ggml_tensor * cur      = nullptr;
    size_t              received = 0;
};
static repack_async_state s_async[GGML_CUDA_MAX_DEVICES];
static std::mutex s_async_mutex;

void ggml_cuda_repack_set_tensor_async(int device, cudaStream_t stream,
        ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    std::lock_guard<std::mutex> lock(s_async_mutex);
    repack_async_state & st = s_async[device];
    const size_t total = ggml_nbytes(tensor);

    ggml_cuda_set_device(device);
    if (st.cap < total) {
        // cudaFree synchronizes, but growth is monotonic across a load
        if (st.scratch != nullptr) {
            CUDA_CHECK(cudaFree(st.scratch));
        }
        CUDA_CHECK(cudaMalloc(&st.scratch, total));
        st.cap = total;
    }
    if (st.cur != tensor) {
        // the loader finishes one tensor before starting the next - a switch
        // with bytes outstanding means interleaved chunks and silent corruption
        GGML_ASSERT(st.cur == nullptr && "repack async upload: previous tensor incomplete");
        st.cur      = tensor;
        st.received = 0;
    }
    GGML_ASSERT(offset + size <= total);
    CUDA_CHECK(cudaMemcpyAsync(st.scratch + offset, data, size, cudaMemcpyHostToDevice, stream));
    st.received += size;

    if (st.received == total) {
        const int64_t ne0      = tensor->ne[0];
        const int64_t ne1      = tensor->ne[1];
        const int64_t ne2      = tensor->ne[2];
        const int64_t n_blocks = ne0 / 32;
        const int64_t qs_str   = repack_qs_row_stride(tensor->type, ne0);
        const size_t  src_str  = total / ne2;
        const size_t  dst_str  = repack_gcn_nbytes(tensor->type, ne0, ne1);
        const int64_t n_out    = ne2 * ne1 * n_blocks;
        const int     block    = 256;
        const int     grid     = (int) std::min<int64_t>((n_out + block - 1) / block, 65535);
        switch (tensor->type) {
            case GGML_TYPE_Q8_0:
                repack_q8_0_kernel<<<grid, block, 0, stream>>>(
                    st.scratch, (uint8_t *) tensor->data, ne1, n_blocks, qs_str,
                    ne1 * qs_str, (int64_t) src_str, (int64_t) dst_str, n_out);
                break;
            case GGML_TYPE_MXFP4:
                repack_mxfp4_kernel<<<grid, block, 0, stream>>>(
                    st.scratch, (uint8_t *) tensor->data, ne1, n_blocks, qs_str,
                    ne1 * qs_str, (int64_t) src_str, (int64_t) dst_str, n_out);
                break;
            default:
                GGML_ABORT("unsupported repack type for async upload");
        }
        CUDA_CHECK(cudaGetLastError());
        st.cur      = nullptr;
        st.received = 0;
    }
}

void ggml_cuda_repack_async_release(int device) {
    // Drain any in-flight pinned uploads from the sync/meta path first - this
    // runs at the first graph compute, so nothing may still be uploading.
    {
        std::lock_guard<std::mutex> lock(s_upl_mutex);
        for (int k = 0; k < 2; k++) {
            upl_slot & u = s_upl[device][k];
            if (u.ev_live) {
                CUDA_CHECK(cudaEventSynchronize(u.ev));
                u.ev_live = false;
            }
        }
    }
    if (s_async[device].scratch == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(s_async_mutex);
    repack_async_state & st = s_async[device];
    if (st.scratch != nullptr) {
        GGML_ASSERT(st.cur == nullptr && "repack async upload: tensor incomplete at release");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(st.scratch));
        st.scratch = nullptr;
        st.cap     = 0;
    }
}

// Set-tensor override: repack supported Q8_0 weights into the GCN layout on upload;
// everything else is copied through unchanged.
static void ggml_backend_cuda_repack_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {

    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buffer->buft->context;
    ggml_cuda_set_device(ctx->device);

    if (!ggml_cuda_repack_tensor_supported(tensor)) {
        CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size,
            cudaMemcpyHostToDevice, cudaStreamPerThread));
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return;
    }

    const size_t t_nbytes = ggml_nbytes(tensor);

    if (offset != 0 || size != t_nbytes) {
        // Partial write (the meta splitter under -sm tensor): accumulate into a
        // persistent per-device host staging buffer - grown on demand, never
        // zero-filled, never freed between tensors - then repack once complete.
        // Two measured wrong shapes for this path: a fresh zero-filled vector
        // per lane tensor (~74 s of a 96 s MoE load: ~35 GB of value-init), and
        // streaming segments to the device scratch (207 s: tens of thousands of
        // tiny pageable H2D driver round-trips). The segments must stay host-
        // side and cheap; the tensor uploads once.
        // Completed tensors go to the DEVICE repack path - one big async H2D of
        // canonical bytes from pinned staging plus the repack kernel, the same
        // machinery that put -sm layer loads at vanilla parity. Two pinned
        // slots per device, event-guarded, so the next tensor's segments never
        // overwrite an upload still in flight. (Streaming individual segments
        // there instead measured 207 s - the granularity matters, not the path.)
        static std::mutex staging_mutex;
        struct dev_staging {
            uint8_t *    buf     = nullptr;   // pinned
            size_t       cap     = 0;
            const void * key     = nullptr;
            size_t       filled  = 0;
            cudaEvent_t  ev      = nullptr;
            bool         ev_live = false;
        };
        static dev_staging s_stage[GGML_CUDA_MAX_DEVICES][2];
        static int         s_stage_turn[GGML_CUDA_MAX_DEVICES] = {};

        std::lock_guard<std::mutex> lock(staging_mutex);
        const int dev = ctx->device;
        dev_staging & d = s_stage[dev][s_stage_turn[dev]];
        if (d.key != tensor->data && d.key != nullptr) {
            GGML_ASSERT(d.filled == 0 && "repack staging: previous tensor incomplete");
        }
        if (d.ev_live) {
            // slot's previous upload must land before its bytes are overwritten
            CUDA_CHECK(cudaEventSynchronize(d.ev));
            d.ev_live = false;
        }
        if (d.cap < t_nbytes) {
            if (d.buf != nullptr) {
                CUDA_CHECK(cudaFreeHost(d.buf));
            }
            CUDA_CHECK(cudaMallocHost((void **) &d.buf, t_nbytes));
            d.cap = t_nbytes;
        }
        if (d.key != tensor->data) {
            d.key    = tensor->data;
            d.filled = 0;
        }
        GGML_ASSERT(offset + size <= t_nbytes);
        memcpy(d.buf + offset, data, size);
        d.filled += size;
        if (d.filled == t_nbytes) {
            d.key    = nullptr;
            d.filled = 0;
            if (s_upl_stream[dev] == nullptr) {
                CUDA_CHECK(cudaStreamCreateWithFlags(&s_upl_stream[dev], cudaStreamNonBlocking));
            }
            ggml_cuda_repack_set_tensor_async(dev, s_upl_stream[dev], tensor, d.buf, 0, t_nbytes);
            if (d.ev == nullptr) {
                CUDA_CHECK(cudaEventCreateWithFlags(&d.ev, cudaEventDisableTiming));
            }
            CUDA_CHECK(cudaEventRecord(d.ev, s_upl_stream[dev]));
            d.ev_live = true;
            s_stage_turn[dev] ^= 1;
        }
        return;
    }

    repack_and_upload(buffer, tensor, (const uint8_t *) data);
}

// The CUDA buffer's own free_buffer, captured once so the repack wrapper can
// purge its view-cache entries and then delegate.
static void (*s_cuda_free_buffer)(ggml_backend_buffer_t) = nullptr;

static void ggml_backend_cuda_repack_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buffer->buft->context;
    repack_view_cache_purge(ctx->device, ggml_backend_buffer_get_base(buffer),
        ggml_backend_buffer_get_size(buffer));
    if (s_cuda_free_buffer != nullptr) {
        s_cuda_free_buffer(buffer);
    }
}

static ggml_backend_buffer_t ggml_backend_cuda_repack_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buft->context;

    ggml_backend_buffer_t buffer =
        ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(ctx->device), size);
    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.set_tensor  = ggml_backend_cuda_repack_buffer_set_tensor;
    s_cuda_free_buffer        = buffer->iface.free_buffer;
    buffer->iface.free_buffer = ggml_backend_cuda_repack_buffer_free_buffer;
    // Weights are write-once; the inherited CUDA get_tensor would misread the
    // two-plane layout as canonical, so it must be explicitly nulled
    // (ggml_backend_tensor_get has no null guard).
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    buffer->iface.set_tensor_2d = nullptr;
    buffer->iface.get_tensor_2d = nullptr;
    return buffer;
}

static size_t ggml_backend_cuda_repack_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128;
}

static size_t ggml_backend_cuda_repack_buffer_type_get_alloc_size(
        ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    if (ggml_cuda_repack_tensor_supported(tensor)) {
        return repack_gcn_nbytes(tensor->type, tensor->ne[0], tensor->ne[1]) * tensor->ne[2];
    }
    return ggml_nbytes(tensor);
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_repack_buffer_type_interface = {
    ggml_backend_cuda_repack_buffer_type_get_name,
    ggml_backend_cuda_repack_buffer_type_alloc_buffer,
    ggml_backend_cuda_repack_buffer_type_get_alignment,
    nullptr,
    ggml_backend_cuda_repack_buffer_type_get_alloc_size,
    nullptr,
};

// Repacked buffer type: gfx906 only, else returns nullptr so the caller falls
// back to the normal CUDA buffer type. Turning repack off is handled a level
// up: --no-repack clears use_extra_bufts, so this buft is never asked for.
// Gating discovery here makes "repack off" fully native: no repack buft is
// offered to the loader, so no structural repack code (meta supports_op,
// set_tensor bypass, fusion suppression) is ever active.
ggml_backend_buffer_type_t ggml_backend_cuda_repack_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }
    // Kernels only compile for gfx906 (__gfx906__ guard); a wider GCN gate would
    // hand two-plane weights to archs whose repack kernels are NO_DEVICE_CODE.
    if (ggml_cuda_info().devices[device].cc != GGML_CUDA_CC_VEGA20) {
        return nullptr;
    }
    static ggml_backend_buffer_type buft_storage[GGML_CUDA_MAX_DEVICES];
    static bool initialized[GGML_CUDA_MAX_DEVICES] = {};

    if (!initialized[device]) {
        buft_storage[device] = {
            ggml_backend_cuda_repack_buffer_type_interface,
            ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
            new ggml_backend_cuda_repack_buffer_type_context{
                                 device, GGML_CUDA_NAME + std::to_string(device) + "_Repacked"},
        };
        initialized[device] = true;
    }
    return &buft_storage[device];
}
