#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"

#if defined(__linux__)
#include <execinfo.h>
#endif
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis) {
    switch (split_axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
            return "0";
        case GGML_BACKEND_SPLIT_AXIS_1:
            return "1";
        case GGML_BACKEND_SPLIT_AXIS_2:
            return "2";
        case GGML_BACKEND_SPLIT_AXIS_3:
            return "3";
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            return "MIRRORED";
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            return "PARTIAL";
        case GGML_BACKEND_SPLIT_AXIS_NONE:
            return "NONE";
        case GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            return "UNKNOWN";
        default:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    size_t tps      = 0; // TP group size (devices in one stage). Equal to simple_devs.size() for single-stage.
    size_t n_stages = 1; // simple_devs.size() / tps

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, size_t tps,
            ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        const size_t n_devs = this->simple_devs.size();
        this->tps      = (tps == 0) ? n_devs : tps;
        this->n_stages = n_devs / this->tps;
        name        = std::string("Meta(");
        description = std::string("Meta(");
        for (size_t i = 0; i < n_devs; i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (this->simple_devs[i]);
            description += ggml_backend_dev_description(this->simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, tps, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.tps, other.get_split_state, other.get_split_state_ud);
    }
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->name.c_str();
}

static const char * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->description.c_str();
}

static void ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    *free  = 0;
    *total = 0;
    for (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *free  += tmp_free;
        *total += tmp_total;
    }
}

static enum ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false, // get_host_buffer_type works, but exposing it changes scheduler host-buffer placement
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ true,  // proxied via simple devices (fan-out)
        /* .mmap_support          = */ true,
    };
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
    }
}

struct ggml_backend_meta_event_context {
    std::vector<ggml_backend_event_t> simple_events;
};

static ggml_backend_event_t ggml_backend_meta_device_event_new(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    auto * ev_ctx = new ggml_backend_meta_event_context();
    ev_ctx->simple_events.reserve(meta_dev_ctx->simple_devs.size());
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_event_t simple_ev = ggml_backend_event_new(simple_dev);
        if (simple_ev == nullptr) {
            for (ggml_backend_event_t e : ev_ctx->simple_events) {
                ggml_backend_event_free(e);
            }
            delete ev_ctx;
            return nullptr;
        }
        ev_ctx->simple_events.push_back(simple_ev);
    }
    return new ggml_backend_event{dev, ev_ctx};
}

static void ggml_backend_meta_device_event_free(ggml_backend_dev_t /*dev*/, ggml_backend_event_t event) {
    auto * ev_ctx = (ggml_backend_meta_event_context *) event->context;
    for (ggml_backend_event_t e : ev_ctx->simple_events) {
        ggml_backend_event_free(e);
    }
    delete ev_ctx;
    delete event;
}

static void ggml_backend_meta_device_event_synchronize(ggml_backend_dev_t /*dev*/, ggml_backend_event_t event) {
    auto * ev_ctx = (ggml_backend_meta_event_context *) event->context;
    for (ggml_backend_event_t e : ev_ctx->simple_events) {
        ggml_backend_event_synchronize(e);
    }
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_buffer_type_t * ggml_backend_meta_device_get_extra_bufts(ggml_backend_dev_t dev);

static bool ggml_backend_meta_buft_is_repack(ggml_backend_buffer_type_t buft);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const ggml_tensor * w = op->src[i];
        if (w == nullptr || w->buffer == nullptr) continue;
        ggml_backend_buffer_type_t w_buft = ggml_backend_buffer_get_type(w->buffer);
        if (!ggml_backend_buft_is_meta(w_buft) || !ggml_backend_meta_buft_is_repack(w_buft)) continue;

        // a repacked weight can only be consumed as src0
        if (i != 0) return false;
        const bool ok_mm   = op->op == GGML_OP_MUL_MAT    && ggml_n_dims(w) == 2;
        const bool ok_mmid = op->op == GGML_OP_MUL_MAT_ID && ggml_n_dims(w) == 3 &&
                             op->src[2] != nullptr && op->src[2]->type == GGML_TYPE_I32;
        if (!ok_mm && !ok_mmid) return false;

        // Keep in sync with the types ggml_cuda_repack_tensor_supported admits.
        // The per-type ne0 alignment and the per-lane axis-0 split check follow
        // from ggml_blck_size below, so per-lane slices that are not
        // block-aligned correctly stay canonical.
        switch (w->type) {
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_MXFP4:
                break;
            default:
                return false;
        }
        if (op->src[1] == nullptr || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;

        // Buffer-independent repackability: enforced here during buft selection so the
        // gate agrees with ggml_cuda_repack_mul_mat_should_fire at dispatch. The CUDA
        // all_of delegation below cannot cover this case: selection always calls
        // supports_op with a null-data dummy buffer, so the early return below would
        // otherwise skip it.
        const int64_t blck = ggml_blck_size(w->type);
        if (w->ne[0] % blck != 0) return false;
        if (!ggml_is_contiguous(w)) return false;

        if (w->data == nullptr) {
            // Buffer not allocated yet (buft selection). Only the per-lane split
            // alignment below is deferred: it needs the real split state, and
            // calculate_split_state aborts loudly on a misaligned axis-0 split.
            return true;
        }

        const ggml_backend_meta_split_state ss =
            meta_dev_ctx->get_split_state(w, meta_dev_ctx->get_split_state_ud);
        if (ss.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            // rows are split per-lane; each lane's row count must stay block-aligned
            // (calculate_split_state already asserts this for axis-0 weights)
            const size_t n_bufs = meta_dev_ctx->simple_devs.size();
            for (size_t s = 0; s < ss.n_segments; s++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    if (ss.ne[s*n_bufs + j] % blck != 0) return false;
                }
            }
        }
    }
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    if (!ggml_backend_dev_is_meta(dev_buft)) {
        return false;
    }
    const ggml_backend_meta_device_context * meta_dev_ctx      = (const ggml_backend_meta_device_context *) dev->context;
    const ggml_backend_meta_device_context * meta_buft_dev_ctx = (const ggml_backend_meta_device_context *) dev_buft->context;
    // Exact match, or one device list a prefix of the other. The prefix case is a
    // subset draft group borrowing the target's tensors: lane i of the narrower
    // Meta device is device i of the wider one, so per-device storage stays
    // index-aligned. Only tensors mirrored on lanes the narrow side owns are
    // reachable, which the graph-build transfer path already enforces.
    const size_t n = std::min(meta_dev_ctx->simple_devs.size(), meta_buft_dev_ctx->simple_devs.size());
    for (size_t i = 0; i < n; i++) {
        if (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}

static const char * ggml_backend_meta_reg_get_name(ggml_backend_reg_t) {
    return "Meta";
}

static size_t ggml_backend_meta_reg_get_device_count(ggml_backend_reg_t) {
    return 0;
}

static ggml_backend_dev_t ggml_backend_meta_reg_get_device(ggml_backend_reg_t, size_t) {
    return nullptr;
}

static void * ggml_backend_meta_reg_get_proc_address(ggml_backend_reg_t, const char * name) {
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        return (void *) ggml_backend_meta_device_get_extra_bufts;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_meta_reg_iface = {
    /* .get_name          = */ ggml_backend_meta_reg_get_name,
    /* .get_device_count  = */ ggml_backend_meta_reg_get_device_count,
    /* .get_device        = */ ggml_backend_meta_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_meta_reg_get_proc_address,
};

static ggml_backend_reg ggml_backend_meta_reg = {
    /* .api_version = */ GGML_BACKEND_API_VERSION,
    /* .iface       = */ ggml_backend_meta_reg_iface,
    /* .context     = */ nullptr,
};

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ ggml_backend_meta_device_event_new,
    /* .event_free           = */ ggml_backend_meta_device_event_free,
    /* .event_synchronize    = */ ggml_backend_meta_device_event_synchronize,
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

static size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

static ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, size_t tps,
        ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs > 0);
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    GGML_ASSERT(tps == 0 || (tps > 0 && n_devs % tps == 0));
    // TODO: this is not thread-safe - needs to be fixed
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, tps, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    // Only publish a registry when at least one lane actually offers extra buffer
    // types. Otherwise leave it null, exactly as before, so no caller of
    // ggml_backend_dev_backend_reg() sees a behaviour change.
    bool any_extra_bufts = false;
    for (ggml_backend_dev_t simple_dev : simple_devs) {
        ggml_backend_reg_t r = ggml_backend_dev_backend_reg(simple_dev);
        auto fn = r ? (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(r, "ggml_backend_dev_get_extra_bufts") : nullptr;
        if (fn != nullptr) {
            ggml_backend_buffer_type_t * e = fn(simple_dev);
            if (e != nullptr && *e != nullptr) {
                any_extra_bufts = true;
                break;
            }
        }
    }

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ any_extra_bufts ? &ggml_backend_meta_reg : nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

struct ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    bool repack = false;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts, bool repack = false)
        : simple_bufts(std::move(simple_bufts)), repack(repack) {
        name = "Meta(";
        for (size_t i = 0; i < this->simple_bufts.size(); i++) {
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(this->simple_bufts[i]);
        }
        name += ")";
    }

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return std::tie(simple_bufts, repack) < std::tie(other.simple_bufts, other.repack);
    }
};

static size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) buft->context;
    return meta_buft_ctx->name.c_str();
}

static ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

static size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return max_size;
}

static size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    return max_alloc_size;
}

static bool ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        if (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            return false;
        }
    }
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_meta_buffer_type_is_host,
};

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static bool ggml_backend_meta_buft_is_repack(ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_meta(buft) &&
           ((const ggml_backend_meta_buffer_type_context *) buft->context)->repack;
}

static ggml_backend_buffer_type_t ggml_backend_meta_buffer_type_from_simple(
        ggml_backend_dev_t dev, std::vector<ggml_backend_buffer_type_t> simple_bufts, bool repack) {
    static std::mutex mutex;
    static std::map<std::tuple<ggml_backend_dev_t, std::vector<ggml_backend_buffer_type_t>, bool>,
                    struct ggml_backend_buffer_type> cache;
    std::lock_guard<std::mutex> lock(mutex);

    // The buffer type retains dev, whose split callback can carry model-local
    // userdata. Reusing it for another Meta device would leave tensor split
    // queries bound to the first model after that model has been freed.
    auto key = std::make_tuple(dev, simple_bufts, repack);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return &it->second;
    }
    auto * buft_ctx = new ggml_backend_meta_buffer_type_context(std::move(simple_bufts), repack);
    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    return &cache.emplace(std::move(key), meta_buft).first->second;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }

    const size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = new ggml_backend_meta_buffer_type_context(simple_bufts);

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        if (simple_host_buft == nullptr) {
            return nullptr;
        }
        if (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } else if (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            return nullptr;
        }
    }
    return host_buft;
}

static ggml_backend_buffer_type_t * ggml_backend_meta_device_get_extra_bufts(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    const size_t n_bufs = meta_dev_ctx->simple_devs.size();

    static std::mutex mutex;
    static std::map<ggml_backend_dev_t, std::vector<ggml_backend_buffer_type_t>> cache;
    std::lock_guard<std::mutex> lock(mutex);

    auto & result = cache[dev];
    if (!result.empty()) {
        return result.data();
    }

    // fetch each lane's NULL-terminated extra buft list once
    std::vector<std::vector<ggml_backend_buffer_type_t>> per_lane(n_bufs);
    size_t n_slots = SIZE_MAX;
    for (size_t k = 0; k < n_bufs; k++) {
        ggml_backend_dev_t simple_dev = meta_dev_ctx->simple_devs[k];
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(simple_dev);
        auto fn = reg ? (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts") : nullptr;
        if (fn != nullptr) {
            for (ggml_backend_buffer_type_t * e = fn(simple_dev); e != nullptr && *e != nullptr; e++) {
                per_lane[k].push_back(*e);
            }
        }
        n_slots = std::min(n_slots, per_lane[k].size());
    }

    // slot-wise composition: repack on EVERY lane, or not offered at all
    for (size_t s = 0; s < n_slots; s++) {
        std::vector<ggml_backend_buffer_type_t> slot_bufts;
        slot_bufts.reserve(n_bufs);
        for (size_t k = 0; k < n_bufs; k++) {
            slot_bufts.push_back(per_lane[k][s]);
        }
        result.push_back(ggml_backend_meta_buffer_type_from_simple(dev, std::move(slot_bufts), /*repack=*/true));
    }

    result.push_back(nullptr);
    GGML_LOG_DEBUG("meta extra bufts: dev=%p n_bufs=%zu n_slots=%zu result[0]=%s\n",
        (void *) dev, n_bufs, n_slots, result.size() > 1 ? ggml_backend_buft_name(result[0]) : "(none)");
    return result.data();
}

//

// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
struct ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::map<const ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(const ggml_init_params & params, const int n_simple) {
        ctxs.reserve(n_simple);
        for (int i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

struct ggml_backend_meta_buffer_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[2];
    int stc_compute_index      = 0;
    int stc_compute_index_next = 0;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    static constexpr size_t nbtc = GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding);
    std::map<std::pair<const ggml_tensor *, bool>, std::pair<ggml_backend_meta_split_state, char[nbtc]>> split_state_cache;

    int debug;

    ggml_backend_meta_buffer_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            const std::vector<ggml_backend_buffer_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute{std::move(stc_compute_0), std::move(stc_compute_1)} {
        this->bufs.reserve(bufs.size());
        for (ggml_backend_buffer_t buf : bufs) {
            this->bufs.emplace_back(buf);
        }
        const char * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(const ggml_tensor * tensor) {
        if (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            return stc_static;
        }
        return stc_compute[stc_compute_index];
    }
};

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    delete buf_ctx;
}

static size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->bufs.size();
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());
    return buf_ctx->bufs[index].get();
}

static struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    auto it = stc.simple_tensors.find(tensor);
    if (it == stc.simple_tensors.end()) {
        return nullptr;
    }
    return it->second[index];
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync);

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
        ggml_backend_meta_simple_tensor_container & stc, const struct ggml_tensor * tensor, bool assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    auto split_states_equal = [&](const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) -> bool {
        if (a.axis != b.axis) {
            return false;
        }
        for (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            for (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t sum_b = 0;
            for (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            if (sum_a != sum_b) {
                return false;
            }
        }
        return true;
    };

    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } else if (!split_states_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                break;
            }
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN) {
            // This aborts immediately below, so name the node and its sources rather than
            // failing with only a line number. An unclassifiable op is normally an arch
            // whose routing has not been taught to the meta backend yet.
            GGML_LOG_ERROR("%s: cannot infer split state for node '%s' op=%s scalar_only=%d, srcs:\n",
                           __func__, tensor->name, ggml_op_name(tensor->op), (int) scalar_only);
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                    continue;
                }
                GGML_LOG_ERROR("    src%zu '%s' axis=%d\n",
                               i, tensor->src[i]->name, (int) src_ss[i].axis);
            }
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        return ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        return src_ss[0];
    };

    // SUM_ROWS produces a partial result when the input is split on axis 0:
    // each rank sums its slice locally, then the
    // PARTIAL state forces an AllReduce when a downstream op needs the full value.
    // For axis >= 1 the reduction is purely per-row and stays in the input's split.
    auto handle_axis0_reduce = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        return src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        if (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            return src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_concat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        const ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            return src_ss[1];
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            return src_ss[0];
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            return ret;
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        // Batched matmul against a BROADCAST shared operand: src0 is mirrored with a
        // singleton batch (deepseek4's single KV latent under -fa 0) while src1 carries a
        // head split on a batch axis. Each device matmuls only its own heads against the
        // shared operand, so the result keeps src1's split and owes no reduction.
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_3)) {
            GGML_ASSERT(tensor->src[0]->ne[src_ss[1].axis] == 1);
            return src_ss[1];
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        if (src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (int) src_ss[0].axis, tensor->src[1]->name, (int) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_reshape = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                int64_t base_ne_in = 1;
                for (int dim = 0; dim <= src_ss[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                if (src_ss[0].n_segments == 1) {
                    base_ne_in /= src_ss[0].nr[0];
                    if (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        return {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                    }
                    if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && src_ss[0].nr[0] == 1) {
                        bool complete_rows = true;
                        for (size_t j = 0; j < n_bufs; j++) {
                            const int64_t ne = src_ss[0].ne[j];
                            complete_rows = complete_rows && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        if (complete_rows) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t base_ne_out = 1;
                for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    base_ne_out *= tensor->ne[dim];
                    if (base_ne_out % base_ne_in == 0) {
                        return {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out/base_ne_in)}, 1};
                    }
                    if (base_ne_out > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_cpy = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            return handle_reshape(src_ss);
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_view = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            return handle_reshape(src_ss);
        }
        const int axis = src_ss[0].axis;
        {
            bool all_strides_the_same = true;
            for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                if (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    continue;
                }
                if (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = false;
                    break;
                }
            }
            if (all_strides_the_same) {
                return src_ss[0];
            }
        }
        if (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            for (int dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                if (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_permute = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_transpose = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(int(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3:
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_get_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_set_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]));
        return src_ss[0];
    };

    auto handle_rope = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return src_ss[0];
    };

    auto handle_pad = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        return src_ss[0];
    };

    auto handle_flash_attn_ext = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(tensor->src[3] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

        // Design A: fully replicated attention, every device runs it identically -> MIRRORED.
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }

        // MLA/MQA head-split: Q is head-split while the single shared KV latent stays MIRRORED.
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        const bool kv_split = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2;
        const bool kv_mirrored = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        GGML_ASSERT(kv_split || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_lightning_indexer = [&](
            const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        for (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    };

    auto handle_ssm_conv = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == src_ss[1].axis) {
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
            }
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_gated_delta_net = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // state shape is (S_v*S_v*H, K, n_seqs); the heads dim is nested inside axis 0,
        // so a head-aligned split on the input cache reshapes to axis 0 here (not axis 2).
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_nelements(tensor) == 0) {
            return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                const int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                for (size_t s = 0; s < ret.n_segments; s++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            } else if (ret.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            return ret;
        }

        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1});
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                continue;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(stc, tensor->src[i], /*assume_sync =*/ true);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            case GGML_OP_NONE: {
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ADD:
            case GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_ADD1:
            case GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
            case GGML_OP_SIN:
            case GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUM_ROWS: {
                split_state = handle_axis0_reduce(src_ss);
            } break;
            case GGML_OP_CUMSUM:
            case GGML_OP_MEAN:
            case GGML_OP_ARGMAX:
            case GGML_OP_COUNT_EQUAL: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_REPEAT:
            case GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } break;
            case GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_RMS_NORM_BACK:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } break;
            case GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } break;
            case GGML_OP_CONT:
            case GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } break;
            case GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } break;
            case GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } break;
            case GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } break;
            case GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } break;
            case GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } break;
            case GGML_OP_DIAG:
            case GGML_OP_DIAG_MASK_INF:
            case GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_ROPE_BACK: {
                // Same data layout and per-head independence as the forward rope, so it
                // carries a head split identically. It was scalar_only only because no
                // arch had yet fed it split data - deepseek4's de-rope of the attention
                // output does, under the MLA head split.
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONV_TRANSPOSE_1D:
            case GGML_OP_IM2COL:
            case GGML_OP_IM2COL_BACK:
            case GGML_OP_IM2COL_3D:
            case GGML_OP_CONV_2D:
            case GGML_OP_CONV_3D:
            case GGML_OP_CONV_2D_DW:
            case GGML_OP_CONV_TRANSPOSE_2D:
            case GGML_OP_POOL_1D:
            case GGML_OP_POOL_2D:
            case GGML_OP_POOL_2D_BACK:
            case GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } break;
            case GGML_OP_PAD_REFLECT_1D:
            case GGML_OP_ROLL:
            case GGML_OP_ARANGE:
            case GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } break;
            case GGML_OP_SSM_SCAN:
            case GGML_OP_WIN_PART:
            case GGML_OP_WIN_UNPART:
            case GGML_OP_GET_REL_POS:
            case GGML_OP_ADD_REL_POS:
            case GGML_OP_RWKV_WKV6:
            case GGML_OP_GATED_LINEAR_ATTN:
            case GGML_OP_RWKV_WKV7:
            case GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } break;
            case GGML_OP_LIGHTNING_INDEXER: {
                split_state = handle_lightning_indexer(src_ss);
            } break;
            case GGML_OP_DSV4_HC_COMB:
            case GGML_OP_DSV4_HC_PRE:
            case GGML_OP_DSV4_HC_POST: {
                // DeepSeek-V4 custom attention ops. Replicated under design A (mirrored in, mirrored out).
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
            case GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CROSS_ENTROPY_LOSS:
            case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_OPT_STEP_ADAMW:
            case GGML_OP_OPT_STEP_SGD:
            case GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            default: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            } break;
        }
        if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            bool first_src_split_by_axis = true;
            const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    continue;
                }
                if (first_src_split_by_axis) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        if (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            const int64_t div = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            GGML_ASSERT(split_state.ne[j] % div == 0);
                            split_state.ne[j] /= div;
                        }
                    }
                } else {
                    GGML_ASSERT(split_state.n_segments == 1);
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t sum = 0;
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        const int64_t ratio_lhs = split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis];
                        const int64_t ratio_rhs = sum * tensor->ne[split_state.axis];
                        if (ratio_lhs != ratio_rhs) {
                            GGML_LOG_ERROR(
                                "META SPLIT RATIO MISMATCH: node '%s' op=%s axis=%d ne[axis]=%lld dev=%zu ne=%lld nr=%u"
                                " | src%zu '%s' axis=%d ne[axis]=%lld sum=%lld | lhs=%lld rhs=%lld\n",
                                tensor->name, ggml_op_name(tensor->op), (int) split_state.axis,
                                (long long) tensor->ne[split_state.axis], j,
                                (long long) split_state.ne[j], split_state.nr[0],
                                i, tensor->src[i]->name, (int) src_ss[i].axis,
                                (long long) tensor->src[i]->ne[src_ss[i].axis], (long long) sum,
                                (long long) ratio_lhs, (long long) ratio_rhs);
                        }
                        GGML_ASSERT(ratio_lhs == ratio_rhs);
                    }
                }
                first_src_split_by_axis = false;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        return split_state;
    };

    const std::pair key = std::make_pair(tensor, assume_sync);
    auto it = buf_ctx->split_state_cache.find(key);
    if (it != buf_ctx->split_state_cache.end() && memcmp(it->second.second, (const char *) tensor, sizeof(it->second.second)) != 0) {
        buf_ctx->split_state_cache.clear();
        it = buf_ctx->split_state_cache.end();
    }

    if (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[key].first = calculate_split_state();
        memcpy(buf_ctx->split_state_cache[key].second, tensor, sizeof(buf_ctx->split_state_cache[key].second));
        if (buf_ctx->debug > 0) {
            std::string srcs_info;
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                    continue;
                }
                if (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                const ggml_backend_meta_split_state split_state =
                        ggml_backend_meta_get_split_state(tensor->src[i], true);
                GGML_ASSERT(split_state.n_segments == 1);
                const char * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                for (size_t j = 0; j < n_bufs; j++) {
                    if (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]) + "x" + std::to_string(split_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ne_info.empty()) {
                    ne_info += ", ";
                }
                const ggml_backend_meta_split_state & ss = buf_ctx->split_state_cache[key].first;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(buf_ctx->split_state_cache[key].first.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[key].first;
    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        for (size_t s = 0; s < ret.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        assert(ne_ret == tensor->ne[int(ret.axis)]);
    }
#endif // NDEBUG
    return ret;
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    return ggml_backend_meta_get_split_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

static void * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000; // FIXME
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync =*/ true);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    int split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    for (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].get();
        ggml_backend_buffer_t   simple_buf = buf_ctx->bufs[j].get();

        if ((simple_buf != nullptr) && ggml_backend_buffer_is_multi_buffer(simple_buf)) {
            // see https://github.com/ggml-org/llama.cpp/issues/22197
            GGML_ABORT("multi buffers are not supported by the meta backend");
        }

        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            for (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j] * split_state.nr[s];
            }
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            if (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                const int split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true).axis;
                GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                bool split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    const size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    if (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = true;
                        break;
                    }
                }
                if (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
        } else if (simple_buf != nullptr) {
            t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
        }
        t_ij->extra = tensor->extra;
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            if (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } else if (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] == nullptr || !ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            continue;
        }

        const ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        if (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            continue;
        }
        for (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            for (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_simple_bufs + j] * split_state_src.nr[s];
            }
            if (ne_sum == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    buf_ctx->stc_compute_index = buf_ctx->stc_compute_index_next;
    return ggml_backend_meta_buffer_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

static void ggml_backend_meta_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state =
            ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        for (int64_t row = 0; row < row_count; row++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (row_start + row)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            return;
        }

        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    for (int64_t row = 0; row < row_count; row++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (row_start + row)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        / chunk_size_full;
            const int64_t i_stop  = (offset + size) / chunk_size_full;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size == 0) {
                    continue;
                }
                // each simple tensor is contiguous, so its target chunks are packed
                ggml_backend_tensor_memset(simple_tensor, value, i_start*chunk_size, (i_stop - i_start)*chunk_size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(offset % sizeof(float) == 0);
            GGML_ASSERT(size   % sizeof(float) == 0);
            const size_t n_values = size / sizeof(float);
            size_t n_contributors = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                n_contributors += split_state.ne[j] != 0;
            }
            const bool has_contributor_mask = n_contributors != 0;
            if (!has_contributor_mask) {
                n_contributors = n_bufs;
            }
            std::vector<float> tmp(n_values);
            for (size_t i = 0; i < n_values; i++) {
                tmp[i] = ((const float *) data)[i] / n_contributors;
            }
            std::vector<float> zero;
            if (has_contributor_mask) {
                zero.resize(n_values, 0.0f);
            }
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const float * partial = has_contributor_mask && split_state.ne[j] == 0 ? zero.data() : tmp.data();
                ggml_backend_tensor_set(simple_tensor, partial, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++){
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_meta_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    const ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    bufs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != nullptr);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));
    }
    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    const ggml_init_params params_static = {
        /*.mem_size   =*/ ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    const ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom*ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_buffer_t> bufs(n_simple_bufts, nullptr);
    ggml_backend_meta_buffer_context * meta_buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    ggml_context * ctx_meta = ctx;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        bool any_nonzero_slice = false;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_nelements(t) != 0) {
                any_nonzero_slice = true;
                break;
            }
        }
        if (any_nonzero_slice) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } else {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_buffer(simple_buft, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = meta_buf_ctx->bufs[i].get();
            }
        }
        if (!meta_buf_ctx->bufs[i]) {
            // Out of memory on one device is a load failure, not a bug: hand the
            // caller the same nullptr a plain buffer type returns, with the meta
            // tensors un-stamped, so llama reports "unable to allocate" and exits.
            GGML_LOG_ERROR("%s: failed to allocate the %s slice %zu of a meta buffer%c",
                __func__, ggml_backend_buft_name(simple_buft), i, 10);
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta); t != nullptr; t = ggml_get_next_tensor(ctx_meta, t)) {
                t->buffer = nullptr;
                t->data   = nullptr;
            }
            ggml_backend_buffer_free(meta_buf);
            return nullptr;
        }
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()));
    }
    return meta_buf;
}

//
// meta backend
//

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

// Persistent worker pool that issues one lane's graph_compute_async per thread, so all
// lanes of a subgraph start together instead of in device order. See the comment on
// ggml_backend_meta_context::parallel_dispatch for the measurement that motivates it.
//
// Wait strategy: a SHORT bounded spin to catch the fast case without a syscall, then BLOCK
// on a condition variable. An earlier version spun indefinitely, which pegged one core per
// worker - 7 of 16 cores at 8 lanes - because the idle gap between fork-joins is roughly
// token_time/n_subgraphs, on the order of 160 us. That is far too long to busy-wait, and
// it is antisocial on a shared machine. The ~5-10 us wake cost is small against the
// per-lane issue cost the pool exists to overlap (about 200 us of stagger at 8 lanes).
// The spin count is not worth tuning: 0 and 512 measure the same, and 65536 doubles
// host CPU for 1.3% less throughput.
//
// Thread safety: ggml_cuda_set_device resolves the current device through cudaGetDevice,
// which is thread-local in the CUDA/HIP runtime, and every per-lane backend owns its own
// context, streams, graph cache and memory pool. Lanes therefore share no mutable state.
static inline uint64_t ggml_meta_prof_now() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Accumulates into *acc for the lifetime of the scope. Used on graph_compute
// and synchronize, both of which have many early returns.
struct ggml_meta_prof_scope {
    uint64_t * acc;
    uint64_t   t0;
    bool       on;
    ggml_meta_prof_scope(uint64_t * a, bool o) : acc(a), t0(o ? ggml_meta_prof_now() : 0), on(o) {}
    ~ggml_meta_prof_scope() { if (on) { *acc += ggml_meta_prof_now() - t0; } }
};

struct ggml_backend_meta_lane_dispatcher {
    struct job {
        ggml_backend_t backend = nullptr;
        ggml_cgraph *  cgraph  = nullptr;
        ggml_status    status  = GGML_STATUS_SUCCESS;
    };


    std::vector<std::thread> workers;
    std::vector<job>         jobs;
    std::atomic<uint64_t>    generation{0};
    std::atomic<uint32_t>    remaining{0};
    std::atomic<bool>        stop{false};

    std::mutex              mtx;
    std::condition_variable cv_work;   // main -> workers
    std::condition_variable cv_done;   // workers -> main

    static const int spin_iters = 512;

    ~ggml_backend_meta_lane_dispatcher() { shutdown(); }

    void shutdown() {
        if (workers.empty()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mtx);
            stop.store(true, std::memory_order_release);
            generation.fetch_add(1, std::memory_order_release);
        }
        cv_work.notify_all();
        for (std::thread & t : workers) {
            if (t.joinable()) {
                t.join();
            }
        }
        workers.clear();
        jobs.clear();
    }

    // n_workers is lane_count-1: the calling thread always runs one lane itself.
    void ensure(size_t n_workers) {
        if (workers.size() == n_workers) {
            return;
        }
        shutdown();
        if (n_workers == 0) {
            return;
        }
        jobs.assign(n_workers, job{});
        stop.store(false, std::memory_order_relaxed);
        generation.store(0, std::memory_order_relaxed);
        remaining.store(0, std::memory_order_relaxed);
        workers.reserve(n_workers);
        for (size_t w = 0; w < n_workers; w++) {
            workers.emplace_back([this, w]() {
                uint64_t seen = 0;
                while (true) {
                    // Short spin first: if the main thread is already publishing, this
                    // catches it without paying a futex round trip.
                    uint64_t gen = generation.load(std::memory_order_acquire);
                    for (int s = 0; gen == seen && s < spin_iters; s++) {
                        gen = generation.load(std::memory_order_acquire);
                    }
                    if (gen == seen) {
                        std::unique_lock<std::mutex> lk(mtx);
                        cv_work.wait(lk, [&] {
                            return generation.load(std::memory_order_acquire) != seen;
                        });
                        gen = generation.load(std::memory_order_acquire);
                    }
                    seen = gen;
                    if (stop.load(std::memory_order_acquire)) {
                        return;
                    }
                    job & jb = jobs[w];
                    if (jb.cgraph != nullptr) {
                        jb.status = ggml_backend_graph_compute_async(jb.backend, jb.cgraph);
                    }
                    if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        std::lock_guard<std::mutex> lk(mtx);
                        cv_done.notify_one();
                    }
                }
            });
        }
    }

    // Publish one job per worker and release them. Every worker decrements `remaining`
    // exactly once per round, including workers left with a null cgraph, so the join below
    // is balanced regardless of how many lanes were actually filled.
    void dispatch_async() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            remaining.store((uint32_t) workers.size(), std::memory_order_release);
            generation.fetch_add(1, std::memory_order_release);
        }
        cv_work.notify_all();
    }

    ggml_status join() {
        uint32_t left = remaining.load(std::memory_order_acquire);
        for (int s = 0; left != 0 && s < spin_iters; s++) {
            left = remaining.load(std::memory_order_acquire);
        }
        if (left != 0) {
            std::unique_lock<std::mutex> lk(mtx);
            cv_done.wait(lk, [&] {
                return remaining.load(std::memory_order_acquire) == 0;
            });
        }
        ggml_status ret = GGML_STATUS_SUCCESS;
        for (job & jb : jobs) {
            if (jb.cgraph != nullptr && jb.status != GGML_STATUS_SUCCESS) {
                ret = jb.status;
            }
            jb.cgraph = nullptr;
            jb.status = GGML_STATUS_SUCCESS;
        }
        return ret;
    }
};

struct ggml_backend_meta_context {
    struct cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        int           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    struct backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;

        backend_config(ggml_backend_t backend, const size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_steps;
    int                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    // Per-subgraph metadata for multi-stage. closure indicates what to do after the
    // subgraph runs:
    //   NONE     = nothing (last subgraph, end of cgraph)
    //   AR       = AllReduce within the subgraph's stage (boundary was a PARTIAL node)
    //   TRANSFER = lane-pair broadcast from this stage to the next subgraph's stage
    // xfer is populated only when closure == TRANSFER. It holds the MIRRORED graph nodes
    // produced in this stage (or earlier) that are still LIVE (consumed by the new stage's
    // compute). Just transferring the last node of the prior subgraph is insufficient. The
    // residual stream tensor (e.g. l_out-19 in transformer) is produced in the transition
    // subgraph's first node and consumed by the new stage's residual ADDs, separately from
    // the last node (e.g. attn_norm-20).
    // BCAST: copy the subgraph's last node from the stage's first lane to its
    // other lanes. Used after TOP_K selections: the redundant mirrored compute
    // of the selection scores is not bitwise identical across lanes, and once
    // the top-k is live (DSV4 lightning indexer past n_kv = indexer_top_k * 128
    // tokens) a near-tie flip gives every lane a different block selection,
    // hence a different attention mask, and the head-split partials sum an
    // inconsistent mixture. Broadcasting the selection restores the single-GPU
    // semantics of one decision.
    enum class subgraph_closure : int8_t { NONE = 0, AR = 1, TRANSFER = 2, BCAST = 3 };
    struct subgraph_meta {
        size_t                     stage   = 0;
        subgraph_closure           closure = subgraph_closure::NONE;
        std::vector<ggml_tensor *> xfer;
    };
    std::vector<subgraph_meta> subgraphs;

    // tps and n_stages cached from the meta device context. tps == n_devs and n_stages == 1
    // for the vanilla single-stage TP case; tps < n_devs and n_stages > 1 partitions backends
    // into n_stages contiguous blocks of tps simple_backends each, with one comm_ctx per
    // block confining AllReduce to that block's lanes.
    size_t                               tps      = 0;
    size_t                               n_stages = 1;
    std::vector<void *>                  comm_ctxs;       // size == n_stages; entry may be null if comm_init unavailable
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;

    // Which stage last computed a given tensor, recorded during partition.
    //
    // A MIRRORED tensor is only written on the lanes of the stage that produced it.
    // With n_stages == 1 that is every lane, so a reader can take any of them, but
    // with n_stages > 1 a fixed lane index silently returns an untouched buffer for
    // anything another stage computed. A speculative drafter reading target hidden
    // states hits exactly this and gets zeros, which costs acceptance rather than
    // raising anything.
    std::unordered_map<const ggml_tensor *, size_t> tensor_stage;

    // Owning stage of each persistent buffer, kept ACROSS graphs. tensor_stage is
    // cleared per graph, so a graph that only writes a KV buffer (the speculative
    // K/V injection) has no way to learn which stage that buffer belongs to and
    // would default to stage 0, writing a copy nobody reads.
    std::unordered_map<const ggml_tensor *, size_t> persist_buffer_stage;

    // Fragment-mode state. A cgraph with uid == 0 is a graph view from the
    // scheduler's eval-callback path: a subrange of one logical graph, computed one
    // call at a time. Stage context and tensor validity then have to survive across
    // graph_compute calls, and stage-crossing reads have to be transferred at the
    // consuming fragment because the whole-graph walk that would have placed a
    // TRANSFER closure never sees both sides of the dependency.
    int frag_last_stage = 0;
    // Bitmask of stages whose lanes hold valid data for a graph node. Reset to the
    // computing stage when the node is recomputed so a transferred-bit from a
    // previous ubatch cannot go stale.
    std::unordered_map<const ggml_tensor *, uint32_t> frag_stage_bits;
    // Last stage that wrote through a view into a persistent (non-compute) buffer,
    // keyed by the view root. Covers KV-cache writes whose readback view has no
    // src edge to the writing node.
    std::unordered_map<const ggml_tensor *, size_t> frag_root_stage;

    // Split AllReduce. comm_ar_prepare does the shared host setup and returns the
    // number of ranks to issue, or < 0 when this call is not eligible (size gate
    // to NCCL, non-F32, one-shot path), in which case the ordinary comm_allreduce
    // path is used unchanged. Recording a whole token needs the split form: the
    // ranks have to be issued from inside the capture, one lane at a time.
    int  (*comm_ar_prepare)(void *, struct ggml_tensor **) = nullptr;
    void (*comm_ar_launch_rank)(void *, int)               = nullptr;

    // Whole-token graph capture. One recorded graph per lane, replayed each
    // token, so the per-boundary host round trips (and the submission spread
    // the barrier bills for them) collapse to one per token. On by default,
    // GGML_META_TOKEN_GRAPH=0 goes back to per-subgraph dispatch.
    bool token_graph = false;
    bool (*tg_capture_begin)(ggml_backend_t)        = nullptr;
    void * (*tg_capture_end)(ggml_backend_t)        = nullptr;
    void (*tg_graph_launch)(ggml_backend_t, void *) = nullptr;
    void (*tg_graph_free)(ggml_backend_t, void *)   = nullptr;
    // One recorded graph set per cgraph shape - a server alternates shapes,
    // and a single-slot cache would reset its warmup on every switch. Only
    // single-stage graphs whose closures are all AllReduce are captured, so
    // a shape records exactly one run: the whole token.
    struct tg_run {
        size_t              stage     = 0;
        size_t              i_beg     = 0;   // first subgraph in the run
        size_t              i_end     = 0;   // one past the last
        std::vector<void *> exec;            // one graph per lane of this stage
    };
    struct tg_entry {
        size_t              uid    = 0;
        int                 warm   = 0;      // tokens seen on this shape
        bool                failed = false;  // capture rejected: never retry
        // Subgraphs [0, covered) are captured in `runs`. Anything at or past
        // `covered` is dispatched per-subgraph after the replay. Only the
        // GGML_META_TG_LIMIT debug knob makes this less than n_subgraphs.
        size_t              covered = SIZE_MAX;
        std::vector<tg_run> runs;
    };
    std::vector<tg_entry> tg_cache;
    static const size_t   tg_cache_max = 8;

    // The VRAM scratch pool still grows on the first few tokens, and hipMalloc
    // is illegal inside a stream capture, so a shape must run normally a few
    // times before it can be recorded, or capture aborts inside
    // ggml_cuda_pool_leg::alloc.
    static const int      tg_warm_needed = 4;

    tg_entry * tg_lookup(size_t uid) {
        for (auto & e : tg_cache) {
            if (e.uid == uid) {
                return &e;
            }
        }
        if (tg_cache.size() >= tg_cache_max) {
            // Evict the oldest entry, releasing its graphs.
            tg_free_entry(tg_cache.front());
            tg_cache.erase(tg_cache.begin());
        }
        tg_cache.push_back(tg_entry{});
        tg_cache.back().uid = uid;
        return &tg_cache.back();
    }

    void tg_free_entry(tg_entry & e) {
        for (tg_run & r : e.runs) {
            if (tg_graph_free != nullptr) {
                for (size_t k = 0; k < r.exec.size(); k++) {
                    if (r.exec[k] != nullptr) {
                        tg_graph_free(backend_configs[r.stage * tps + k].backend, r.exec[k]);
                    }
                }
            }
            r.exec.clear();
        }
        e.runs.clear();
    }
    void *                               xfer_comm_ctx = nullptr;
    ggml_backend_comm_sendrecv_tensor_t  comm_sendrecv = nullptr;
    bool                                 xfer_comm_default = false;
    bool                                 xfer_comm_moe_large = false;
    size_t                               xfer_comm_moe_threshold = 1024 * 1024;
    bool                                 graph_has_moe_ops = false;

    // Optional dedicated-stream cross-backend copy. Both pointers are non-null if the
    // simple backend exposes the queue+drain pair via proc_address. Used by stage_transfer
    // to issue cross-stage copies on a side stream so they don't serialize behind compute
    // on the source main stream. Each (src_lane, dst_lane) pair takes one queue call per
    // boundary tensor and exactly one drain call after all queues. The drain host-syncs
    // the side stream because the cross-device GPU dependency from pp_copy_stream to
    // dst->main is not reliable on HIP under GPU_MAX_HW_QUEUES=8 (gpt-oss-120b SWA past
    // 4K tokens emitted '?' tokens with the previous all-event shape).
    ggml_backend_cpy_tensor_async_dedicated_queue_t cpy_async_dedicated_queue = nullptr;
    ggml_backend_cpy_tensor_async_dedicated_drain_t cpy_async_dedicated_drain = nullptr;

    // Debug flags cached at init from env vars. The graph_compute and stage_transfer
    // hot paths used to call getenv on every invocation (per-ubatch and per-transfer);
    // measured cost is negligible but caching here keeps the hot path tidy and reads
    // each env var exactly once per backend.
    bool dbg_part = false; // GGML_META_PART_DEBUG (only fires on partition rebuild)
    bool dbg_run  = false; // GGML_META_RUN_DEBUG  (per graph_compute)
    bool dbg_xfer = false; // GGML_META_XFER_DEBUG (per stage_transfer)
    bool dbg_chunk = false; // GGML_META_CHUNK_TRACE (entry/exit stamp per graph_compute)
    bool layer_seam_cost = true; // GGML_META_LAYER_SEAM_COST (default on)

    // Concurrent per-lane graph dispatch. On by default, GGML_META_PARALLEL_DISPATCH=0
    // restores the serial per-lane issue.
    //
    // The subgraph loop in graph_compute issues ggml_backend_graph_compute_async to each
    // lane in turn, so lane 0 is enqueued first and lane N-1 last. Nearly every subgraph
    // ends in an AllReduce that then waits for the last lane to arrive, so the per-lane
    // host issue cost converts directly into rank skew. Measured on 4 GPUs: the 4th lane
    // arrives 86.8 us late, ranks leave the collective within 0.64 us of each other, and
    // 74.9% of the AR kernel time is that wait. With 80 AllReduce subgraphs per token the
    // staircase is rebuilt 80 times. Issuing the lanes concurrently removes it.
    bool parallel_dispatch = false;
    ggml_backend_meta_lane_dispatcher dispatcher;

    // GGML_META_PROFILE=1: where a token's wall time actually goes.
    // ns_compute is the whole graph_compute (pure host enqueue), of which
    // ns_lanes is issuing the per-lane subgraphs and ns_close is the closure
    // (AllReduce or stage transfer). ns_sync is the later wait on the GPU.
    // host-bound shows up as ns_compute >> ns_sync.
    bool     prof            = false;
    uint64_t prof_calls      = 0;
    uint64_t prof_subgraphs  = 0;
    uint64_t prof_ns_compute = 0;
    uint64_t prof_ns_lanes   = 0;
    uint64_t prof_ns_close   = 0;
    uint64_t prof_ns_sync    = 0;
    uint64_t prof_ns_wall0   = 0;

    void prof_report() const {
        if (!prof || prof_calls == 0) {
            return;
        }
        const double wall = (double) (ggml_meta_prof_now() - prof_ns_wall0);
        const double c    = (double) prof_calls;
        fprintf(stderr,
            "[meta-prof] tps=%zu calls=%llu subgraphs/call=%.1f | per call: "
            "compute=%.1f us (lanes=%.1f close=%.1f other=%.1f)  sync=%.1f us | "
            "totals: compute=%.1f%% sync=%.1f%% of %.1f s wall\n",
            tps, (unsigned long long) prof_calls, prof_subgraphs / c,
            prof_ns_compute / c / 1e3,
            prof_ns_lanes   / c / 1e3,
            prof_ns_close   / c / 1e3,
            (double)(prof_ns_compute - prof_ns_lanes - prof_ns_close) / c / 1e3,
            prof_ns_sync    / c / 1e3,
            100.0 * prof_ns_compute / wall,
            100.0 * prof_ns_sync    / wall,
            wall / 1e9);
    }

    // Sync-fallback scratch for set_tensor_async on layouts the chunk-by-chunk path can't handle:
    // multi-segment splits, PARTIAL axis, and repacked buffers. Sequentially-arriving chunks
    // accumulate here (persistent buffers, never zero-filled), and a completed tensor is handed
    // to ONE worker thread that runs the sync set_tensor splice - so the caller can read the
    // next tensor from disk while the previous one splices, packs and uploads. Two slots give
    // a depth-2 pipeline; the worker is joined in free() and drained in synchronize().
    struct fallback_accum {
        const ggml_tensor *        tensor = nullptr;
        std::unique_ptr<uint8_t[]> buf;
        size_t                     cap    = 0;
        size_t                     filled = 0;
    };
    fallback_accum          accum[2];
    int                     accum_turn = 0;
    std::thread             accum_worker;
    std::mutex              accum_mutex;
    std::condition_variable accum_cv;
    fallback_accum *        accum_job  = nullptr;   // pending job, depth 1
    bool                    accum_stop = false;

    void accum_worker_loop() {
        std::unique_lock<std::mutex> lock(accum_mutex);
        for (;;) {
            accum_cv.wait(lock, [&] { return accum_job != nullptr || accum_stop; });
            if (accum_job == nullptr) {
                return;
            }
            fallback_accum * job = accum_job;
            lock.unlock();
            ggml_backend_tensor_set(const_cast<ggml_tensor *>(job->tensor), job->buf.get(), 0, ggml_nbytes(job->tensor));
            lock.lock();
            job->tensor = nullptr;
            job->filled = 0;
            accum_job   = nullptr;
            accum_cv.notify_all();
        }
    }
    // hand a completed slot to the worker; blocks while the previous job is still running
    void accum_submit(fallback_accum * slot) {
        std::unique_lock<std::mutex> lock(accum_mutex);
        if (!accum_worker.joinable()) {
            accum_worker = std::thread([this] { accum_worker_loop(); });
        }
        accum_cv.wait(lock, [&] { return accum_job == nullptr; });
        accum_job = slot;
        accum_cv.notify_all();
    }
    void accum_drain() {
        std::unique_lock<std::mutex> lock(accum_mutex);
        accum_cv.wait(lock, [&] { return accum_job == nullptr; });
    }
    void accum_shutdown() {
        {
            std::unique_lock<std::mutex> lock(accum_mutex);
            accum_cv.wait(lock, [&] { return accum_job == nullptr; });
            accum_stop = true;
            accum_cv.notify_all();
        }
        if (accum_worker.joinable()) {
            accum_worker.join();
        }
    }

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
        const bool copy_only = params != nullptr && strcmp(params, "copy-only") == 0;
        const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
        const size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        tps      = meta_dev_ctx->tps;
        n_stages = meta_dev_ctx->n_stages;
        GGML_ASSERT(tps > 0 && n_stages > 0 && tps * n_stages == n_devs);
        n_reduce_steps = std::ceil(std::log2(tps));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        // Per-stage comm_ctx: each stage covers [stage*tps, (stage+1)*tps) and runs its own
        // AllReduce. tps == 1 stages skip comm_init (no AR needed within a single GPU).
        // GGML_META_NO_COMM=1 forces the generic butterfly fallback (debug aid).
        ggml_backend_reg_t simple_reg = ggml_backend_dev_backend_reg(
            ggml_backend_get_device(simple_backends[0]));
        ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t)
            ggml_backend_reg_get_proc_address(simple_reg, "ggml_backend_comm_init");
        auto comm_set_pipeline_stages = (void (*)(void *, size_t))
            ggml_backend_reg_get_proc_address(simple_reg, "ggml_backend_comm_set_pipeline_stages");

        comm_ctxs.assign(n_stages, nullptr);
        const char * env_no_comm = getenv("GGML_META_NO_COMM");
        const bool   no_comm     = copy_only || (env_no_comm && atoi(env_no_comm) != 0);
        if (tps > 1 && !no_comm && comm_init != nullptr) {
            for (size_t s = 0; s < n_stages; s++) {
                comm_ctxs[s] = comm_init(simple_backends.data() + s * tps, tps);
                if (comm_ctxs[s] != nullptr && comm_set_pipeline_stages != nullptr) {
                    comm_set_pipeline_stages(comm_ctxs[s], n_stages);
                }
            }
        }
        // Pull the AR func pointer once if any stage has a comm_ctx.
        for (size_t s = 0; s < n_stages; s++) {
            if (comm_ctxs[s] != nullptr) {
                comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                    ggml_backend_reg_get_proc_address(simple_reg, "ggml_backend_comm_allreduce_tensor");
                GGML_ASSERT(comm_allreduce != nullptr);
                comm_ar_prepare = (int (*)(void *, struct ggml_tensor **))
                    ggml_backend_reg_get_proc_address(simple_reg, "ggml_backend_comm_allreduce_prepare");
                comm_ar_launch_rank = (void (*)(void *, int))
                    ggml_backend_reg_get_proc_address(simple_reg, "ggml_backend_comm_allreduce_launch_rank");
                tg_capture_begin = (bool (*)(ggml_backend_t))
                    ggml_backend_reg_get_proc_address(simple_reg, "ggml_backend_token_capture_begin");
                tg_capture_end = (void * (*)(ggml_backend_t))
                    ggml_backend_reg_get_proc_address(simple_reg, "ggml_backend_token_capture_end");
                tg_graph_launch = (void (*)(ggml_backend_t, void *))
                    ggml_backend_reg_get_proc_address(simple_reg, "ggml_backend_token_graph_launch");
                tg_graph_free = (void (*)(ggml_backend_t, void *))
                    ggml_backend_reg_get_proc_address(simple_reg, "ggml_backend_token_graph_free");
                break;
            }
        }

        // Optional RCCL/NCCL point-to-point transfer path for stage boundaries. On HIP
        // with GPU_MAX_HW_QUEUES > 4, cross-device stream waits on ordinary HIP events
        // are not reliable for the dedicated memcpy path. A global comm lets the backend
        // express stage transfers as send/recv operations. The CUDA backend stages each
        // source tensor before sending so the original compute buffer can be reused while
        // the P2P transfer is still in flight. On ROCm, the comm path is also faster for
        // wider stage transfers (tps >= 3); keep tps=2 on dedicated copies unless HWQ8
        // requires the comm path.
        const char * env_xfer_comm = getenv("GGML_META_XFER_RCCL");
        const bool   xfer_forced   = env_xfer_comm && atoi(env_xfer_comm) != 0;
        const bool   xfer_disabled = env_xfer_comm && atoi(env_xfer_comm) == 0;
        auto env_u64 = [](const char * name, uint64_t default_value) {
            const char * value = getenv(name);
            if (value == nullptr || value[0] == '\0') {
                return default_value;
            }

            char * end = nullptr;
            const unsigned long long parsed = strtoull(value, &end, 10);
            return end != value ? (uint64_t) parsed : default_value;
        };

        const char * env_hwq       = getenv("GPU_MAX_HW_QUEUES");
        const bool   hwq_gt4       = env_hwq && atoi(env_hwq) > 4;
        const bool   is_rocm       = strcmp(ggml_backend_reg_name(simple_reg), "ROCm") == 0;
        const bool   rocm_prefer_xfer_comm = is_rocm && tps >= 3;
        xfer_comm_default = xfer_forced || hwq_gt4 || rocm_prefer_xfer_comm;
        xfer_comm_moe_large = is_rocm && tps == 2;
        xfer_comm_moe_threshold = env_u64("GGML_META_MOE_XFER_RCCL_THRESHOLD", 1024 * 1024);
        const bool   want_xfer_comm = n_stages > 1 && !no_comm && !xfer_disabled &&
                                      (xfer_comm_default || xfer_comm_moe_large);
        if (want_xfer_comm && comm_init != nullptr) {
            comm_sendrecv = (ggml_backend_comm_sendrecv_tensor_t)
                ggml_backend_reg_get_proc_address(simple_reg, "ggml_backend_comm_sendrecv_tensor");
            if (comm_sendrecv != nullptr) {
                xfer_comm_ctx = comm_init(simple_backends.data(), n_devs);
                // Size the staging ring from the pipeline depth: one buffer per
                // stage plus one in flight. The backend default is the
                // compile-time maximum, which makes a shallow pipeline pay a
                // deep one's staging footprint.
                if (xfer_comm_ctx != nullptr) {
                    auto set_depth = (void (*)(void *, size_t))
                        ggml_backend_reg_get_proc_address(simple_reg, "ggml_backend_comm_set_staging_depth");
                    if (set_depth != nullptr) {
                        set_depth(xfer_comm_ctx, n_stages + 1);
                    }
                }
            }
        }

        // Look up the dedicated-stream copy queue+drain pair (optional). When available,
        // stage_transfer uses them to avoid host-blocking and main-stream serialization on
        // cross-stage copies. Both must be present together; if the backend only exposes one
        // (older build), fall back to the sync path. GGML_META_NO_DEDICATED_CPY=1 also forces
        // the sync-copy fallback (debug aid for A/B).
        const char * env_no_ded = getenv("GGML_META_NO_DEDICATED_CPY");
        const bool   no_ded     = env_no_ded && atoi(env_no_ded) != 0;
        if (n_stages > 1 && !no_ded) {
            cpy_async_dedicated_queue = (ggml_backend_cpy_tensor_async_dedicated_queue_t)
                ggml_backend_reg_get_proc_address(simple_reg, "ggml_backend_cpy_tensor_async_dedicated_queue");
            cpy_async_dedicated_drain = (ggml_backend_cpy_tensor_async_dedicated_drain_t)
                ggml_backend_reg_get_proc_address(simple_reg, "ggml_backend_cpy_tensor_async_dedicated_drain");
            if (cpy_async_dedicated_queue == nullptr || cpy_async_dedicated_drain == nullptr) {
                cpy_async_dedicated_queue = nullptr;
                cpy_async_dedicated_drain = nullptr;
            }
        }

        auto env_flag = [](const char * name) {
            const char * v = getenv(name);
            return v && atoi(v) != 0;
        };
        // For flags that default on, so that only an explicit 0 turns them off.
        auto env_flag_on = [](const char * name) {
            const char * v = getenv(name);
            return v == nullptr || atoi(v) != 0;
        };
        dbg_part = env_flag("GGML_META_PART_DEBUG");
        dbg_run  = env_flag("GGML_META_RUN_DEBUG");
        dbg_xfer = env_flag("GGML_META_XFER_DEBUG");
        dbg_chunk = env_flag("GGML_META_CHUNK_TRACE");
        layer_seam_cost = env_flag_on("GGML_META_LAYER_SEAM_COST");

        // On by default, and only does anything with more than one lane per stage.
        //
        // Restricted to CUDA/ROCm sub-backends on purpose. The dispatcher itself is
        // portable C++11, but the SAFETY argument is not: it relies on the current device
        // being thread-local (cudaGetDevice) and on each lane owning its own context,
        // streams, graph cache and pool. This meta backend wraps arbitrary
        // ggml_backend_dev_t, and concurrent graph_compute has not been verified for
        // Vulkan / SYCL / Metal / CPU sub-backends. Refuse rather than assume.
        prof = env_flag("GGML_META_PROFILE");
        if (prof) {
            prof_ns_wall0 = ggml_meta_prof_now();
        }
        parallel_dispatch = env_flag_on("GGML_META_PARALLEL_DISPATCH") && tps > 1;
        // The recording loop issues each rank itself, on one thread, so this needs
        // the split AllReduce entry points and the concurrent lane dispatch.
        token_graph = env_flag_on("GGML_META_TOKEN_GRAPH") && parallel_dispatch;
        if (parallel_dispatch) {
            for (size_t i = 0; i < n_devs; i++) {
                ggml_backend_dev_t d = ggml_backend_meta_dev_simple_dev(meta_dev, i);
                ggml_backend_reg_t r = d != nullptr ? ggml_backend_dev_backend_reg(d) : nullptr;
                const char * rn = r != nullptr ? ggml_backend_reg_name(r) : nullptr;
                const bool ok = rn != nullptr &&
                                (strcmp(rn, "CUDA") == 0 || strcmp(rn, "ROCm") == 0);
                if (!ok) {
                    GGML_LOG_DEBUG("%s: serial lane dispatch, concurrent issue is "
                                   "unverified for backend '%s'\n",
                                   __func__, rn != nullptr ? rn : "(unknown)");
                    parallel_dispatch = false;
                    break;
                }
            }
        }
        if (parallel_dispatch) {
            dispatcher.ensure(tps - 1);
        }
    }

    ~ggml_backend_meta_context() {
        prof_report();

        // Captured graph executables are owned by their simple backends. Release
        // them while those backends (and their CUDA/HIP contexts) are still alive.
        for (auto & entry : tg_cache) {
            tg_free_entry(entry);
        }
        tg_cache.clear();

        ggml_backend_comm_free_t comm_free = nullptr;
        if (xfer_comm_ctx != nullptr) {
            comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != nullptr);
            comm_free(xfer_comm_ctx);
        }
        for (size_t s = 0; s < comm_ctxs.size(); s++) {
            if (comm_ctxs[s] == nullptr) continue;
            if (comm_free == nullptr) {
                comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                    ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
                GGML_ASSERT(comm_free != nullptr);
            }
            comm_free(comm_ctxs[s]);
        }
        for (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    backend_ctx->accum_shutdown();
    delete backend_ctx;
    delete backend;
}

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    if (size == 0) {
        return;
    }

    if (ggml_backend_meta_buft_is_repack(ggml_backend_buffer_get_type(tensor->buffer))) {
        // Forwarding raw chunks to the sync path asserts in the splitter
        // (partial writes cannot be spliced). Accumulate and dispatch whole
        // tensors through the worker instead.
        ggml_backend_meta_context * be_ctx = (ggml_backend_meta_context *) backend->context;
        const size_t total = ggml_nbytes(tensor);
        if (offset == 0 && size == total) {
            be_ctx->accum_drain();   // keep tensor order for the lane buffers
            ggml_backend_tensor_set(tensor, data, 0, size);
            return;
        }
        auto & acc = be_ctx->accum[be_ctx->accum_turn];
        if (acc.tensor != tensor) {
            GGML_ASSERT(acc.tensor == nullptr && "meta accum: slot busy on tensor switch");
            if (acc.cap < total) {
                acc.buf.reset(new uint8_t[total]);   // default-init, no zero fill
                acc.cap = total;
            }
            acc.tensor = tensor;
            acc.filled = 0;
        }
        GGML_ASSERT(offset + size <= total);
        memcpy(acc.buf.get() + offset, data, size);
        acc.filled += size;
        if (acc.filled == total) {
            be_ctx->accum_submit(&acc);
            be_ctx->accum_turn ^= 1;
            // make sure the next slot is free before the caller reuses it
            if (be_ctx->accum[be_ctx->accum_turn].tensor != nullptr) {
                be_ctx->accum_drain();
            }
        }
        return;
    }

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);

    // Multi-segment, nr-broadcast, and PARTIAL cannot be dispatched chunk-by-chunk. Pass the whole
    // tensor through if we already have it, otherwise buffer sequentially-arriving chunks until the
    // last byte, then dispatch via the sync set_tensor path which handles those layouts.
    if (split_state.n_segments != 1 || split_state.nr[0] != 1 || split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
        const size_t total = ggml_nbytes(tensor);
        if (offset == 0 && size == total) {
            ggml_backend_tensor_set(tensor, data, 0, size);
            return;
        }
        ggml_backend_meta_context * be_ctx = (ggml_backend_meta_context *) backend->context;
        auto & acc = be_ctx->accum[be_ctx->accum_turn];
        if (acc.tensor != tensor) {
            GGML_ASSERT(acc.tensor == nullptr && "meta accum: slot busy on tensor switch");
            if (acc.cap < total) {
                acc.buf.reset(new uint8_t[total]);   // default-init, no zero fill
                acc.cap = total;
            }
            acc.tensor = tensor;
            acc.filled = 0;
        }
        GGML_ASSERT(offset + size <= total);
        memcpy(acc.buf.get() + offset, data, size);
        acc.filled += size;
        if (acc.filled == total) {
            be_ctx->accum_submit(&acc);
            be_ctx->accum_turn ^= 1;
            if (be_ctx->accum[be_ctx->accum_turn].tensor != nullptr) {
                be_ctx->accum_drain();
            }
        }
        return;
    }

    // The fallback above intercepts every multi-segment / PARTIAL layout; whatever reaches the
    // chunk-by-chunk dispatch below is a plain single-segment, unrepeated split.
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];

            // Per-device dispatch for a single row's column range [col_off, col_off + len).
            // Used for the unaligned head/tail when the chunk doesn't land on row boundaries.
            // Zero-size device slices are skipped (upstream zero-sized-slice TP fix).
            auto write_partial_row = [&](int64_t R, size_t col_off, size_t len, const char * src) {
                size_t col_start_j = 0;
                for (size_t j = 0; j < n_backends; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                    if (chunk_size_j == 0) {
                        continue;
                    }
                    const size_t col_end_j    = col_start_j + chunk_size_j;
                    const size_t s = std::max(col_start_j, col_off);
                    const size_t e = std::min(col_end_j,   col_off + len);
                    if (s < e) {
                        ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                        const size_t dst_off  = (size_t) R * chunk_size_j + (s - col_start_j);
                        const size_t src_off  = s - col_off;
                        const size_t copy_len = e - s;
                        ggml_backend_tensor_set_async(simple_backend, simple_tensor, src + src_off, dst_off, copy_len);
                    }
                    col_start_j = col_end_j;
                }
            };

            const char * src       = (const char *) data;
            size_t       pos       = offset;
            size_t       remaining = size;

            // Partial head: bytes from current pos up to the next row boundary, capped by remaining.
            if ((pos % chunk_size_full) != 0) {
                const int64_t R       = (int64_t) (pos / chunk_size_full);
                const size_t  col_off = pos % chunk_size_full;
                const size_t  len     = std::min(chunk_size_full - col_off, remaining);
                write_partial_row(R, col_off, len, src);
                src       += len;
                pos       += len;
                remaining -= len;
            }

            // Aligned middle: full rows. One strided 2D async dispatch per device.
            const int64_t i_first = (int64_t) (pos / chunk_size_full);
            const int64_t n_rows  = (int64_t) (remaining / chunk_size_full);
            if (n_rows > 0) {
                size_t offset_j = 0;
                for (size_t j = 0; j < n_backends; j++) {
                    ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                    if (chunk_size_j == 0) {
                        continue;
                    }
                    ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, src + offset_j,
                        i_first * chunk_size_j, chunk_size_j,
                        n_rows, chunk_size_j, chunk_size_full);
                    offset_j += chunk_size_j;
                }
                GGML_ASSERT(offset_j == chunk_size_full);
                const size_t consumed = (size_t) n_rows * chunk_size_full;
                src       += consumed;
                pos       += consumed;
                remaining -= consumed;
            }

            // Partial tail: remaining bytes starting at column 0 of the current row.
            if (remaining > 0) {
                GGML_ASSERT((pos % chunk_size_full) == 0);
                const int64_t R = (int64_t) (pos / chunk_size_full);
                write_partial_row(R, 0, remaining, src);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error: meta set_tensor_async unhandled split axis %d", (int) split_state.axis);
        }
    }
}

static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (char *) data + offset_j,
                    i_start * chunk_size_j, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // MIRRORED means every lane OF THE OWNING STAGE holds the value, not every
            // lane of the backend. With n_stages > 1 a fixed index can name a lane on a
            // stage that never computed this tensor, whose buffer is then untouched and
            // reads back as zeros. Prefer a lane in the stage that produced it.
            // Single stage keeps the original lane: every lane holds the value, and the
            // last one is the natural choice since it is not the lane the calling thread
            // drives under parallel dispatch. Only redirect when stages make it wrong.
            auto * meta_ctx = (ggml_backend_meta_context *) backend->context;
            size_t simple_backend_idx = n_backends - 1;
            if (meta_ctx->n_stages > 1 && meta_ctx->tps > 0) {
                const auto it = meta_ctx->tensor_stage.find(tensor);
                if (it != meta_ctx->tensor_stage.end()) {
                    // Last lane of the owning stage, mirroring the single-stage choice.
                    const size_t lane = (it->second + 1) * meta_ctx->tps - 1;
                    if (lane < n_backends) {
                        simple_backend_idx = lane;
                    }
                }
            }
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, simple_backend_idx);
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, simple_backend_idx);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend) {
    ((ggml_backend_meta_context *) backend->context)->accum_drain();
    ggml_backend_meta_context * sync_ctx = (ggml_backend_meta_context *) backend->context;
    if (sync_ctx->dbg_chunk) {
        const double t0 = ggml_time_us()/1000.0;
        const size_t n = ggml_backend_meta_n_backends(backend);
        for (size_t i = 0; i < n; i++) {
            ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
        }
        const double waited = ggml_time_us()/1000.0 - t0;
        fprintf(stderr, "[chunk] sync  t=%9.3f ms waited=%.3f ms\n", t0, waited);
#if defined(__linux__)
        // name the caller of any expensive drain - every extern frame
        // symbolizes, which is enough to attribute the wait
        if (waited > 50.0) {
            void * frames[16];
            const int n_frames = backtrace(frames, 16);
            backtrace_symbols_fd(frames, n_frames, fileno(stderr));
        }
#endif
        return;
    }
    ggml_backend_meta_context * prof_ctx = (ggml_backend_meta_context *) backend->context;
    ggml_meta_prof_scope prof_guard(&prof_ctx->prof_ns_sync, prof_ctx->prof);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

// Entry/exit wall stamps around graph_compute. The exit fires on every return
// path via the destructor, so host gaps BETWEEN graphs and enqueue windows
// WITHIN a graph can be read off one trace. Diagnostic for cross-chunk stage
// overlap: a gap between one graph's exit and the next one's enter is host
// blocking above the backend, a contiguous trace with a long wall tail is
// GPU-side serialization below it.
struct ggml_meta_chunk_trace_guard {
    bool on = false;
    ggml_meta_chunk_trace_guard(bool on_, const struct ggml_cgraph * cgraph) : on(on_) {
        if (on) {
            fprintf(stderr, "[chunk] enter t=%9.3f ms n_nodes=%d uid=%llu\n",
                    ggml_time_us()/1000.0, cgraph->n_nodes, (unsigned long long) cgraph->uid);
        }
    }
    ~ggml_meta_chunk_trace_guard() {
        if (on) {
            fprintf(stderr, "[chunk] exit  t=%9.3f ms\n", ggml_time_us()/1000.0);
        }
    }
};

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == nullptr);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    ggml_meta_chunk_trace_guard chunk_guard(backend_ctx->dbg_chunk, cgraph);
    ggml_meta_prof_scope prof_guard(&backend_ctx->prof_ns_compute, backend_ctx->prof);
    backend_ctx->prof_calls += backend_ctx->prof ? 1 : 0;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    const bool needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    // uid == 0 marks a graph view from the scheduler's eval-callback path: a subrange
    // of one logical graph. Whole graphs always carry a uid, so this selects the
    // fragment-safe handling (persistent stage state, cross-fragment transfers).
    const bool fragment = cgraph->uid == 0;

    bool max_nnodes_raised = false;
    if (cgraph->n_nodes > backend_ctx->max_nnodes) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = true;
        assert(needs_rebuild);
    }

    if (needs_rebuild) {
        std::set<ggml_backend_buffer_t> used_buffers;
        for (int i = 0; i < cgraph->n_leafs; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->leafs[i]->buffer)) {
                used_buffers.emplace(cgraph->leafs[i]->buffer);
            }
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->nodes[i]->buffer)) {
                used_buffers.emplace(cgraph->nodes[i]->buffer);
            }
        }
        for (ggml_backend_buffer_t buf : used_buffers) {
            ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;
            buf_ctx->stc_compute_index_next = buf_ctx->stc_compute_index ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_index_next];
            for (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;
        backend_ctx->graph_has_moe_ops = false;

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->op == GGML_OP_MUL_MAT_ID || node->op == GGML_OP_ADD_ID) {
                    backend_ctx->graph_has_moe_ops = true;
                }
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    // FIXME s_copy_main is on the CPU and its view seems to be incorrectly added to the graph nodes.
                    // For regular usage this doesn't matter since it's a noop but trying to call ggml_backend_meta_buffer_simple_tensor results in a crash.
                    bcj.nodes[i] = node;
                    continue;
                }
                bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(node, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed_branch = [&](const int i) -> int {
                int id = i; // i_delayed
                int idr = i; // i_delayed return, last safe return value

                ggml_tensor * node = cgraph->nodes[id];
                int32_t n_used = ggml_node_get_use_count(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    while (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            break;
                        }
                        bool safe = true;
                        for (int s = 0; s < GGML_MAX_SRC; s++) {
                            if (next->src[s] == nullptr) {
                                continue;
                            }
                            if (next->src[s] == node) {
                                safe = false;
                                break;
                            }
                            if (ggml_backend_meta_get_split_state(next->src[s], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = false;
                                break;
                            }
                        }
                        if (!safe) {
                            break;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                if (id + 1 >= cgraph->n_nodes) {
                    return idr;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_ADD_ID && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_get_split_state(next->src[2], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                while (true) {
                    skip_unrelated();
                    if (id + 1 >= cgraph->n_nodes) {
                        return idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_MUL && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    } else {
                        break;
                    }
                }

                if (n_used != node->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    return idr;
                }
                for (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_VIEW || next->view_src != node || next->view_offs != k*node->nb[1] ||
                            next->ne[0] != node->ne[0] || next->ne[1] != node->ne[2] || next->nb[1] != node->nb[2] ||
                            ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                for (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                idr = id;
                return idr;
            };

            // AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial branches.
            // This wrapper and the call-site fold below both collapse that pair. Measured
            // equivalent on deepseek4 (88.0 subgraphs per token either way), and they are kept
            // together because this one needs the ADD adjacent while the call-site fold walks
            // to a later consumer, which is the shape a split shared expert produces.
            auto get_i_delayed = [&](const int i) -> int {
                const int i_delayed = get_i_delayed_branch(i);
                ggml_tensor * node = cgraph->nodes[i_delayed];

                if (ggml_node_get_use_count(cgraph, i_delayed) != 1) {
                    return i_delayed;
                }

                for (int id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * next = cgraph->nodes[id];
                    if (next->view_src == node) {
                        return i_delayed;
                    }
                    for (int s = 0; s < GGML_MAX_SRC; s++) {
                        if (next->src[s] == node) {
                            return i_delayed;
                        }
                    }

                    if (next->view_src != nullptr && next->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(next->view_src->buffer)) {
                        continue;
                    }
                    if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                        continue;
                    }

                    const int i_other = id;
                    const int i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    if (ggml_node_get_use_count(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        return i_delayed;
                    }

                    ggml_tensor * sum = cgraph->nodes[i_other_delayed + 1];
                    if (sum->op != GGML_OP_ADD ||
                            !ggml_are_same_shape(node, other) || node->type != other->type || sum->type != node->type ||
                            !((sum->src[0] == node && sum->src[1] == other) ||
                              (sum->src[0] == other && sum->src[1] == node)) ||
                            ggml_backend_meta_get_split_state(sum, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        return i_delayed;
                    }

                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        const bool compute       = bcj.nodes[i]->flags       & GGML_TENSOR_FLAG_COMPUTE;
                        const bool compute_other = bcj.nodes[i_other]->flags & GGML_TENSOR_FLAG_COMPUTE;
                        if (compute != compute_other) {
                            return i_delayed;
                        }
                    }
                    return i_other_delayed + 1;
                }
                return i_delayed;
            };

            // Determine the owning stage of a node by looking at which lanes have non-zero ne[]
            // in its inferred split_state. -1 means "no per-lane info" (MIRRORED, PARTIAL,
            // empty NONE), which the caller resolves by inheriting the previous active stage.
            // TODO: ggml_backend_meta_get_split_state is recursive over src tensors and gets
            // called once per cgraph node here plus again in the xfer_set build below. For
            // very large graphs (gpt-oss-120b) this rebuild-time cost adds up. A memoization
            // layer keyed on (tensor*, assume_sync) would amortize it, but the partition path
            // only runs on cgraph shape change so this is microsecond-level on the steady
            // state and not a priority.
            auto node_owning_stage = [&](const ggml_tensor * node) -> int {
                const ggml_backend_meta_split_state ss = ggml_backend_meta_get_split_state(node, /*assume_sync =*/ true);
                if (ss.axis < 0 || ss.axis >= GGML_MAX_DIMS) {
                    return -1;
                }
                for (size_t j = 0; j < n_backends; j++) {
                    int64_t sum = 0;
                    for (size_t s = 0; s < ss.n_segments; s++) {
                        sum += ss.ne[s * n_backends + j];
                    }
                    if (sum > 0) {
                        return (int)(j / backend_ctx->tps);
                    }
                }
                return -1;
            };

            backend_ctx->subgraphs.clear();
            if (!fragment) {
                // Fragments accumulate into the map instead: the readback lane choice
                // in get_tensor needs entries from earlier fragments of the same graph.
                backend_ctx->tensor_stage.clear();
            }

            // Persistent-buffer (KV cache) write and read indices, needed so the
            // stage seam can avoid splitting a layer between its cache writes and
            // the reads of those same rows. Without this the seam can land
            // mid-attention and the whole cache layer has to be relayed.
            auto view_root = [](const ggml_tensor * t) -> const ggml_tensor * {
                while (t != nullptr && t->view_src != nullptr) {
                    t = t->view_src;
                }
                return t;
            };
            std::unordered_map<const ggml_tensor *, int> persist_first_write;
            std::unordered_map<const ggml_tensor *, int> persist_last_read;
            if (backend_ctx->n_stages > 1) {
                for (int i = 0; i < cgraph->n_nodes; i++) {
                    ggml_tensor * nd = cgraph->nodes[i];
                    if (nd->view_src != nullptr) {
                        const ggml_tensor * rt = view_root(nd);
                        if (rt != nullptr && rt->buffer != nullptr &&
                                ggml_backend_buffer_get_usage(rt->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
                            auto it = persist_first_write.find(rt);
                            if (it == persist_first_write.end()) persist_first_write[rt] = i;
                        }
                    }
                    for (int sx = 0; sx < GGML_MAX_SRC; sx++) {
                        const ggml_tensor * s = nd->src[sx];
                        if (s == nullptr) continue;
                        const ggml_tensor * rt = view_root(s);
                        if (rt == nullptr || rt->buffer == nullptr) continue;
                        if (ggml_backend_buffer_get_usage(rt->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) continue;
                        if (ggml_backend_buffer_get_usage(rt->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) continue;
                        persist_last_read[rt] = i;
                    }
                }
            }

            // Build the dependency index before choosing stage boundaries. The same
            // collector is used both to price candidate seams and, after partitioning,
            // to populate the transfer sets. Keeping one implementation guarantees that
            // the cost comparison includes persistent-buffer dependencies exactly as the
            // execution path will transfer them.
            std::unordered_map<const ggml_tensor *, int> xfer_node_index;
            xfer_node_index.reserve((size_t) cgraph->n_nodes * 2);
            std::unordered_map<const ggml_tensor *, std::vector<int>> xfer_buffer_writers;
            for (int ii = 0; ii < cgraph->n_nodes; ii++) {
                ggml_tensor * nd = cgraph->nodes[ii];
                xfer_node_index[nd] = ii;
                if (nd->view_src == nullptr) {
                    continue;
                }
                const ggml_tensor * root = view_root(nd);
                if (root == nullptr || root->buffer == nullptr ||
                        ggml_backend_buffer_get_usage(root->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
                    continue;
                }
                xfer_buffer_writers[root].push_back(ii);
            }

            auto collect_xfer_set = [&](int boundary_idx) {
                std::vector<ggml_tensor *> result;
                std::unordered_set<ggml_tensor *> seen;
                for (int k = boundary_idx; k < cgraph->n_nodes; k++) {
                    ggml_tensor * node = cgraph->nodes[k];
                    for (int sx = 0; sx < GGML_MAX_SRC; sx++) {
                        ggml_tensor * src = node->src[sx];
                        if (src == nullptr) {
                            continue;
                        }
                        const auto it = xfer_node_index.find(src);
                        if (it == xfer_node_index.end()) {
                            const ggml_tensor * root = view_root(src);
                            const auto wit = xfer_buffer_writers.find(root);
                            if (wit == xfer_buffer_writers.end()) {
                                continue;
                            }
                            for (int widx : wit->second) {
                                if (widx >= boundary_idx) {
                                    continue;
                                }
                                ggml_tensor * wnode = cgraph->nodes[widx];
                                if (!seen.insert(wnode).second) {
                                    continue;
                                }
                                const ggml_backend_meta_split_state wss =
                                    ggml_backend_meta_get_split_state(wnode, /*assume_sync =*/ true);
                                if (wss.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                    result.push_back(wnode);
                                }
                            }
                            continue;
                        }
                        if (it->second >= boundary_idx || !seen.insert(src).second) {
                            continue;
                        }
                        const ggml_backend_meta_split_state ss =
                            ggml_backend_meta_get_split_state(src, /*assume_sync =*/ true);
                        if (ss.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            result.push_back(src);
                        }
                    }
                }
                return result;
            };

            // Derive the transfer set for an earlier boundary from the already-built
            // later-boundary set. collect_xfer_set scans future consumers in order, so
            // scanning only [early, late) and then appending the filtered later set
            // produces the same order and membership as a full collect at early.
            auto derive_earlier_xfer_set = [&](const std::vector<ggml_tensor *> & later_set,
                                               int early, int late) {
                std::vector<ggml_tensor *> result;
                std::unordered_set<ggml_tensor *> seen;
                auto include = [&](ggml_tensor * tensor) {
                    const auto it = xfer_node_index.find(tensor);
                    if (it == xfer_node_index.end() || it->second >= early ||
                            !seen.insert(tensor).second) {
                        return;
                    }
                    const ggml_backend_meta_split_state ss =
                        ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ true);
                    if (ss.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        result.push_back(tensor);
                    }
                };
                for (int k = early; k < late; k++) {
                    ggml_tensor * node = cgraph->nodes[k];
                    for (int sx = 0; sx < GGML_MAX_SRC; sx++) {
                        ggml_tensor * src = node->src[sx];
                        if (src == nullptr) {
                            continue;
                        }
                        const auto it = xfer_node_index.find(src);
                        if (it != xfer_node_index.end()) {
                            include(src);
                            continue;
                        }
                        const ggml_tensor * root = view_root(src);
                        const auto wit = xfer_buffer_writers.find(root);
                        if (wit == xfer_buffer_writers.end()) {
                            continue;
                        }
                        for (int widx : wit->second) {
                            if (widx >= early) {
                                continue;
                            }
                            include(cgraph->nodes[widx]);
                        }
                    }
                }
                for (ggml_tensor * tensor : later_set) {
                    include(tensor);
                }
                return result;
            };

            std::unordered_map<int, std::vector<ggml_tensor *>> planned_xfer_sets;

            // A graph with no owned node anywhere cannot derive a stage from its
            // own contents, so every node inherits the walk's initial stage and any
            // write lands there - which is wrong when the buffer belongs elsewhere.
            // Only such a graph may fall back to the recorded buffer owner; a graph
            // that owns nodes already places itself and must not be disturbed.
            bool graph_has_owner = false;
            if (backend_ctx->n_stages > 1) {
                for (int i = 0; i < cgraph->n_nodes; i++) {
                    if (node_owning_stage(cgraph->nodes[i]) >= 0) {
                        graph_has_owner = true;
                        break;
                    }
                }
            }

            int i_start = 0;
            // Fragments inherit the active stage from the previous fragment.
            int current_stage = fragment ? backend_ctx->frag_last_stage : 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    continue;
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(node, /*assume_sync =*/ false);
                if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(node));
                }

                // Stage detection. A node whose inferred split_state has a clear per-lane
                // owning stage (i.e., non-zero ne[] only inside one stage's lane range)
                // determines the active stage going forward. MIRRORED/PARTIAL inherit.
                int n_stage = node_owning_stage(node);
                if (n_stage < 0 && !graph_has_owner && backend_ctx->n_stages > 1 &&
                        !backend_ctx->persist_buffer_stage.empty()) {
                    // No per-lane information, but if this node writes a persistent
                    // buffer whose owning stage is already known, that stage owns the
                    // write too - otherwise it lands on a copy its reader never sees.
                    const ggml_tensor * wr = node;
                    while (wr != nullptr && wr->view_src != nullptr) {
                        wr = wr->view_src;
                    }
                    if (wr != nullptr) {
                        const auto it_b = backend_ctx->persist_buffer_stage.find(wr);
                        if (it_b != backend_ctx->persist_buffer_stage.end()) {
                            n_stage = (int) it_b->second;
                        }
                    }
                }
                bool stage_transition = (backend_ctx->n_stages > 1 && n_stage >= 0 && n_stage != current_stage && i > i_start);
                if (stage_transition) {
                    // Would this seam split a persistent buffer - writes before it,
                    // reads after it? If so the whole buffer would have to be
                    // relayed. Rewind the seam to before its first write so the
                    // writes land on the same stage as the reads.
                    // Fixed-point: moving the seam can make a DIFFERENT persistent
                    // buffer straddle it (measured: cache_k snapped the seam to its
                    // first write while the indexer state's compress write sat one
                    // node earlier, so the state buffer straddled the moved seam and
                    // the relay clobbered the new stage's freshly written state).
                    // Rewind until no buffer has a write before and a read after the
                    // evolving seam position.
                    auto rewind_persistent = [&](int boundary) {
                        for (bool changed = true; changed; ) {
                            changed = false;
                            for (const auto & kv : persist_first_write) {
                                const int w = kv.second;
                                if (w < i_start || w >= boundary) continue;
                                const auto it_r = persist_last_read.find(kv.first);
                                if (it_r == persist_last_read.end() || it_r->second < boundary) continue;
                                if (w < boundary) { boundary = w; changed = true; }
                            }
                        }
                        return boundary;
                    };

                    int snap_to = rewind_persistent(i);

                    // Hyperconnection graphs can expose the next stage only after a
                    // substantial prefix of the next layer. The layer input is marked by
                    // l_last-N at the start of the block. Price that true layer seam using
                    // the exact transfer-set collector and select it only when it reduces
                    // bytes (then operation count). Decode shapes therefore remain on the
                    // established boundary whenever moving the seam would relay HCA state.
                    const ggml_tensor * first = cgraph->nodes[i_start];
                    // Transfer bytes are the dominant cost for prompt-sized graphs.
                    // Decode and speculative-verify graphs are too narrow for that
                    // metric alone: moving their seam can save a small copy while
                    // worsening stage balance. Keep their established boundary.
                    if (backend_ctx->layer_seam_cost && ggml_nrows(first) > 16 && i_start + 1 < i &&
                            std::strncmp(first->name, "l_last-", 7) == 0) {
                        const int candidate = rewind_persistent(i_start + 1);
                        if (candidate < snap_to) {
                            const int current = snap_to;
                            auto current_set   = collect_xfer_set(current);
                            auto candidate_set = derive_earlier_xfer_set(current_set, candidate, current);
                            auto xfer_cost = [](const std::vector<ggml_tensor *> & xfer) {
                                size_t bytes = 0;
                                for (const ggml_tensor * tensor : xfer) {
                                    bytes += ggml_nbytes(tensor);
                                }
                                return std::make_pair(bytes, xfer.size());
                            };
                            const auto current_cost   = xfer_cost(current_set);
                            const auto candidate_cost = xfer_cost(candidate_set);
                            if (candidate_cost < current_cost) {
                                snap_to = candidate;
                                planned_xfer_sets.emplace(candidate, std::move(candidate_set));
                            } else {
                                planned_xfer_sets.emplace(current, std::move(current_set));
                            }
                            if (backend_ctx->dbg_part) {
                                fprintf(stderr,
                                        "[meta-seam] uid=%zu rows=%lld stage=%d->%d current=%d bytes=%zu ops=%zu "
                                        "candidate=%d bytes=%zu ops=%zu selected=%d\n",
                                        (size_t) cgraph->uid, (long long) ggml_nrows(first), current_stage, n_stage,
                                        current, current_cost.first, current_cost.second,
                                        candidate, candidate_cost.first, candidate_cost.second, snap_to);
                            }
                        }
                    }
                    if (snap_to >= i_start && snap_to < i) {
                        // re-run from the snap point under the new stage. snap_to ==
                        // i_start moves the whole block: the closed subgraph is then
                        // empty and only carries the TRANSFER closure whose xfer set
                        // (built later against this boundary) ships the pre-boundary
                        // deps to the new stage.
                        i = snap_to - 1;
                        stage_transition = false;
                        for (size_t j = 0; j < n_backends; j++) {
                            auto & bcj = backend_ctx->backend_configs[j];
                            bcj.cgraphs[n_subgraphs].offset = i_start;
                        }
                        backend_ctx->subgraphs.push_back({ (size_t) current_stage,
                                                           ggml_backend_meta_context::subgraph_closure::TRANSFER, {} });
                        n_subgraphs++;
                        i_start = snap_to;
                        current_stage = n_stage;
                        continue;
                    }
                }
                if (stage_transition) {
                    // Close the previous subgraph at [i_start, i-1] with TRANSFER closure so
                    // the boundary tensor (this subgraph's last node) gets broadcast to the
                    // new stage's lanes before we run the new stage's first node.
                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        bcj.cgraphs[n_subgraphs].offset = i_start;
                    }
                    backend_ctx->subgraphs.push_back({ (size_t) current_stage,
                                                       ggml_backend_meta_context::subgraph_closure::TRANSFER, {} });
                    n_subgraphs++;
                    i_start = i;
                    current_stage = n_stage;
                } else if (n_stage >= 0) {
                    current_stage = n_stage;
                }

                // Record the owning stage so a later MIRRORED read can pick a lane that
                // actually holds the data instead of a fixed index.
                backend_ctx->tensor_stage[node] = (size_t) current_stage;

                // Remember which stage owns each persistent buffer this node touches,
                // so a later graph that only writes it can be placed correctly.
                if (backend_ctx->n_stages > 1) {
                    auto proot = [](const ggml_tensor * t) -> const ggml_tensor * {
                        while (t != nullptr && t->view_src != nullptr) t = t->view_src;
                        return t;
                    };
                    const ggml_tensor * wr = proot(node);
                    if (wr != nullptr && wr->buffer != nullptr &&
                            ggml_backend_buffer_get_usage(wr->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                            ggml_backend_buffer_get_usage(wr->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        backend_ctx->persist_buffer_stage[wr] = (size_t) current_stage;
                    }
                }

                const bool ar_close  = (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL);
                // TOP_K selections must be uniform across the stage's lanes: the
                // mirrored score compute is not bitwise identical (per-lane reduce
                // ordering), and once the selection is live a near-tie flip gives
                // every lane a different mask. Close the subgraph and broadcast the
                // first lane's result. GGML_META_TOPK_BCAST=0 disables.
                static const bool topk_bcast = []() {
                    const char * env = getenv("GGML_META_TOPK_BCAST");
                    return env == nullptr || atoi(env) != 0;
                }();
                // A TOP_K whose scores came from the fused lightning indexer
                // needs no broadcast: the scores are AllReduce outputs, which
                // are bit-identical on every lane by construction, so the
                // per-lane bitonic selection already agrees. Only the unfused
                // path (per-lane mirrored score compute, lane-local reduce
                // ordering) can disagree on near-ties.
                const bool bcast_close = topk_bcast && backend_ctx->tps > 1 &&
                    node->op == GGML_OP_TOP_K &&
                    !(node->src[0] != nullptr && node->src[0]->op == GGML_OP_LIGHTNING_INDEXER);
                const bool end_close = (i + 1 == cgraph->n_nodes);
                if (!ar_close && !bcast_close && !end_close) {
                    continue;
                }

                int i_delayed = bcast_close ? i : get_i_delayed(i);

                // Fold this reduce into a following ADD of two partial sums: the
                // lane-local ADD of partials is itself a partial of the total, so
                // the ADD closes with ONE AllReduce covering both operands. The
                // other operand's producer chain sits between the delayed node and
                // the ADD and is absorbed into this subgraph - it computes
                // lane-locally. Without this, splitting the shared expert closes
                // two AllReduces per layer (routed experts, then the ADD).
                if (ar_close && !bcast_close) {
                    ggml_tensor * dnode = cgraph->nodes[i_delayed];
                    if (ggml_node_get_use_count(cgraph, i_delayed) == 1) {
                        for (int k = i_delayed + 1; k < cgraph->n_nodes; k++) {
                            ggml_tensor * cons = cgraph->nodes[k];
                            bool uses = false;
                            for (int s = 0; s < GGML_MAX_SRC; s++) {
                                uses = uses || cons->src[s] == dnode;
                            }
                            if (!uses) {
                                continue;
                            }
                            if (cons->op == GGML_OP_ADD) {
                                ggml_tensor * other = cons->src[0] == dnode ? cons->src[1] : cons->src[0];
                                if (other != nullptr &&
                                        ggml_backend_meta_get_split_state(other, false).axis ==
                                        GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                                    i_delayed = k;
                                }
                            }
                            break;
                        }
                    }
                }

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                if (i_delayed > i) {
                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        if ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            for (int ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].offset = i_start;
                }
                // A PARTIAL last node still needs its AllReduce even at graph end: a
                // fragment ends on every node, and the consumer in the next fragment
                // (or the readback) assumes the sum has been reduced.
                backend_ctx->subgraphs.push_back({ (size_t) current_stage,
                                                   ar_close
                                                       ? ggml_backend_meta_context::subgraph_closure::AR
                                                       : (bcast_close
                                                           ? ggml_backend_meta_context::subgraph_closure::BCAST
                                                           : ggml_backend_meta_context::subgraph_closure::NONE),
                                                   {} });
                n_subgraphs++;
                i_start = i + 1;
            }
            if (i_start < cgraph->n_nodes) {
                // Only reachable for fragments: trailing nodes the walk skipped
                // (host-buffer views) still need a subgraph so every node lands in a
                // per-lane graph. A whole graph always ends on a computable node.
                for (size_t j = 0; j < n_backends; j++) {
                    backend_ctx->backend_configs[j].cgraphs[n_subgraphs].offset = i_start;
                }
                backend_ctx->subgraphs.push_back({ (size_t) current_stage,
                                                   ggml_backend_meta_context::subgraph_closure::NONE, {} });
                n_subgraphs++;
                i_start = cgraph->n_nodes;
            }
            GGML_ASSERT(i_start == cgraph->n_nodes);

            if (fragment) {
                backend_ctx->frag_last_stage = current_stage;
                for (int i = 0; i < cgraph->n_nodes; i++) {
                    ggml_tensor * node = cgraph->nodes[i];
                    if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                        continue;
                    }
                    const auto it = backend_ctx->tensor_stage.find(node);
                    const size_t node_stage = it != backend_ctx->tensor_stage.end() ? it->second : (size_t) current_stage;
                    backend_ctx->frag_stage_bits[node] = 1u << node_stage;
                    if (node->view_src != nullptr) {
                        const ggml_tensor * root = node;
                        while (root->view_src != nullptr) {
                            root = root->view_src;
                        }
                        if (root->buffer != nullptr &&
                            ggml_backend_buffer_get_usage(root->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
                            backend_ctx->frag_root_stage[root] = node_stage;
                        }
                    }
                }
            }

            // Build per-transition xfer sets. A transition between sg i and sg i+1 needs to
            // broadcast every MIRRORED graph node that was produced in sg i's stage (or any
            // earlier stage) and is referenced by any node in sg i+1 or later. Transferring
            // only the prior subgraph's last node is insufficient because residual-stream
            // tensors (e.g. l_out-N) are produced earlier in the same subgraph but consumed
            // by the next stage's residual ADDs.
            {
                // Dependencies that flow through a persistent buffer instead of an src edge.
                // A KV-cache write is a SET_ROWS node writing through a view of the cache; the
                // matching read is a separate view of the same cache. No src edge links them,
                // and the cache is not a graph node, so the scan below cannot see the
                // dependency - it skips the read's source as "likely a weight" and leaves the
                // write stranded on the producing stage's lanes.
                //
                // That is a silent wrong answer whenever a stage boundary falls between a
                // cache write and its read. On DeepSeek-V4 at -tps 4 (2 stages) it cost +1.77%
                // perplexity: head-splitting attn_q_b makes an attention weight stage-1-owned,
                // which moves the transition from node 4592 to 4482, in between two writes to
                // cache_k_l22 and the view that reads them back.
                for (size_t s = 0; s < n_subgraphs; s++) {
                    if (backend_ctx->subgraphs[s].closure != ggml_backend_meta_context::subgraph_closure::TRANSFER) {
                        continue;
                    }
                    const int boundary_idx = (s + 1 < n_subgraphs)
                        ? backend_ctx->backend_configs[0].cgraphs[s + 1].offset
                        : cgraph->n_nodes;
                    auto planned = planned_xfer_sets.find(boundary_idx);
                    if (planned != planned_xfer_sets.end()) {
                        backend_ctx->subgraphs[s].xfer = std::move(planned->second);
                    } else {
                        backend_ctx->subgraphs[s].xfer = collect_xfer_set(boundary_idx);
                    }
                }
            }

            // GGML_META_PART_DEBUG=1 dumps per-subgraph stage/closure summary on rebuild.
            if (backend_ctx->dbg_part) {
                size_t n_ar = 0, n_xfer = 0, n_none = 0;
                for (const auto & sg : backend_ctx->subgraphs) {
                    n_ar   += (sg.closure == ggml_backend_meta_context::subgraph_closure::AR);
                    n_xfer += (sg.closure == ggml_backend_meta_context::subgraph_closure::TRANSFER);
                    n_none += (sg.closure == ggml_backend_meta_context::subgraph_closure::NONE);
                }
                fprintf(stderr, "[meta-part] n_subgraphs=%zu n_stages=%zu tps=%zu | AR=%zu XFER=%zu NONE=%zu\n",
                        n_subgraphs, backend_ctx->n_stages, backend_ctx->tps, n_ar, n_xfer, n_none);
                for (size_t k = 0; k < n_subgraphs; k++) {
                    const size_t off  = backend_ctx->backend_configs[0].cgraphs[k].offset;
                    const size_t off2 = (k + 1 < n_subgraphs) ? (size_t) backend_ctx->backend_configs[0].cgraphs[k+1].offset : (size_t) cgraph->n_nodes;
                    const auto & sg   = backend_ctx->subgraphs[k];
                    const char * cn   = (sg.closure == ggml_backend_meta_context::subgraph_closure::AR) ? "AR" :
                                        (sg.closure == ggml_backend_meta_context::subgraph_closure::TRANSFER) ? "XFER" :
                                        (sg.closure == ggml_backend_meta_context::subgraph_closure::BCAST) ? "BCAST" : "NONE";
                    fprintf(stderr, "[meta-part]   sg%zu stage=%zu closure=%s nodes=[%zu,%zu) last=%s xfer_set=%zu\n",
                            k, sg.stage, cn, off, off2,
                            off2 > off ? cgraph->nodes[off2 - 1]->name : "(empty)",
                            sg.xfer.size());
                    if (sg.closure == ggml_backend_meta_context::subgraph_closure::TRANSFER) {
                        for (ggml_tensor * t : sg.xfer) {
                            fprintf(stderr, "[meta-part]     xfer: %s\n", t->name);
                        }
                    }
                }
            }
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;

        if (max_tmp_size > backend_ctx->max_tmp_size) {
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx->max_tmp_size = max_tmp_size;
        }

        if (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            const size_t n_nodes_per_device = 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            const size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            const size_t mem_per_device_graphs_main = backend_ctx->max_subgraphs*ggml_graph_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            const size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            const size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            const ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                // Recreate up to max_subgraphs at max_nnodes capacity, not this graph's
                // counts: the ctx reset above destroyed every previously created graph,
                // and a later graph may reuse any index up to the maxima without passing
                // through this branch again. Fragments hit this (their node and subgraph
                // counts vary independently), whole graphs raise both maxima together.
                for (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, /*grads =*/ false);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_graph].cgraph_main;
                const size_t i_node_start = bcj.cgraphs[i_graph].offset;
                const size_t i_node_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_node_stop - i_node_start;
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                for (size_t i_node = i_node_start; i_node < i_node_stop; i_node++) {
                    ggml_tensor * node_ij = bcj.nodes[i_node];
                    cgraph_ij->nodes[i_node - i_node_start] = node_ij;
                    const size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_node]);
                    const size_t hash_pos_ij = ggml_hash_insert(&cgraph_ij->visited_hash_set, node_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_graph_next_uid();
            }
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    auto get_node_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, sizeof(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        return ret;
    };
    auto set_tmp_data = [&](ggml_tensor * tensor, const size_t j, const size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_buf];
        if (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, backend_ctx->max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
    };
    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        return ret;
    };

    // Generic butterfly fallback. Operates within a single stage's tps lanes
    // [lane_lo, lane_lo + lane_count). Used when the backend-native AllReduce is not
    // available (e.g. comm_init returned null) or returned false. Indices are local k in
    // [0, lane_count); j = lane_lo + k addresses backend_configs.
    auto allreduce_fallback = [&](size_t i, size_t lane_lo, size_t lane_count) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(lane_count, nullptr);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        for (size_t k = 0; k < lane_count; k++) {
            auto & bcj = backend_ctx->backend_configs[lane_lo + k];
            ggml_tensor * node = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            if (node->flags & GGML_TENSOR_FLAG_COMPUTE) {
                continue;
            }
            ggml_tensor * node_zero = get_node_aux(node);
            node_zero->op = GGML_OP_SCALE; // FIXME 0.0f * NaN == NaN
            node_zero->src[0] = node;
            ggml_set_op_params_f32(node_zero, 0, 0.0f);
            node_zero->data = node->data;
            node_zero->buffer = node->buffer;
            node_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

            step_cgraphs[k] = get_cgraph_aux();
            step_cgraphs[k]->nodes[0] = node_zero;
            step_cgraphs[k]->n_nodes = 1;
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[k]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        auto push_data = [&](const size_t k_src, const size_t k_dst, const size_t i_buf) {
            assert(step_cgraphs[k_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[lane_lo + k_src];
            auto & bcj_dst = backend_ctx->backend_configs[lane_lo + k_dst];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(node_src));
            GGML_ASSERT(ggml_is_contiguous(node_dst));

            ggml_tensor * node_tmp = get_node_aux(node_dst);
            set_tmp_data(node_tmp, lane_lo + k_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_tmp);

            ggml_tensor * node_red = get_node_aux(node_dst);
            node_red->view_src = node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
            node_red->view_offs = node_dst->view_offs;
            node_red->op = GGML_OP_ADD;
            node_red->src[0] = node_dst;
            node_red->src[1] = node_tmp;
            node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(node_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = node_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[k_dst] = cgraph_aux;
        };

        size_t offset_j = lane_count/2;
        while (offset_j > 0 && (offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        const size_t offset_j_max = offset_j;
        size_t i_buf = 0;

        // If lane_count is not a power of 2, fold in the excess prior to butterfly reduction:
        for (size_t k_src = 2*offset_j_max; k_src < lane_count; k_src++) {
            const size_t k_dst = k_src - 2*offset_j_max;
            push_data(k_src, k_dst, i_buf);
            const ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[lane_lo + k_dst].backend, step_cgraphs[k_dst]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        for (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            for (size_t k = 0; k < 2*offset_j_max; k++) {
                const size_t k_other = k ^ offset_j;
                if (k_other >= lane_count) {
                    continue;
                }
                push_data(k, k_other, i_buf);
            }

            for (size_t k = 0; k < 2*offset_j_max; k++) {
                if (step_cgraphs[k] == nullptr) {
                    continue;
                }
                auto & bcj = backend_ctx->backend_configs[lane_lo + k];
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[k]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            i_buf++;
        }
        assert(i_buf == backend_ctx->n_reduce_steps);

        // If lane_count is not a power of 2, copy back the reduced tensors to the excess:
        for (size_t k = 2*offset_j_max; k < lane_count; k++) {
            auto & bcj_src = backend_ctx->backend_configs[lane_lo + (k - 2*offset_j_max)];
            auto & bcj_dst = backend_ctx->backend_configs[lane_lo + k];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_dst);
        }

        return GGML_STATUS_SUCCESS;
    };

    // Stage transition: lane-pair-wise broadcast of every MIRRORED tensor from this stage's
    // lanes to the next stage's lanes that's in the precomputed xfer set for this transition.
    // The xfer set was built at partition time (see backend_ctx->subgraphs[i].xfer) and
    // covers exactly the tensors stage_b's compute reads from stage_a's outputs.
    // MIRRORED tensors have full-size simple_tensor allocations on every lane, so the per-
    // lane copy lands in pre-allocated memory.
    auto transfer_tensors = [&](const std::vector<ggml_tensor *> & xfer_set, size_t stage_a, size_t stage_b) -> ggml_status {
        const size_t lane_lo_a = stage_a * backend_ctx->tps;
        const size_t lane_lo_b = stage_b * backend_ctx->tps;
        const bool   xfer_debug = backend_ctx->dbg_xfer;
        if (xfer_set.empty()) {
            return GGML_STATUS_SUCCESS;
        }

        size_t max_xfer_nbytes = 0;
        for (ggml_tensor * boundary : xfer_set) {
            ggml_tensor * sj = ggml_backend_meta_buffer_simple_tensor(boundary, lane_lo_a);
            GGML_ASSERT(sj != nullptr);
            max_xfer_nbytes = std::max(max_xfer_nbytes, ggml_nbytes(sj));
        }
        const bool use_xfer_comm =
            backend_ctx->xfer_comm_default ||
            (backend_ctx->xfer_comm_moe_large && backend_ctx->graph_has_moe_ops &&
             max_xfer_nbytes >= backend_ctx->xfer_comm_moe_threshold);

        if (use_xfer_comm && backend_ctx->xfer_comm_ctx != nullptr && backend_ctx->comm_sendrecv != nullptr) {
            std::vector<ggml_backend_t>         src_backends;
            std::vector<ggml_backend_t>         dst_backends;
            std::vector<const ggml_tensor *>    src_tensors;
            std::vector<ggml_tensor *>          dst_tensors;
            src_backends.reserve(xfer_set.size() * backend_ctx->tps);
            dst_backends.reserve(xfer_set.size() * backend_ctx->tps);
            src_tensors.reserve(xfer_set.size() * backend_ctx->tps);
            dst_tensors.reserve(xfer_set.size() * backend_ctx->tps);

            for (ggml_tensor * boundary : xfer_set) {
                for (size_t k = 0; k < backend_ctx->tps; k++) {
                    ggml_backend_t src_backend = backend_ctx->backend_configs[lane_lo_a + k].backend;
                    ggml_backend_t dst_backend = backend_ctx->backend_configs[lane_lo_b + k].backend;
                    ggml_tensor * sj = ggml_backend_meta_buffer_simple_tensor(boundary, lane_lo_a + k);
                    ggml_tensor * dj = ggml_backend_meta_buffer_simple_tensor(boundary, lane_lo_b + k);
                    GGML_ASSERT(sj != nullptr && dj != nullptr);
                    GGML_ASSERT(ggml_nbytes(sj) == ggml_nbytes(dj));
                    if (xfer_debug) {
                        fprintf(stderr, "[xfer-comm] stage %zu->%zu lane %zu->%zu name=%s nbytes=%zu\n",
                            stage_a, stage_b, lane_lo_a + k, lane_lo_b + k, boundary->name, ggml_nbytes(sj));
                    }
                    src_backends.push_back(src_backend);
                    dst_backends.push_back(dst_backend);
                    src_tensors.push_back(sj);
                    dst_tensors.push_back(dj);
                }
            }

            const bool ok = backend_ctx->comm_sendrecv(
                backend_ctx->xfer_comm_ctx, src_tensors.size(),
                src_backends.data(), dst_backends.data(), src_tensors.data(), dst_tensors.data());
            if (ok) {
                return GGML_STATUS_SUCCESS;
            }
        }

        if (backend_ctx->cpy_async_dedicated_queue != nullptr) {
            // Fallback when the RCCL sendrecv path above is unavailable (NCCL not built,
            // GGML_META_XFER_RCCL=0, or comm_sendrecv returned false). Memcpy on a side
            // stream so it doesn't serialize behind src compute. Two-phase: queue per
            // (boundary, lane), then one drain per (src, dst) lane pair. The drain
            // conditionally host-syncs pp_copy_stream under HWQ > 4 - see
            // ggml_pp_drain_should_sync in the CUDA backend.
            for (ggml_tensor * boundary : xfer_set) {
                for (size_t k = 0; k < backend_ctx->tps; k++) {
                    ggml_backend_t src_backend = backend_ctx->backend_configs[lane_lo_a + k].backend;
                    ggml_backend_t dst_backend = backend_ctx->backend_configs[lane_lo_b + k].backend;
                    ggml_tensor * sj = ggml_backend_meta_buffer_simple_tensor(boundary, lane_lo_a + k);
                    ggml_tensor * dj = ggml_backend_meta_buffer_simple_tensor(boundary, lane_lo_b + k);
                    GGML_ASSERT(sj != nullptr && dj != nullptr);
                    GGML_ASSERT(ggml_nbytes(sj) == ggml_nbytes(dj));
                    if (xfer_debug) {
                        fprintf(stderr, "[xfer-async] stage %zu->%zu lane %zu->%zu name=%s nbytes=%zu\n",
                            stage_a, stage_b, lane_lo_a + k, lane_lo_b + k, boundary->name, ggml_nbytes(sj));
                    }
                    const bool ok = backend_ctx->cpy_async_dedicated_queue(src_backend, dst_backend, sj, dj);
                    GGML_ASSERT(ok && "cpy_async_dedicated_queue returned false at runtime");
                }
            }
            // Drain: one event_b record + one dst->main wait per (src_lane, dst_lane). The
            // FIFO on each src's pp_copy_stream guarantees event_b fires only after every
            // queued memcpy for that lane has completed, so dst->main resumes the next
            // subgraph with all boundary buffers fully written.
            for (size_t k = 0; k < backend_ctx->tps; k++) {
                ggml_backend_t src_backend = backend_ctx->backend_configs[lane_lo_a + k].backend;
                ggml_backend_t dst_backend = backend_ctx->backend_configs[lane_lo_b + k].backend;
                const bool ok = backend_ctx->cpy_async_dedicated_drain(src_backend, dst_backend);
                GGML_ASSERT(ok && "cpy_async_dedicated_drain returned false at runtime");
            }
            return GGML_STATUS_SUCCESS;
        }

        // Fallback: synchronous path. The buffer-iface cpy_tensor path issues the memcpy on
        // cudaStreamPerThread (separate from the backend's main stream) and host-syncs, so
        // it doesn't serialize subsequent compute on the backend stream. Used when the simple
        // backend doesn't expose the dedicated queue+drain pair.
        for (size_t k = 0; k < backend_ctx->tps; k++) {
            ggml_backend_synchronize(backend_ctx->backend_configs[lane_lo_a + k].backend);
        }
        for (ggml_tensor * boundary : xfer_set) {
            for (size_t k = 0; k < backend_ctx->tps; k++) {
                ggml_tensor * sj = ggml_backend_meta_buffer_simple_tensor(boundary, lane_lo_a + k);
                ggml_tensor * dj = ggml_backend_meta_buffer_simple_tensor(boundary, lane_lo_b + k);
                GGML_ASSERT(sj != nullptr && dj != nullptr);
                GGML_ASSERT(ggml_nbytes(sj) == ggml_nbytes(dj));
                if (xfer_debug) {
                    fprintf(stderr, "[xfer-sync] stage %zu->%zu lane %zu->%zu name=%s nbytes=%zu\n",
                        stage_a, stage_b, lane_lo_a + k, lane_lo_b + k, boundary->name, ggml_nbytes(sj));
                }
                ggml_backend_tensor_copy(sj, dj);
            }
        }
        return GGML_STATUS_SUCCESS;
    };

    auto stage_transfer = [&](size_t i_subgraph, size_t stage_a, size_t stage_b) -> ggml_status {
        const ggml_status st = transfer_tensors(backend_ctx->subgraphs[i_subgraph].xfer, stage_a, stage_b);

        return st;
    };

    // Fragment pre-pass: fragments execute one at a time, so a stage-crossing data
    // dependency can span two graph_compute calls where the whole-graph walk would
    // have placed a TRANSFER closure. Bring every MIRRORED source this fragment
    // reads up to date on the consuming stage's lanes before running it. Weights are
    // mirrored on every lane already and inputs are written to all lanes by
    // set_tensor, so only tracked graph nodes and persistent-buffer views transfer.
    if (fragment && backend_ctx->n_stages > 1) {
        std::unordered_set<const ggml_tensor *> in_frag;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            in_frag.insert(cgraph->nodes[i]);
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                continue;
            }
            const auto itn = backend_ctx->tensor_stage.find(node);
            if (itn == backend_ctx->tensor_stage.end()) {
                continue;
            }
            const size_t node_stage = itn->second;
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                ggml_tensor * src = node->src[s];
                if (src == nullptr || in_frag.count(src) > 0) {
                    continue;
                }
                if (src->buffer == nullptr || !ggml_backend_buffer_is_meta(src->buffer)) {
                    continue;
                }
                if (ggml_backend_buffer_get_usage(src->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                    continue;
                }
                const ggml_backend_meta_split_state ss = ggml_backend_meta_get_split_state(src, /*assume_sync =*/ true);
                if (ss.axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                    continue;
                }
                const auto itb = backend_ctx->frag_stage_bits.find(src);
                if (itb != backend_ctx->frag_stage_bits.end()) {
                    if (itb->second & (1u << node_stage)) {
                        continue;
                    }
                    const auto its = backend_ctx->tensor_stage.find(src);
                    if (its == backend_ctx->tensor_stage.end() || its->second == node_stage) {
                        continue;
                    }
                    const ggml_status st = transfer_tensors({ src }, its->second, node_stage);
                    if (st != GGML_STATUS_SUCCESS) {
                        return st;
                    }
                    itb->second |= 1u << node_stage;
                } else {
                    // Not a tracked graph node - possibly a view of a persistent buffer
                    // (KV cache) that another stage's fragment wrote through. No dedup:
                    // the written region can grow between reads.
                    const ggml_tensor * root = src;
                    while (root->view_src != nullptr) {
                        root = root->view_src;
                    }
                    const auto itr = backend_ctx->frag_root_stage.find(root);
                    if (itr == backend_ctx->frag_root_stage.end() || itr->second == node_stage) {
                        continue;
                    }
                    const ggml_status st = transfer_tensors({ src }, itr->second, node_stage);
                    if (st != GGML_STATUS_SUCCESS) {
                        return st;
                    }
                }
            }
        }
    }

    // ---- whole-token graph ------------------------------------------------
    // One host round trip per lane per STAGE instead of one per subgraph boundary,
    // so the submission spread the AllReduce barrier bills is paid n_stages times
    // rather than at every one of the ~81 (dense) to ~172 (head-split MLA) closures.
    // Single-stage is one run, i.e. the whole token, unchanged.
    // First subgraph the per-subgraph loop below must dispatch. The token-graph
    // paths raise it past the captured prefix when a replay or a fresh capture
    // already issued the leading subgraphs.
    size_t i_dispatch_first = 0;

    if (backend_ctx->token_graph && cgraph->uid != 0 &&
        backend_ctx->tg_capture_begin != nullptr && backend_ctx->tg_capture_end != nullptr &&
        backend_ctx->comm_ar_prepare != nullptr && backend_ctx->comm_ar_launch_rank != nullptr &&
        backend_ctx->tps > 1 && backend_ctx->n_stages * backend_ctx->tps == n_backends &&
        backend_ctx->comm_ctxs.size() >= backend_ctx->n_stages) {

        auto * tge = backend_ctx->tg_lookup(cgraph->uid);

        // Replay. Each run is one launch per lane of its stage, then the stage transfer
        // on the host, which is what the next stage's inputs depend on.
        if (!tge->runs.empty()) {
            bool ready = true;
            for (const auto & r : tge->runs) {
                if (r.exec.size() != backend_ctx->tps) { ready = false; break; }
                for (void * e : r.exec) {
                    if (e == nullptr) { ready = false; break; }
                }
                if (!ready) { break; }
            }
            if (ready) {
                for (const auto & r : tge->runs) {
                    const size_t lane_lo = r.stage * backend_ctx->tps;
                    for (size_t k = 0; k < backend_ctx->tps; k++) {
                        backend_ctx->tg_graph_launch(
                            backend_ctx->backend_configs[lane_lo + k].backend, r.exec[k]);
                    }
                }
                if (tge->covered >= backend_ctx->n_subgraphs) {
                    return GGML_STATUS_SUCCESS;
                }
                // Debug-limited capture: the tail past the captured prefix runs
                // through the ordinary per-subgraph loop below.
                i_dispatch_first = tge->covered;
                goto per_subgraph_dispatch;
            }
        }

        tge->warm++;

        // Record, once this shape has run enough times for the scratch pool to
        // stop growing. A run ends at a TRANSFER closure, which stays on the host.
        // An AllReduce that falls back to NCCL on the size gate still cannot be
        // captured, and that only shows up at prepare time below.
        if (!tge->failed && tge->warm > backend_ctx->tg_warm_needed) {
            // Debug bisect knob: capture only the first GGML_META_TG_LIMIT
            // subgraphs and dispatch the rest per-subgraph. Localizes which
            // subgraph range replays incorrectly. Default: capture everything.
            static const size_t tg_limit = []() {
                const char * e = getenv("GGML_META_TG_LIMIT");
                return e != nullptr ? (size_t) strtoull(e, nullptr, 10) : (size_t) SIZE_MAX;
            }();
            const size_t n_cap = std::min(backend_ctx->n_subgraphs, tg_limit);
            std::vector<ggml_backend_meta_context::tg_run> runs;
            {
                ggml_backend_meta_context::tg_run cur;
                bool open = false;
                for (size_t i = 0; i < n_cap; i++) {
                    const auto & sg = backend_ctx->subgraphs[i];
                    if (!open) {
                        cur = {};
                        cur.stage = sg.stage;
                        cur.i_beg = i;
                        open = true;
                    }
                    if (sg.stage != cur.stage) {
                        // A stage change without an intervening transfer is a shape this
                        // pass does not model. Fall back rather than capture it wrong.
                        // `open` must drop with the runs: the push after this loop would
                        // otherwise resurrect the partial run and replay would silently
                        // skip every subgraph past it.
                        runs.clear();
                        open = false;
                        break;
                    }
                    cur.i_end = i + 1;
                    if (sg.closure == ggml_backend_meta_context::subgraph_closure::BCAST) {
                        // The broadcast is a host-side copy that capture cannot
                        // record - fall back to per-subgraph dispatch. Same rule as
                        // above: close the run or the post-loop push turns this
                        // rejection into a partial capture that truncates the token.
                        runs.clear();
                        open = false;
                        break;
                    }
                    if (sg.closure == ggml_backend_meta_context::subgraph_closure::TRANSFER) {
                        // multi-stage shapes are not captured
                        runs.clear();
                        open = false;
                        break;
                    }
                }
                if (open) {
                    runs.push_back(cur);
                }
            }

            // Multi-stage capture stays off: it produced wrong results on MoE
            // targets and measured no win, the stage transfer reintroduces the
            // barrier skew the capture removes.
            bool eligible = !runs.empty() && backend_ctx->n_stages == 1;
            for (const auto & r : runs) {
                if (r.i_end <= r.i_beg) { eligible = false; break; }
                if (backend_ctx->comm_ctxs[r.stage] == nullptr) { eligible = false; break; }
            }

            if (!eligible) {
                tge->failed = true;
                GGML_LOG_DEBUG("%s: uid %zu not eligible for token graph\n",
                               __func__, (size_t) cgraph->uid);
            } else {
                backend_ctx->tg_free_entry(*tge);
                tge->runs    = runs;
                tge->covered = n_cap;

                bool ok = true;
                std::vector<ggml_tensor *> nodes;
                for (auto & r : tge->runs) {
                    if (!ok) { break; }
                    const size_t lane_lo = r.stage * backend_ctx->tps;
                    void * comm = backend_ctx->comm_ctxs[r.stage];
                    r.exec.assign(backend_ctx->tps, nullptr);

                    for (size_t k = 0; k < backend_ctx->tps && ok; k++) {
                        ggml_backend_t bj = backend_ctx->backend_configs[lane_lo + k].backend;
                        ggml_backend_synchronize(bj);   // the stream must be quiet to enter capture
                        if (!backend_ctx->tg_capture_begin(bj)) { ok = false; break; }

                        for (size_t i = r.i_beg; i < r.i_end; i++) {
                            const auto & sg = backend_ctx->subgraphs[i];
                            auto & bcj = backend_ctx->backend_configs[lane_lo + k];
                            if (bcj.cgraphs[i].cgraph_main->n_nodes == 0) {
                                continue;
                            }
                            if (ggml_backend_graph_compute_async(bj, bcj.cgraphs[i].cgraph_main)
                                    != GGML_STATUS_SUCCESS) {
                                ok = false; break;
                            }
                            if (sg.closure != ggml_backend_meta_context::subgraph_closure::AR) {
                                continue;
                            }
                            nodes.clear();
                            nodes.reserve(backend_ctx->tps);
                            for (size_t m = 0; m < backend_ctx->tps; m++) {
                                ggml_cgraph * cg = backend_ctx->backend_configs[lane_lo + m].cgraphs[i].cgraph_main;
                                nodes.push_back(cg->nodes[cg->n_nodes - 1]);
                            }
                            const int nr = backend_ctx->comm_ar_prepare(comm, nodes.data());
                            if (nr < 0) { ok = false; break; }
                            if (nr > 0) {
                                backend_ctx->comm_ar_launch_rank(comm, (int) k);
                            }
                        }

                        void * exec = backend_ctx->tg_capture_end(bj);
                        if (!ok || exec == nullptr) { ok = false; break; }
                        r.exec[k] = exec;
                    }
                }

                if (ok) {
                    // Capture records without executing, so this token still runs.
                    for (const auto & r : tge->runs) {
                        const size_t lane_lo = r.stage * backend_ctx->tps;
                        for (size_t k = 0; k < backend_ctx->tps; k++) {
                            backend_ctx->tg_graph_launch(
                                backend_ctx->backend_configs[lane_lo + k].backend, r.exec[k]);
                        }
                    }
                    GGML_LOG_DEBUG("%s: uid %zu captured %zu runs x %zu lanes, %zu/%zu subgraphs\n",
                                   __func__, (size_t) cgraph->uid, tge->runs.size(),
                                   backend_ctx->tps, tge->covered, backend_ctx->n_subgraphs);
                    if (tge->covered >= backend_ctx->n_subgraphs) {
                        return GGML_STATUS_SUCCESS;
                    }
                    i_dispatch_first = tge->covered;
                    goto per_subgraph_dispatch;
                }

                tge->failed = true;
                backend_ctx->tg_free_entry(*tge);
                GGML_LOG_DEBUG("%s: uid %zu capture failed, using per-subgraph dispatch\n",
                               __func__, (size_t) cgraph->uid);
            }
        }
    }
    // ------------------------------------------------------------------------

per_subgraph_dispatch:
    const bool run_debug = backend_ctx->dbg_run;
    for (size_t i = i_dispatch_first; i < backend_ctx->n_subgraphs; i++) {
        const auto & sg      = backend_ctx->subgraphs[i];
        const size_t stage   = sg.stage;
        const size_t lane_lo = stage * backend_ctx->tps;
        const size_t lane_hi = lane_lo + backend_ctx->tps;
        if (run_debug) {
            ggml_cgraph * cg0 = backend_ctx->backend_configs[lane_lo].cgraphs[i].cgraph_main;
            const char * last = (cg0->n_nodes > 0 && cg0->nodes[cg0->n_nodes-1]) ? cg0->nodes[cg0->n_nodes-1]->name : "(none)";
            const char * first = (cg0->n_nodes > 0 && cg0->nodes[0]) ? cg0->nodes[0]->name : "(none)";
            fprintf(stderr, "[run] sg%zu stage=%zu lanes=[%zu,%zu) n_nodes=%d first=%s last=%s closure=%d\n",
                    i, stage, lane_lo, lane_hi, cg0->n_nodes, first, last, (int) sg.closure);
        }
        const uint64_t prof_t0 = backend_ctx->prof ? ggml_meta_prof_now() : 0;

        // Issuing the lanes in order makes lane_hi-1 arrive at the closure below last, and
        // the AllReduce bills that wait as its own runtime. Hand the trailing lanes to the
        // worker pool so every lane starts together, and keep lane_lo on this thread.
        if (backend_ctx->parallel_dispatch && lane_hi - lane_lo > 1) {
            auto & disp = backend_ctx->dispatcher;
            GGML_ASSERT(disp.jobs.size() >= lane_hi - lane_lo - 1);
            for (size_t j = lane_lo + 1; j < lane_hi; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                auto & jb  = disp.jobs[j - lane_lo - 1];
                jb.backend = bcj.backend;
                // an empty subgraph (seam snapped to the block start) carries only
                // its closure - nothing to compute on the lanes
                jb.cgraph  = bcj.cgraphs[i].cgraph_main->n_nodes > 0 ? bcj.cgraphs[i].cgraph_main : nullptr;
            }
            disp.dispatch_async();

            auto & bc0 = backend_ctx->backend_configs[lane_lo];
            const ggml_status status_self = bc0.cgraphs[i].cgraph_main->n_nodes > 0
                ? ggml_backend_graph_compute_async(bc0.backend, bc0.cgraphs[i].cgraph_main)
                : GGML_STATUS_SUCCESS;

            // Join before inspecting either status: the workers must be quiesced before we
            // can return, otherwise they would still be touching backend state.
            const ggml_status status_workers = disp.join();
            if (status_self != GGML_STATUS_SUCCESS) {
                return status_self;
            }
            if (status_workers != GGML_STATUS_SUCCESS) {
                return status_workers;
            }

        } else {
            for (size_t j = lane_lo; j < lane_hi; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                if (bcj.cgraphs[i].cgraph_main->n_nodes == 0) {
                    continue;
                }
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }

        const uint64_t prof_t1 = backend_ctx->prof ? ggml_meta_prof_now() : 0;
        if (sg.closure == ggml_backend_meta_context::subgraph_closure::AR && backend_ctx->tps > 1) {
            bool backend_allreduce_success = false;
            if (backend_ctx->comm_ctxs[stage] != nullptr) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(backend_ctx->tps);
                for (size_t j = lane_lo; j < lane_hi; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes - 1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctxs[stage], nodes.data());
            }

            if (!backend_allreduce_success) {
                const ggml_status status = allreduce_fallback(i, lane_lo, backend_ctx->tps);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        } else if (sg.closure == ggml_backend_meta_context::subgraph_closure::TRANSFER) {
            GGML_ASSERT(i + 1 < backend_ctx->n_subgraphs);
            const size_t stage_b = backend_ctx->subgraphs[i + 1].stage;
            const ggml_status status = stage_transfer(i, stage, stage_b);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        } else if (sg.closure == ggml_backend_meta_context::subgraph_closure::BCAST && backend_ctx->tps > 1) {
            // Broadcast the first lane's selection to the stage's other lanes so
            // every lane applies the same decision. The first lane must have
            // finished producing before its buffer is read, and the receiving
            // lanes must not run their next subgraph until the copy landed - a
            // host sync on the stage's lanes covers both orderings. Cost: one
            // small I32 tensor per csa layer.
            auto & bc0 = backend_ctx->backend_configs[lane_lo];
            ggml_cgraph * cg0 = bc0.cgraphs[i].cgraph_main;
            ggml_tensor * n0 = cg0->nodes[cg0->n_nodes - 1];
            ggml_backend_synchronize(bc0.backend);
            for (size_t j = lane_lo + 1; j < lane_hi; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                ggml_cgraph * cgj = bcj.cgraphs[i].cgraph_main;
                ggml_tensor * nj = cgj->nodes[cgj->n_nodes - 1];
                ggml_backend_tensor_copy_async(bc0.backend, bcj.backend, n0, nj);
            }
            for (size_t j = lane_lo + 1; j < lane_hi; j++) {
                ggml_backend_synchronize(backend_ctx->backend_configs[j].backend);
            }
        }
        if (backend_ctx->prof) {
            const uint64_t prof_t2 = ggml_meta_prof_now();
            backend_ctx->prof_ns_lanes += prof_t1 - prof_t0;
            backend_ctx->prof_ns_close += prof_t2 - prof_t1;
            backend_ctx->prof_subgraphs++;
        }
        // closure == NONE: last subgraph, no closure action.
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_meta_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    auto * ev_ctx = (ggml_backend_meta_event_context *) event->context;
    const size_t n = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(ev_ctx->simple_events.size() == n);
    for (size_t j = 0; j < n; j++) {
        ggml_backend_event_record(ev_ctx->simple_events[j], ggml_backend_meta_simple_backend(backend, j));
    }
}

static void ggml_backend_meta_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    auto * ev_ctx = (ggml_backend_meta_event_context *) event->context;
    const size_t n = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(ev_ctx->simple_events.size() == n);
    for (size_t j = 0; j < n; j++) {
        ggml_backend_event_wait(ggml_backend_meta_simple_backend(backend, j), ev_ctx->simple_events[j]);
    }
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ ggml_backend_meta_event_record,
    /* .event_wait              = */ ggml_backend_meta_event_wait,
    /* .graph_optimize          = */ nullptr,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

size_t ggml_backend_meta_n_stages(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->n_stages;
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}
