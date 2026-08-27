#pragma once

// Private scheduler controls shared by ggml and in-tree consumers.

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Force a host drain before overwriting non-graph inputs. Used by contexts
// whose tiny ubatches can outrun cross-device event ordering.
GGML_API void ggml_backend_sched_set_sync_non_graph_inputs(ggml_backend_sched_t sched, bool enabled);

#ifdef __cplusplus
}
#endif
