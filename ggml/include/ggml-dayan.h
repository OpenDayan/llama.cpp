#pragma once

#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_DAYAN_NAME "DAYAN"

#define GGML_DAYAN_SLOT_TRACE_MAX_NODES   5
#define GGML_DAYAN_SLOT_TRACE_MAX_OUTPUTS 4
#define GGML_DAYAN_GRAPH_PLAN_MAX_INPUTS  8

struct ggml_backend_dayan_slot_trace_event {
    uint32_t                   version;
    const char *               kind;
    size_t                     slot_index;
    const struct ggml_tensor * nodes[GGML_DAYAN_SLOT_TRACE_MAX_NODES];
    size_t                     node_count;
    const struct ggml_tensor * outputs[GGML_DAYAN_SLOT_TRACE_MAX_OUTPUTS];
    size_t                     output_count;
    bool                       executed;
    int32_t                    status;
    int64_t                    elapsed_us;
    uint32_t                   output_mask;
    uint32_t                   sort_k;
};

typedef void (*ggml_backend_dayan_slot_trace_callback)(const struct ggml_backend_dayan_slot_trace_event * event,
                                                       void *                                             user_data);

typedef bool (*ggml_backend_dayan_slot_filter_callback)(const struct ggml_backend_dayan_slot_trace_event * event,
                                                        void *                                             user_data);

typedef void (*ggml_backend_dayan_set_slot_trace_callback_t)(ggml_backend_dayan_slot_trace_callback callback,
                                                             void *                                 user_data);

typedef void (*ggml_backend_dayan_set_slot_filter_callback_t)(ggml_backend_dayan_slot_filter_callback callback,
                                                              void *                                  user_data);

struct ggml_backend_dayan_graph_plan_slot {
    uint32_t                   version;
    const char *               kind;
    const char *               executor;
    const char *               launch_name;
    const char *               kernel;
    const char *               task_mode;
    uint32_t                   task_dim_x;
    size_t                     slot_index;
    const struct ggml_tensor * nodes[GGML_DAYAN_SLOT_TRACE_MAX_NODES];
    size_t                     node_count;
    const struct ggml_tensor * inputs[GGML_DAYAN_GRAPH_PLAN_MAX_INPUTS];
    const char *               input_roles[GGML_DAYAN_GRAPH_PLAN_MAX_INPUTS];
    size_t                     input_count;
    const struct ggml_tensor * outputs[GGML_DAYAN_SLOT_TRACE_MAX_OUTPUTS];
    size_t                     output_count;
    uint32_t                   output_mask;
    uint32_t                   sort_k;
    uint32_t                   task_dim_y;
    uint32_t                   task_dim_z;
};

typedef void (*ggml_backend_dayan_graph_plan_callback)(const struct ggml_backend_dayan_graph_plan_slot * slot,
                                                       void *                                             user_data);

typedef bool (*ggml_backend_dayan_get_graph_plan_t)(const struct ggml_cgraph *             graph,
                                                    ggml_backend_dayan_graph_plan_callback callback,
                                                    void *                                 user_data);

GGML_BACKEND_API ggml_backend_t     ggml_backend_dayan_init(void);
GGML_BACKEND_API bool               ggml_backend_is_dayan(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_dayan_reg(void);
GGML_BACKEND_API bool               ggml_backend_dayan_graph_rewrite(ggml_backend_t        backend,
                                                                     struct ggml_context * ctx,
                                                                     struct ggml_cgraph *  cgraph);
GGML_BACKEND_API void ggml_backend_dayan_set_slot_trace_callback(ggml_backend_dayan_slot_trace_callback callback,
                                                                 void *                                 user_data);
GGML_BACKEND_API void ggml_backend_dayan_set_slot_filter_callback(ggml_backend_dayan_slot_filter_callback callback,
                                                                  void *                                  user_data);
GGML_BACKEND_API bool ggml_backend_dayan_get_graph_plan(const struct ggml_cgraph *             graph,
                                                        ggml_backend_dayan_graph_plan_callback callback,
                                                        void *                                 user_data);

#ifdef __cplusplus
}
#endif
