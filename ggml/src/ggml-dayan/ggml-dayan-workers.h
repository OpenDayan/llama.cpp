#pragma once

#if defined(__DAYAN_DEVICE__)

#include <dayan_sys.h>

namespace ggml_dayan_matmul_detail {

struct NativeContext {
    uint64_t uram = 0;
    uint64_t bytes = 0;
    uint32_t task = 0;
    uint32_t tasks = 0;
};

struct NativeWorkerPlan {
    uint64_t elements;
    uint64_t units;
    uint32_t bytes;
    uint8_t lsu = 2;
    uint8_t mat = 0;
    uint8_t vec = 1;
};

template <typename Function> struct NativeWorker {
    const Function * function;
    NativeContext context;

    static void entry(void * argument) {
        const auto & worker = *(const NativeWorker *) argument;
        (*worker.function)(worker.context);
    }
};

template <typename Function>
static bool native_dispatch(const NativeContext & context, NativeWorkerPlan plan, const Function & function) {
    if (context.tasks != 0) { return false; }
    const uint32_t task = dayan_task_id(DAYAN_TASK_X), tasks = dayan_task_dim(DAYAN_TASK_X);
    const uint64_t uram = (uintptr_t) dayan_uram_base();
    if (!uram || !tasks || task >= tasks || plan.bytes > dayan_uram_size()) { dayan_thread_exit(-22); }
    if (plan.elements < (uint64_t) tasks * 2 * 32768 || plan.units < tasks * 2 || 2ULL * plan.bytes > dayan_uram_size()) {
        function(NativeContext{uram, plan.bytes, task, tasks});
        return true;
    }

    dayan_thread_resources_t resources = {};
    resources.lsu_tcsr_count = plan.lsu;
    resources.mat_tcsr_count = plan.mat;
    resources.vec_tcsr_count = plan.vec;
    NativeWorker<Function> workers[2] = {
        {&function, {uram, plan.bytes, task * 2, tasks * 2}},
        {&function, {uram + plan.bytes, plan.bytes, task * 2 + 1, tasks * 2}},
    };
    int threads[2] = {}, status = 0, created = 0;
    for (int i = 0; i < 2; ++i) {
        const int thread = dayan_create_thread(i == 0 ? "worker-0" : "worker-1", NativeWorker<Function>::entry, &workers[i], &resources);
        if (thread < 0) { status = thread; break; }
        threads[created++] = thread;
    }
    for (int i = 0; i < created; ++i) {
        int64_t result = 0;
        const int joined = dayan_thread_join(threads[i], &result);
        if (status == 0 && joined < 0) { status = joined; }
        if (status == 0 && result < 0) { status = (int) result; }
    }
    if (status < 0) { dayan_thread_exit(status); }
    return true;
}

static uint64_t native_uram(const NativeContext & context) {
    return context.tasks ? context.uram : (uintptr_t) dayan_uram_base();
}

static uint64_t native_uram_size(const NativeContext & context) {
    return context.tasks ? context.bytes : dayan_uram_size();
}

} // namespace ggml_dayan_matmul_detail

#endif
