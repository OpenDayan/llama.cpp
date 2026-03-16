#include "ggml-rnpu.h"
#include <mutex>

#include "./runtime-wrapper.hpp"

#include "../ggml-impl.h"
#include "../ggml-backend-impl.h"

#include "./common.hpp"
#include "ggml.h"

#include "NPURuntime.h"
#include "NPUDriver.h"

// -----------------------------------------------------------------------------------
// ----------------------- NPU backend registery interface ---------------------------
// -----------------------------------------------------------------------------------

/**
 * @brief Retrieve the name of a given registery.
 *
 * This function always returns GGML_RNPU_NAME.
 *
 * @param reg The given registery. (Unused)
 * @return Pointer to C-style string name.
 */
static const char * ggml_backend_rnpu_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_RNPU_NAME;
}

/**
 * @brief Get the number of registered NPU devices in a given registery.
 *
 * @param reg The given registery.
 * @return The number of NPU devices in the registery
 */
static size_t ggml_backend_rnpu_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rnpu_reg_context * ctx = (ggml_backend_rnpu_reg_context *)reg->context;
    return ctx->devices.size();
}

/**
 * @brief Retrieve a specified NPU device in a given registery.
 *
 * @param reg The given registery.
 * @param index The NPU device index to be retrieved.
 * @return Pointer to retrieved NPU device
 */
static ggml_backend_dev_t ggml_backend_rnpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rnpu_reg_context * ctx = (ggml_backend_rnpu_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

/**
 * @brief Reflects a R-NPU function based on a given name.
 *
 * This function is reserved for future use.
 *
 * @param reg The given registery. (Unused)
 * @param name The name of specified GGML backend function name. (Unused)
 * @return {text}
 */
static void * ggml_backend_rnpu_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_rnpu_reg_interface = {
    /* .get_name          = */ ggml_backend_rnpu_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rnpu_reg_get_device_count,
    /* .get_device        = */ ggml_backend_rnpu_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rnpu_reg_get_proc_address,
};



// --------------------------------------------------------------------------------
// ---------------------- NPU backend registery external API ----------------------
// --------------------------------------------------------------------------------

extern struct NPURuntimeState runtime;
extern struct NPUDeviceKernelSetDesc __bin_kernels_pack;

/**
 * @brief R-NPU backend registery, initialize registry when first called.
 *
 * @return Pointer to initialized registery.
 */
ggml_backend_reg_t ggml_backend_rnpu_reg() {
    static ggml_backend_reg reg;
    static ggml_backend_reg_t reg_ptr = nullptr;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        if(!initialized) {

            // ======= NPU Runtime Init =======
            // FIXME: NPU runtime initialize
            GGML_LOG_INFO("[INFO] backend init runtime\n");
            ssize_t err;
            err = npu_runtime_init(&runtime);
            if (err != NPU_SUCCESS) {
                GGML_LOG_WARN("%s: failed to initialize Runtime: %ld, RNPU backend disabled\n", __func__, err);
                initialized = true;
                return nullptr;
            }
            GGML_LOG_INFO("Runtime initialized successfully.\n");

            // FIXME: NPU load kernel sets
            GGML_LOG_INFO("[INFO] backend load kernelset\n");
            err = npu_runtime_load_kernelset(&runtime, &__bin_kernels_pack);
            if (err != NPU_SUCCESS) {
                npu_runtime_deinit(&runtime);
                GGML_LOG_WARN("%s: failed to load kernel set: %ld, RNPU backend disabled\n", __func__, err);
                initialized = true;
                return nullptr;
            }
            GGML_LOG_INFO("Kernel set loaded successfully.\n");

            ggml_backend_rnpu_reg_context * ctx = new ggml_backend_rnpu_reg_context;
            const int n_devices = ggml_rnpu_info().device_count;
            if (n_devices <= 0) {
                GGML_LOG_WARN("%s: no RNPU devices available, backend disabled\n", __func__);
                delete ctx;
                npu_runtime_deinit(&runtime);
                initialized = true;
                return nullptr;
            }

            for (int i = 0; i < n_devices; ++i) {
                ggml_backend_rnpu_device_context * dev_ctx = new ggml_backend_rnpu_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_RNPU_NAME + std::to_string(i);

                ggml_rnpu_set_device(i);
                npuDeviceProp prop;
                npuErr_t prop_err = npuGetDeviceProperties(&prop, i);
                if (prop_err != npuSuccess) {
                    GGML_LOG_WARN("%s: failed to query RNPU device %d properties: %s\n",
                        __func__, i, npuGetErrorString(prop_err));
                    delete dev_ctx;
                    continue;
                }
                dev_ctx->description = prop.name;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_rnpu_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            if (ctx->devices.empty()) {
                GGML_LOG_WARN("%s: no usable RNPU devices after probing, backend disabled\n", __func__);
                delete ctx;
                npu_runtime_deinit(&runtime);
                initialized = true;
                return nullptr;
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_rnpu_reg_interface,
                /* .context     = */ ctx
            };

            reg_ptr = &reg;
        }

        initialized = true;
    }

    return reg_ptr;
}
