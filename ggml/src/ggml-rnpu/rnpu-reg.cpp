#include "ggml-rnpu.h"
#include <mutex>
#include "../ggml-backend-impl.h"

#include "./common.hpp"

extern const ggml_backend_device_i ggml_backend_rnpu_device_interface;


// -----------------------------------------------------------------------------------
// ------------------- NPU backend registery external interface ----------------------
// -----------------------------------------------------------------------------------













static const ggml_backend_reg_i ggml_backend_rnpu_reg_interface = {
    NULL
    // TODO:
    // /* .get_name          = */ ggml_backend_rnpu_reg_get_name,
    // /* .get_device_count  = */ ggml_backend_rnpu_reg_get_device_count,
    // /* .get_device        = */ ggml_backend_rnpu_reg_get_device,
    // /* .get_proc_address  = */ ggml_backend_rnpu_reg_get_proc_address,
};


// --------------------------------------------------------------------------------
// ---------------------- NPU backend registery external API ----------------------
// --------------------------------------------------------------------------------

/**
 * @brief R-NPU backend registery, initialize registry when first called.
 *
 * @return Pointer to initialized registery.
 */
ggml_backend_reg_t ggml_backend_rnpu_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        if(!initialized) {
            ggml_backend_rnpu_reg_context * ctx = new ggml_backend_rnpu_reg_context;
            for (int i = 0; i < ggml_rnpu_info().device_count; ++i) {
                ggml_backend_rnpu_device_context * dev_ctx = new ggml_backend_rnpu_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_RNPU_NAME + std::to_string(i);

                // TODO: NPU property API
                // ggml_rnpu_set_device(i);
                // npuDeivceProp prop;
                // NPU_CHECK(npuGetDeviceProperties(&prop, i))
                // dev_ctx->description = prop.name;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_rnpu_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_rnpu_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}
