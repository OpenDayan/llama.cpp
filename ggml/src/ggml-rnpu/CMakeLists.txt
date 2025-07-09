if(NOT DEFINED RNPU_RT_PATH)
    message(FATAL_ERROR "Missing RNPU_RT_PATH!\n"
            "Solution: Set it via -DRNPU_RT_PATH=/your/path when running cmake")
else()
    message(STATUS "Using R-NPU runtime at: ${RNPU_RT_PATH}")
endif()

set(RNPU_INCLUDE_DIRS
    ${RNPU_RT_PATH}/include
)

file(GLOB GGML_SOURCES_RNPU "*.cpp")
ggml_add_backend_library(ggml-rnpu ${GGML_SOURCES_RNPU})

target_include_directories(ggml-rnpu PRIVATE ${RNPU_INCLUDE_DIRS})

message(STATUS "R-NPU: RNPU_INCLUDE_DIRS = ${RNPU_INCLUDE_DIRS}")
