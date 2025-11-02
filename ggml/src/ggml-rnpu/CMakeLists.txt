if(NOT DEFINED RNPU_RT_PATH)
    message(FATAL_ERROR "Missing RNPU_RT_PATH!\n"
            "Solution: Set it via -DRNPU_RT_PATH=/your/path when running cmake")
else()
    message(STATUS "Using R-NPU runtime at: ${RNPU_RT_PATH}")
endif()

if(NOT DEFINED RNPU_KERNEL_PATH)
    message(FATAL_ERROR "Missing RNPU_KERNEL_PATH!\n"
            "Solution: Set it via -DRNPU_KERNEL_PATH=/your/path when running cmake")
else()
    message(STATUS "Using R-NPU kernels at: ${RNPU_KERNEL_PATH}")
endif()

set(RNPU_INCLUDE_DIRS
    ${RNPU_RT_PATH}/include
)

option(USE_FPGA "Enable FPGA mode" OFF)

file(GLOB  GGML_SOURCES_RNPU "*.cpp")

file(GLOB  RT_C_SRCS "${RNPU_RT_PATH}/src/*.c")
file(GLOB  RT_CXX_SRCS "${RNPU_RT_PATH}/src/*.cpp")
list(APPEND RT_SRCS ${RT_C_SRCS} ${RT_CXX_SRCS})
message(STATUS "R-NPU: RT_SRCS = ${RT_SRCS}")
list(APPEND GGML_SOURCES_RNPU ${RT_SRCS})

file(GLOB  KERNEL_SRCS "${RNPU_KERNEL_PATH}/general_entry_fpga.packed.c")
message(STATUS "R-NPU: KERNSL_SRC = ${KERNEL_SRCS}")
list(APPEND GGML_SOURCES_RNPU ${KERNEL_SRCS})

ggml_add_backend_library(ggml-rnpu ${GGML_SOURCES_RNPU})

target_include_directories(ggml-rnpu PRIVATE ${RNPU_INCLUDE_DIRS})

if(USE_FPGA)
    target_compile_definitions(ggml-rnpu PRIVATE USE_FPGA)
    message(STATUS "Using FPGA mode")
endif()

message(STATUS "R-NPU: RNPU_INCLUDE_DIRS = ${RNPU_INCLUDE_DIRS}")
