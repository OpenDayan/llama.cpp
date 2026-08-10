if (NOT CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(FATAL_ERROR "ggml-dayan currently requires Linux")
endif()

if (NOT DAYAN_ROOT AND DEFINED ENV{DAYAN_ROOT})
    set(DAYAN_ROOT "$ENV{DAYAN_ROOT}")
endif()

if (NOT DAYAN_NPU_KERNELS_DIR AND DEFINED ENV{DAYAN_NPU_KERNELS_DIR})
    set(DAYAN_NPU_KERNELS_DIR "$ENV{DAYAN_NPU_KERNELS_DIR}")
endif()

find_program(DAYAN_DYCC
    NAMES dycc
    HINTS
        "${DAYAN_ROOT}/host/dycc"
        "${DAYAN_ROOT}/bin")

find_path(DAYAN_INCLUDE_DIR
    NAMES dayan_runtime.h
    HINTS
        "${DAYAN_ROOT}/host/include"
        "${DAYAN_ROOT}/include")

find_file(DAYAN_RUNTIME_LIBRARY
    NAMES libDayanRuntime.a
    HINTS
        "${DAYAN_ROOT}/build/host/driver"
        "${DAYAN_ROOT}/lib"
        "${DAYAN_ROOT}/lib64")

find_path(DAYAN_NPU_KERNEL_INCLUDE_DIR
    NAMES npu_instruction.hpp
    HINTS
        "${DAYAN_NPU_KERNELS_DIR}/include"
        "${DAYAN_ROOT}/../npu-kernels/include"
        "${DAYAN_ROOT}/lib/dayan")

if (NOT DAYAN_DYCC)
    message(FATAL_ERROR "ggml-dayan: dycc not found; set DAYAN_ROOT")
endif()
if (NOT DAYAN_INCLUDE_DIR)
    message(FATAL_ERROR "ggml-dayan: Dayan host headers not found; set DAYAN_ROOT")
endif()
if (NOT DAYAN_RUNTIME_LIBRARY)
    message(FATAL_ERROR "ggml-dayan: libDayanRuntime.a not found; build the Dayan host driver or set DAYAN_ROOT")
endif()
if (NOT DAYAN_NPU_KERNEL_INCLUDE_DIR)
    message(FATAL_ERROR "ggml-dayan: npu_instruction.hpp not found; set DAYAN_NPU_KERNELS_DIR")
endif()
message(STATUS "ggml-dayan: dycc: ${DAYAN_DYCC}")
message(STATUS "ggml-dayan: runtime: ${DAYAN_RUNTIME_LIBRARY}")
message(STATUS "ggml-dayan: NPU headers: ${DAYAN_NPU_KERNEL_INCLUDE_DIR}")

set(DAYAN_KERNEL_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/ggml-dayan-kernels.o")

add_custom_command(
    OUTPUT "${DAYAN_KERNEL_OBJECT}"
    COMMAND "${DAYAN_DYCC}"
        -O2 -Wall -Wextra -Werror -Wno-error=format -Wno-error=unused-parameter -Wno-error=unused-variable
        "-I${CMAKE_CURRENT_SOURCE_DIR}"
        "-I${DAYAN_NPU_KERNEL_INCLUDE_DIR}"
        -Xhost -fPIC
        -Xdevice "-I${CMAKE_CURRENT_SOURCE_DIR}/device_include"
        -c "${CMAKE_CURRENT_SOURCE_DIR}/ggml-dayan-kernels.cpp"
        -o "${DAYAN_KERNEL_OBJECT}"
    DEPENDS
        ggml-dayan-kernels.cpp
        ggml-dayan-kernels.h
        ggml-dayan-attention.h
        ggml-dayan-conv.h
        ggml-dayan-elementwise.h
        ggml-dayan-flash-attention.h
        ggml-dayan-gated-delta-net.h
        ggml-dayan-layout.h
        ggml-dayan-matmul.h
        ggml-dayan-memory.h
        ggml-dayan-normalization.h
        ggml-dayan-rope.h
        ggml-dayan-softmax.h
        device_include/cstdint
        device_include/cstdio
        device_include/type_traits
        "${DAYAN_NPU_KERNEL_INCLUDE_DIR}/npu_instruction.hpp"
        "${DAYAN_NPU_KERNEL_INCLUDE_DIR}/npu_lsu_csr.hpp"
        "${DAYAN_NPU_KERNEL_INCLUDE_DIR}/npu_macro_csr.hpp"
        "${DAYAN_NPU_KERNEL_INCLUDE_DIR}/npu_mat_csr.hpp"
        "${DAYAN_NPU_KERNEL_INCLUDE_DIR}/npu_task.hpp"
        "${DAYAN_NPU_KERNEL_INCLUDE_DIR}/npu_kernel.h"
    COMMENT "Compiling Dayan kernels"
    VERBATIM)

set_source_files_properties("${DAYAN_KERNEL_OBJECT}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)

ggml_add_backend_library(ggml-dayan
    ggml-dayan.cpp
    ggml-dayan-graph.cpp
    ggml-dayan-graph.h
    ggml-dayan-kernels.h
    ggml-dayan-attention.h
    ggml-dayan-conv.h
    ggml-dayan-elementwise.h
    ggml-dayan-flash-attention.h
    ggml-dayan-gated-delta-net.h
    ggml-dayan-layout.h
    ggml-dayan-matmul.h
    ggml-dayan-memory.h
    ggml-dayan-normalization.h
    ggml-dayan-rope.h
    ggml-dayan-softmax.h
    "${DAYAN_KERNEL_OBJECT}")

find_package(Threads REQUIRED)
target_include_directories(ggml-dayan PRIVATE "${DAYAN_INCLUDE_DIR}")
target_link_libraries(ggml-dayan PRIVATE "${DAYAN_RUNTIME_LIBRARY}" Threads::Threads)
