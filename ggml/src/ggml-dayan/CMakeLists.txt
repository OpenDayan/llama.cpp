if (NOT CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(FATAL_ERROR "ggml-dayan currently requires Linux")
endif()

if (NOT DAYAN_ROOT AND DEFINED ENV{DAYAN_ROOT})
    set(DAYAN_ROOT "$ENV{DAYAN_ROOT}")
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

find_path(DAYAN_DEVICE_INCLUDE_DIR
    NAMES dayan_intrinsics.h
    HINTS
        "${DAYAN_ROOT}/device/include/api"
        "${DAYAN_ROOT}/include")

find_file(DAYAN_RUNTIME_LIBRARY
    NAMES libDayanRuntime.a
    HINTS
        "${DAYAN_ROOT}/build/host/driver"
        "${DAYAN_ROOT}/lib"
        "${DAYAN_ROOT}/lib64")

if (NOT DAYAN_DYCC)
    message(FATAL_ERROR "ggml-dayan: dycc not found; set DAYAN_ROOT")
endif()
if (NOT DAYAN_INCLUDE_DIR)
    message(FATAL_ERROR "ggml-dayan: Dayan host headers not found; set DAYAN_ROOT")
endif()
if (NOT DAYAN_DEVICE_INCLUDE_DIR)
    message(FATAL_ERROR "ggml-dayan: Dayan Device API headers not found; set DAYAN_ROOT")
endif()
if (NOT DAYAN_RUNTIME_LIBRARY)
    message(FATAL_ERROR "ggml-dayan: libDayanRuntime.a not found; build the Dayan host driver or set DAYAN_ROOT")
endif()
message(STATUS "ggml-dayan: dycc: ${DAYAN_DYCC}")
message(STATUS "ggml-dayan: runtime: ${DAYAN_RUNTIME_LIBRARY}")
message(STATUS "ggml-dayan: Device API: ${DAYAN_DEVICE_INCLUDE_DIR}")

set(DAYAN_KERNEL_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/ggml-dayan-kernels.o")
set(DAYAN_MEGA_EXECUTOR_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/ggml-dayan-mega-executor.o")

add_custom_command(
    OUTPUT "${DAYAN_KERNEL_OBJECT}"
    COMMAND "${CMAKE_COMMAND}"
        "-DDAYAN_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/check-vec-fusion.cmake"
    COMMAND "${DAYAN_DYCC}"
        -O2 -Wall -Wextra -Werror -Wno-error=format -Wno-error=unused-parameter -Wno-error=unused-variable
        "-I${CMAKE_CURRENT_SOURCE_DIR}"
        -Xhost -fPIC
        -Xdevice "-I${CMAKE_CURRENT_SOURCE_DIR}/device_include"
        -c "${CMAKE_CURRENT_SOURCE_DIR}/ggml-dayan-kernels.cpp"
        -o "${DAYAN_KERNEL_OBJECT}"
    DEPENDS
        ggml-dayan-kernels.cpp
        ggml-dayan-kernels.h
        ggml-dayan-activation.h
        ggml-dayan-common.h
        ggml-dayan-attention.h
        ggml-dayan-conv.h
        ggml-dayan-elementwise.h
        ggml-dayan-ffn.h
        ggml-dayan-flash-attention.h
        ggml-dayan-gated-delta-net.h
        ggml-dayan-layout.h
        ggml-dayan-matmul.h
        ggml-dayan-matmul-4c.h
        ggml-dayan-mxfp4.h
        ggml-dayan-mxfp4-layout.h
        ggml-dayan-memory.h
        ggml-dayan-moe.h
        ggml-dayan-normalization.h
        ggml-dayan-rope.h
        ggml-dayan-softmax.h
        device_include/cstdint
        device_include/cstdio
        device_include/type_traits
        check-vec-fusion.cmake
        "${DAYAN_DEVICE_INCLUDE_DIR}/dayan_attrs.h"
        "${DAYAN_DEVICE_INCLUDE_DIR}/dayan_intrinsics.h"
        "${DAYAN_DEVICE_INCLUDE_DIR}/dayan_list.h"
        "${DAYAN_DEVICE_INCLUDE_DIR}/dayan_sys.h"
        "${DAYAN_INCLUDE_DIR}/dayan_kernel.h"
    COMMENT "Compiling Dayan kernels"
    VERBATIM)

add_custom_command(
    OUTPUT "${DAYAN_MEGA_EXECUTOR_OBJECT}"
    COMMAND "${DAYAN_DYCC}"
        -O2 -Wall -Wextra -Werror -Wno-error=format -Wno-error=unused-parameter -Wno-error=unused-variable
        -Wno-error=unused-function -Wno-error=unneeded-internal-declaration
        "-I${CMAKE_CURRENT_SOURCE_DIR}"
        -Xhost -fPIC
        -Xdevice "-I${CMAKE_CURRENT_SOURCE_DIR}/device_include"
        -c "${CMAKE_CURRENT_SOURCE_DIR}/ggml-dayan-mega-executor.cpp"
        -o "${DAYAN_MEGA_EXECUTOR_OBJECT}"
    DEPENDS
        ggml-dayan-mega-executor.cpp
        ggml-dayan-mega-executor.h
        mega-graph/qwen35-dense-bf16.hpp
        mega-graph/qwen35-dense-mxfp4.hpp
        ggml-dayan-kernels.h
        ggml-dayan-activation.h
        ggml-dayan-attention.h
        ggml-dayan-common.h
        ggml-dayan-conv.h
        ggml-dayan-elementwise.h
        ggml-dayan-flash-attention.h
        ggml-dayan-gated-delta-net.h
        ggml-dayan-layout.h
        ggml-dayan-matmul.h
        ggml-dayan-matmul-4c.h
        ggml-dayan-ffn.h
        ggml-dayan-mxfp4.h
        ggml-dayan-mxfp4-layout.h
        ggml-dayan-memory.h
        ggml-dayan-normalization.h
        ggml-dayan-rope.h
        device_include/cstdint
        device_include/cstdio
        device_include/type_traits
        "${DAYAN_DEVICE_INCLUDE_DIR}/dayan_attrs.h"
        "${DAYAN_DEVICE_INCLUDE_DIR}/dayan_intrinsics.h"
        "${DAYAN_DEVICE_INCLUDE_DIR}/dayan_list.h"
        "${DAYAN_DEVICE_INCLUDE_DIR}/dayan_sys.h"
        "${DAYAN_INCLUDE_DIR}/dayan_kernel.h"
    COMMENT "Compiling Dayan mega executor"
    VERBATIM)

set_source_files_properties("${DAYAN_KERNEL_OBJECT}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
set_source_files_properties("${DAYAN_MEGA_EXECUTOR_OBJECT}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)

ggml_add_backend_library(ggml-dayan
    ggml-dayan.cpp
    ggml-dayan-graph.cpp
    ggml-dayan-graph.h
    ggml-dayan-mega.cpp
    ggml-dayan-mega.h
    ggml-dayan-mega-executor.h
    mega-graph/qwen35-dense-bf16.hpp
    mega-graph/qwen35-dense-mxfp4.hpp
    ggml-dayan-kernels.h
    ggml-dayan-activation.h
    ggml-dayan-common.h
    ggml-dayan-attention.h
    ggml-dayan-conv.h
    ggml-dayan-elementwise.h
    ggml-dayan-ffn.h
    ggml-dayan-flash-attention.h
    ggml-dayan-gated-delta-net.h
    ggml-dayan-layout.h
    ggml-dayan-matmul.h
    ggml-dayan-matmul-4c.h
    ggml-dayan-mxfp4.h
    ggml-dayan-mxfp4-layout.h
    ggml-dayan-memory.h
    ggml-dayan-moe.h
    ggml-dayan-normalization.h
    ggml-dayan-rope.h
    ggml-dayan-softmax.h
    "${DAYAN_KERNEL_OBJECT}"
    "${DAYAN_MEGA_EXECUTOR_OBJECT}")

find_package(Threads REQUIRED)
target_include_directories(ggml-dayan PRIVATE "${DAYAN_INCLUDE_DIR}")
target_link_libraries(ggml-dayan PRIVATE "${DAYAN_RUNTIME_LIBRARY}" Threads::Threads)
