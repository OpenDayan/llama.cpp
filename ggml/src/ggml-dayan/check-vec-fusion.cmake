if (NOT DEFINED DAYAN_SOURCE_DIR)
    message(FATAL_ERROR "DAYAN_SOURCE_DIR is required")
endif()

file(GLOB DAYAN_OPERATOR_SOURCES
    "${DAYAN_SOURCE_DIR}/*.h"
    "${DAYAN_SOURCE_DIR}/*.cpp")

set(DAYAN_FORBIDDEN_VEC_PATTERNS
    "issue_f32_"
    "f32_scalar_"
    "ggml_dayan_vec_detail::f32_(add|sub|mul|div|max|min|sfu|sqrt|reduce)"
    "static[ \t]+inline[ \t]+void[ \t]+f32_(add|sub|mul|div|max|min|sfu|sqrt|reduce)")

foreach(SOURCE_FILE IN LISTS DAYAN_OPERATOR_SOURCES)
    file(READ "${SOURCE_FILE}" SOURCE_TEXT)
    foreach(FORBIDDEN_PATTERN IN LISTS DAYAN_FORBIDDEN_VEC_PATTERNS)
        if (SOURCE_TEXT MATCHES "${FORBIDDEN_PATTERN}")
            message(FATAL_ERROR
                "${SOURCE_FILE} contains forbidden generic VEC helper pattern: ${FORBIDDEN_PATTERN}. "
                "Emit the complete fused instruction stream in the owning operator block instead.")
        endif()
    endforeach()
endforeach()
