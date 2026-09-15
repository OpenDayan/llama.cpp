set(TARGET llama-dayan-trace)
add_executable(${TARGET} dayan-trace.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

add_executable(llama-dayan-operator-bench operator-bench.cpp)
target_include_directories(llama-dayan-operator-bench PRIVATE
    "${CMAKE_SOURCE_DIR}/ggml/src/ggml-dayan"
    "${DAYAN_INCLUDE_DIR}"
    "${CMAKE_SOURCE_DIR}/vendor")
target_link_libraries(llama-dayan-operator-bench PRIVATE ggml-dayan ggml)
target_compile_features(llama-dayan-operator-bench PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} llama-dayan-operator-bench RUNTIME)
endif()
