# Dayan source map

Use this map to load only the source needed for a task. Search by symbol because line numbers change.

## Framework integration

| Topic | File and symbols |
| --- | --- |
| Backend rewrite hook | `src/llama-model.cpp`: `llama_model_apply_backend_graph_rewrites()`, `llama_model::build_graph()` |
| Dayan graph assignment | `src/llama-context.cpp`: `llama_graph_assign_dayan()` |
| No-fallback validation | `src/llama-context.cpp`: `llama_graph_is_dayan_only()` |
| Qwen3.5 graph | `src/models/qwen35.cpp`: graph constructor and `build_layer_*()` methods |
| Public backend declarations | `ggml/include/ggml-dayan.h` |

## Backend Host code

| Topic | File and symbols |
| --- | --- |
| Runtime/Context/Stream ownership | `ggml-dayan.cpp`: `ggml_backend_dayan_device_state`, acquire/release helpers |
| Arena and tensor Device views | `ggml-dayan.cpp`: buffer type allocation, `ggml_backend_dayan_buffer_init_tensor()` |
| Slot resource cache | `ggml-dayan.cpp`: `ggml_backend_dayan_slot_resources`, prepare/upload helpers |
| Activation conversion cache | `ggml-dayan.cpp`: `ggml_backend_dayan_prepare_activation()` |
| Slot execution | `ggml-dayan.cpp`: `ggml_backend_dayan_graph_compute()` |
| Device support switch | `ggml-dayan.cpp`: `ggml_backend_dayan_device_supports_op()` |
| Registry rewrite export | `ggml-dayan.cpp`: `ggml_backend_dayan_reg_get_proc_address()` |

## Graph optimization

| Topic | File and symbols |
| --- | --- |
| Slot types and plan | `ggml-dayan-graph.h` |
| Per-op contracts | `ggml-dayan-graph.cpp`: `ggml_backend_dayan_supports_*()` |
| Flash mask reduction | `ggml-dayan-graph.cpp`: `ggml_backend_dayan_flash_attention_lengths()` |
| Cast hoist analysis | `ggml-dayan-graph.cpp`: `ggml_backend_dayan_find_cast_hoist()` |
| Cast alias reconstruction | `ggml-dayan-graph.cpp`: `ggml_backend_dayan_apply_cast_hoist()` |
| Lifetime dependencies | `ggml-dayan-graph.cpp`: `ggml_backend_dayan_preserve_norm_pipeline_inputs()` |
| Graph mutation | `ggml-dayan-graph.cpp`: `ggml_backend_dayan_graph_rewrite()` |
| Slot fusion | `ggml-dayan-graph.cpp`: `ggml_backend_dayan_compile_graph()` |

## Kernel ABI and entries

| Topic | File |
| --- | --- |
| Config structs and wrappers | `ggml-dayan-kernels.h` |
| External kernel entries and descriptors | `ggml-dayan-kernels.cpp` |
| Parameter-check macros | `ggml-dayan-common.h` |
| Build and dycc invocation | `ggml/src/ggml-dayan/CMakeLists.txt` |

## Device implementation by operator

| Operator family | File |
| --- | --- |
| VEC activation helpers | `ggml-dayan-activation.h` |
| Q/K/V and gated projection epilogues | `ggml-dayan-attention.h` |
| SSM convolution | `ggml-dayan-conv.h` |
| Elementwise, reductions, native task helpers | `ggml-dayan-elementwise.h` |
| FlashAttention | `ggml-dayan-flash-attention.h` |
| Gated Delta Net | `ggml-dayan-gated-delta-net.h` |
| Layout operations | `ggml-dayan-layout.h` |
| MatMul and fused GLU | `ggml-dayan-matmul.h` |
| Copy, rows, repeat, set | `ggml-dayan-memory.h` |
| MoE routing helpers | `ggml-dayan-moe.h` |
| RMSNorm, norm, group norm, pipeline | `ggml-dayan-normalization.h` |
| RoPE and MRoPE | `ggml-dayan-rope.h` |
| Softmax | `ggml-dayan-softmax.h` |

## Firmware references

Read the matching public firmware headers before changing low-level work:

- `${DAYAN_ROOT}/device/include/api/dayan_sys.h`
- `${DAYAN_ROOT}/device/include/api/dayan_intrinsics.h`
- `${DAYAN_ROOT}/demo/matmul_bf16_1c.cpp`
- `${DAYAN_ROOT}/demo/matmul_bf16_4c.cpp` for cross-Core sharing

Do not use a different kernel repository as the API reference.

## Test selection by changed file

| Changed area | Minimum focused tests |
| --- | --- |
| `ggml-dayan-graph.*` | `test-dayan-graph`, affected operator test, `test-dayan-qwen35-dense` for production patterns |
| `ggml-dayan-matmul.h` | `test-dayan-matmul`, `test-dayan-graph`, dense model test |
| `ggml-dayan-flash-attention.h` | FlashAttention, attention projection, dense model test, decode replay |
| `ggml-dayan-attention.h` | Attention projection, FlashAttention if layout changes, dense model test |
| `ggml-dayan-gated-delta-net.h` | Gated Delta Net, dense model test |
| `ggml-dayan-conv.h` | `test-dayan-graph ssm-conv`, full graph suite, dense model test |
| Common elementwise/memory/layout helpers | Focused selector, full graph suite, decode replay when a hot signature changes |
| Host buffer or slot resource code | Full graph suite, affected pipelines, dense model test, trace replay |

## Performance tooling

- `tools/dayan-trace/dayan-trace.cpp`: signature creation, unique filtering, cold/warm collection, replay validation, and report fields.
- `tools/dayan-trace/README.md`: supported commands and limitations.
- `tests/test-dayan-hotspots.cpp`: bounded workloads for profiler inspection.
- `BUILD_NOTE.md`: historical measurements and investigation notes. Treat values as history, not current truth.
