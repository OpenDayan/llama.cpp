# Dayan backend

The backend supports contiguous `GGML_OP_MUL_MAT` with BF16 or FP16 weights, including broadcastable `ne2` and `ne3` batches. Activations are stored with the same type as the weights, matrix accumulation uses F32, and an unfused matrix multiplication returns the F32 result required by ggml.

Before scheduling a llama graph, the backend rewrite inserts one matching BF16 or FP16 `CPY` for each unique F32 activation used by an eligible matrix multiplication. Matrix multiplications that share an activation and storage type also share the cast tensor. Direct ggml users that do not run the rewrite can still provide F32 activations; the Dayan graph executor converts each unique activation once.

The rewrite moves a cast before an exclusive metadata-only `VIEW`, `RESHAPE`, `PERMUTE`, or `TRANSPOSE` chain when this does not remove an F32 output or side consumer. It rebuilds the aliases in the low-precision type, and the graph compiler finds the MatMul producer and cast by dependency instead of node adjacency. This keeps the cast in the MatMul epilogue slot without producing both F32 and low-precision matrix outputs.

Each ggml Dayan arena has one Device allocation, and each tensor owns a Device view at its planned arena offset. Graph kernels pass these views directly to their consumers. Intermediate results remain on Device; Device-to-Host transfer occurs only through an explicit tensor read or a scheduler boundary.

URAM bypasses the CPU cache. CPU accesses to URAM need no cache flush or invalidate. LSU, MAT, and VEC dependencies still require the corresponding unit waits or Named Resources. Cache maintenance applies to cached DRAM, including shared kernel configurations.

The graph executor lowers a Dayan split to execution slots. Two adjacent matrix multiplications with the same activation use one pair slot and one activation upload. A matching `MUL_MAT + MUL_MAT + SWIGLU/GEGLU + CPY` sequence uses one fused slot. The common matrix multiplication template accepts a compile-time epilogue function pointer, evaluates the activation with VEC before the final writeback, and quantizes the result to the weight type.

The MoE path has separate kernels for `MUL_MAT_ID`, `ADD_ID`, `ARGSORT`, and `TOP_K`. The graph compiler also recognizes `MUL_MAT_ID + MUL_MAT_ID + SWIGLU/GEGLU + CPY`. This fused slot selects both expert matrices, accumulates both products in F32 URAM buffers, calls the same compile-time VEC epilogue function pointer, and uses the common low-precision writeback. Expert ID views are read with their actual row strides, and routing rows support up to 4096 entries.

Graphs containing `MUL_MAT_ID` retain F32 residual storage. Rounding the residual stream to BF16 can change close router scores and select different experts in later layers.

When an `ARGSORT` result is used only by zero-offset prefix views, the slot selects and sorts only the largest required prefix. This changes routing selection from O(E^2) to O(E*K), retains the original row stride, and preserves the sorted order required by `ggml_argsort_top_k`. Full outputs and other consumers retain full sorting. `GET_ROWS` accepts strided I32/I64 indices, including these routing views. Sorting uses independent Procs with no child Threads or Named Resources.

F32 binary operators pack strided inputs with the existing Device copy kernel before arithmetic or broadcast. This covers the expert output views used in MoE aggregation. These copies are runtime staging within the binary slot. CLAMP accepts infinite bounds, including the lower-only clamp used to normalize selected routing weights.

The four-core MatMul pipeline uses one root MAT worker and three child Threads for DRAM loads, peer copies, and output stores. Named Resources bind buffer reuse to LSU/MAT completion. A shared source has consumer credits for both its local MAT uses and the peer copy. Resource phases persist across repeated batch and expert products within a launch. LSU supports partial output rows, including the single-column shared expert gate.

For the four-core tiles, VisualSwizzle selects `M=7,B=2,S=3` for the 512-byte A/B row strides. The 384-byte C row stride uses linear storage; the 1536-byte C row stride uses `M=7,B=2,S=3`. These layouts have no per-cycle Bank conflicts and cover all 32 Banks across four consecutive blocks. SiLU keeps C linear for VEC access. These are address-mapping results, not measured RTL throughput.

Contiguous F32 residual adds and RMSNorm execute on Dayan. One parameterized normalization pipeline slot handles optional residual ADD, RMSNorm, scale multiplication, and BF16/FP16 quantization. Matrix multiplication remains a separate slot. The graph rewrite keeps every fused input live through the final logical node so the allocator cannot overlap a cast output with an input that another native Task still reads. VEC tensor dimensions expand complete 64-lane chunks into padded URAM scratch space, while LSU tail stores limit the final DRAM write to the valid byte count.

The attention operator package provides dedicated BF16 and FP16 QG, K, and V projection entry points. Q and K fuse low-precision matrix writeback, RMSNorm, RoPE or MRoPE, and output layout conversion. K and V also have indexed KV-cache variants that write directly to cache rows. Flash-Attention has native BF16 and FP16 entries, uses `dayan_uram_base()` directly, and has an explicit causal offset for KV-cache decode. It supports GQA and right-aligned causal masking. The output path fuses `attention * sigmoid(gate)`, low-precision quantization, and the output projection. The graph compiler recognizes the Qwen3.5 gate view, sigmoid, multiply, and output matrix multiplication pattern.

For Qwen3.5 linear-attention layers, `SSM_CONV` followed by SiLU uses one dedicated device function. LSU gathers the F32 channel-major input and depthwise weights into tap-major URAM tiles. VEC performs one elementwise multiply per tap, accumulates the taps in F32, and optionally applies the common SiLU implementation before one output writeback. The kernel does not construct diagonal matrices or use MAT convolution. It supports kernel widths up to 63 and uses 64-token tiles. The convolution remains a separate operator from the delta-net core.

Configure the firmware SDK explicitly:

```sh
cmake -S . -B build-dayan -DGGML_DAYAN=ON -DDAYAN_ROOT=/path/to/npu-firmware
cmake --build build-dayan --target test-backend-ops -j
```

## Test a specific operator

Run commands from the repository root. Start `dayan_server` connected to the device, cmodel, or TestCluster, then select its socket:

```sh
export DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock
```

`test-backend-ops` selects existing test cases by operator name and parameters:

| Argument | Meaning |
| --- | --- |
| `test` | Compare results with the CPU reference; this is the default mode. |
| `-b DAYAN` | Select the Dayan backend. |
| `-o ROPE` | Select an operator by its `ggml_op_desc()` name, without the `GGML_OP_` prefix. A comma-separated list such as `ADD,MUL` selects several operators. |
| `-p '<regex>'` | Filter the printed parameter string, including dtype, shape, and operator options. This selects existing cases; it does not create a new shape. |
| `-j 1` | Run one test worker, which is the default. Keep this setting for TestCluster and run one client at a time. |
| `--list-ops` | List GGML operator names without connecting to a backend. This is not a list of operations supported by Dayan. |

List operator names, then run one small F32 NEOX RoPE case with shape `[64,1,2,1]`:

```sh
./build-dayan/bin/test-backend-ops --list-ops
./build-dayan/bin/test-backend-ops test -b DAYAN -o ROPE -p '^type=f32,ne_a=\[64,1,2,1\],n_dims=64,mode=2,n_ctx=512,fs=1\.000000,ef=0\.000000,af=1\.000000,ff=0,v=0,inplace=0$'
```

Use single quotes around the regex so the shell preserves backslashes. Escape literal brackets as `\[` and `\]`. Omitting `-p` runs all registered variants of the selected operator, which can take a long time in RTL simulation. Check that the output contains the intended case and an `OK` result; an unsupported case or a filter matching no cases does not validate the operator.

For a matrix multiplication smoke case:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-backend-ops -b DAYAN -o MUL_MAT -p 'type_a=bf16,type_b=f32,m=16,n=1,k=256,bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\]'
```

Enable the cores required by the selected kernel in `dayan_server`; four-core MatMul paths need core mask `0xf`. The same client commands work with a TestCluster-backed server.

Replace `test` with `support` to check whether Dayan supports the selected cases without running their numerical comparison. `perf` runs the performance case set, which has different shapes and may require a different `-p` filter; host timings under simulation are not hardware throughput measurements.

## Dedicated tests

The standalone BF16 matrix multiplication smoke test defaults to `M=1`, `N=8064`, and `K=1024`. Pass three dimensions to override the default shape:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-matmul
```

The focused graph test checks all BF16/FP16 SWIGLU/GEGLU fused matrix multiplication variants, including both matrix producer paths, low-precision output quantization, the following matrix multiplication, and the Qwen3.5 0.8B `M=8, N=3584, K=1024` FFN shape:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-graph
```

The same test also checks the Qwen3.5 attention output gate plus projection pattern, fused RMSNorm matrix multiplication paths, residual adds, and the fused SSM convolution plus SiLU path.

Pass a selector to run only one graph test group:

```sh
cmake --build build-dayan --target test-dayan-graph -j
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-graph rope
```

| Selector | Test group |
| --- | --- |
| `rope` | NORMAL, NEOX, MROPE, and IMROPE |
| `ssm-conv` | SSM convolution and fused SiLU |
| `gated-output` | Gated attention output projection |
| `copy` | Device copy paths |
| `get-rows` | Indexed row reads |
| `embedding-norm` | Embedding lookup and normalization |
| `memory` | Memory and layout operators |

The standalone test executables below have matching CMake build targets, for example `cmake --build build-dayan --target test-dayan-attention-projection -j`.

The attention projection test runs QG, K, and V projection, indexed KV-cache writes, causal Flash-Attention, gate fusion, low-precision quantization, and output projection as one numerical data flow for BF16 and FP16 model weights:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-attention-projection
```

The standalone Flash-Attention test checks GQA with a right-aligned causal mask where `q_len < kv_len`:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-flash-attention
```

The Qwen3.5 dense test builds the minimum four-layer hybrid cycle with three gated DeltaNet layers and one causal full-attention layer. Every layer also contains RMSNorm, dense SWIGLU FFN, and residual paths. It compares Dayan outputs and final logits against a CPU graph with the same low-precision boundaries. Gated Delta Net uses a 128-element head so the test exercises the device kernel instead of CPU fallback:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-qwen35-dense
```

The MoE smoke test covers 512 experts, a narrow `ARGSORT` view, `TOP_K`, fused BF16 SWIGLU expert projections, the down expert projection, `ADD_ID`, and expert-weighted aggregation:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-moe
```

The single-core matrix kernel uses one consumer slice. `MUL_MAT_ID` still requires the expert output width to be a multiple of 16. Arbitrary attention masks, attention sinks, and Flash-Attention soft caps are not implemented. Flash-Attention has native BF16 and FP16 entries and requires Q/K/V to share one storage type. Full-model inference requires enough Device memory for all persistent weights and graph arenas; `run_cmodel_server.sh` defaults to a 4096 MiB memory window. A llama model using Dayan rejects any non-metadata model graph node assigned to another backend. Sampling stays on CPU and is not inserted into the Dayan graph.
