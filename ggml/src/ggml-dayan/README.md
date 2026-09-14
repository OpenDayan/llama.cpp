# Dayan backend

BF16, FP16, and MXFP4 weights use the same `ggml-dayan` backend and kernel module. Enabling `GGML_DAYAN` compiles all three paths together; runtime dispatch selects operators by tensor type. Switching between these GGUF weight formats requires no format-specific CMake flags or rebuild.

## Build

Build on Linux with the Dayan SDK's `dycc`, host and Device API headers, and `libDayanRuntime.a` available. Set `DAYAN_ROOT` to the SDK checkout or installation prefix. For a source checkout, build the host dependencies once:

```sh
export DAYAN_ROOT="$HOME/npu-firmware"
make -C "$DAYAN_ROOT" -j6 host-driver dycc
```

From the llama.cpp repository root, configure one build directory for all supported weight formats:

```sh
cmake -S . -B build-dayan \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_DAYAN=ON \
    -DDAYAN_ROOT="$DAYAN_ROOT" \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_SERVER=ON
cmake --build build-dayan -j6
```

Executables are written to `build-dayan/bin`, including `llama-cli`, `llama-server`, `llama-bench`, `llama-dayan-trace`, and `llama-dayan-operator-bench`. `LLAMA_BUILD_SERVER=ON` also enables the CLI target in this checkout. To rebuild only the inference tools and Dayan tests after a change:

```sh
cmake --build build-dayan -j6 --target \
    llama-cli llama-server llama-bench llama-dayan-trace llama-dayan-operator-bench \
    test-backend-ops test-dayan-matmul test-dayan-graph test-dayan-mxfp4
```

The same binaries accept BF16, FP16, and MXFP4 models. Select the model at runtime with `-m /path/to/model.gguf` and the server connection with `DAYAN_SERVER_SOCKET`; the build directory name does not select a weight format.

## Backend behavior

The backend supports contiguous `GGML_OP_MUL_MAT` with BF16 or FP16 weights, including broadcastable `ne2` and `ne3` batches. Activations are stored with the same type as the weights, matrix accumulation uses F32, and an unfused matrix multiplication returns the F32 result required by ggml.

Dense MXFP4 MatMul and FFN use `ggml-dayan-mxfp4.h`. MXFP4 weights must have one matrix batch, contiguous rows, and K divisible by 32. Activation batches are flattened into M. Attention operators and KV storage keep their existing floating-point paths. MXFP4 expert routing is not supported.

The GGUF representation remains the standard 17-byte block: one E8M0 scale followed by 32 FP4 values. Upload splits this into consecutive low-nibble-first FP4 payload and a separate scale view stored in `tensor->extra`. Readback restores the GGUF representation, including partial transfers. The arena allocation includes payload row padding and all scale planes, so the graph allocator owns both lifetimes.

For either operand, let `r` be the A row or B column and `g` the K32 group. The scale offset is `(g / 4) * scale_stride + r * 4 + g % 4`, with `scale_stride = align_up(rows_or_columns, 8) * 4`. Eight lanes occupy 32 bytes in each K128 plane. Scales are unswizzled; only matrix payload follows the selected MAT input swizzle. The load-a and load-b threads load both payload and scales before publishing their Resource. The paired gate/up products share A and release its Resource after both MAT reads.

MXFP4 tiling follows `npu-firmware/demo/matmul_fp4_4c.cpp`: each core keeps two A buffers and splits the macro tile into two N slices, each with its own B and F32 accumulator buffers. A is reused across both N slices and, for FFN, both gate/up products. The four cores own independent output macro tiles. `DayanMxfp4Tile` accounts for payload, scale planes, accumulators, alignment, VEC staging, and a 64-byte guard within 1536 KiB per core:

| Large-M path | Macro M x N x K | MAT slice M x N x K | URAM used | Capacity used |
| --- | --- | --- | --- | --- |
| MXFP4 MatMul, F32 output | 512 x 512 x 640 | 512 x 256 x 640 | 1534.06 KiB | 99.87% |
| MXFP4 MatMul with output conversion | 528 x 512 x 512 | 528 x 256 x 512 | 1535.78 KiB | 99.99% |
| MXFP4 gate/up + activation + conversion | 256 x 512 x 512 | 256 x 256 x 512, twice | 1530.28 KiB | 99.63% |

M shrinks to the actual row count rounded up to eight. Below 128 rows, N targets one quarter of the output columns, rounded up to 256 and capped at 3072; MXFP4 K grows to 1024, 2048, or 4096 when useful and within budget. Small matrices do not fill URAM with unused rows. BF16/FP16 producers retain K=512 and reduce M to fit their wider inputs. Actual tail instructions use valid M/N/K, including K32 tails.

The post thread processes 56 x 128 elements at a time; this is VEC staging, not the MAT slice size. LSU removes the accumulator swizzle into a 640-byte row stride before activation or conversion. The standalone conversion also handles 56 x 128 blocks. URAM A scale planes use `align_up(M * 4, 64)` as their K128 stride so LSU's 64-byte tail zero-fill cannot overwrite the next plane; the external Z layout is unchanged.

VisualSwizzle was evaluated with 32 banks, 32-byte words, eight rows per cycle, actual buffer offsets, and FP4 64-byte or floating-point 32-byte row accesses. A/B strides 256 and 320 use M/B/S=(7,1,3), 512 uses (7,2,3), 1024 uses (7,3,3), and 2048 uses (7,3,4). C uses the same rule for its power-of-two row-stride factor; the default 1024-byte C stride uses (7,3,3). All selected layouts have peak bank pressure 1 and zero excess requests. Four-block coverage is 26/32 for the 320-byte input stride and 32/32 for the other MAT layouts. The 320-byte mapping stays within 256-byte windows to preserve partial input buffer boundaries.

The rewrite shares activation casts and fuses eligible MatMul or SwiGLU/GeGLU producers with an MXFP4 output cast. Dependencies on the final fused node keep input payload and scale planes live through the post stage. Without these dependencies, the compiler leaves the operations separate. BF16/FP16 producers can also write MXFP4, and MXFP4 MatMul can write BF16/FP16 at an attention boundary. The post thread converts each output tile before its DRAM store. A standalone VEC conversion handles inputs whose producers cannot be fused. BF16 groups use the hardware MXFP4 conversion directly; F32 and FP16 inputs first round to BF16. Hardware conversion uses the smallest E8M0 scale that covers the finite normal group maximum and truncates E2M1 magnitudes. This differs from GGML's generic MXFP4 activation quantizer.

The SDK must include the MXFP4 correction in `dayan_intrinsics.h`: `__tx_vec_cvt_mxfp4_bf16<0, 0, 0>()` must encode `0x8027800b`. The older Clang builtin emits `0x8023800b`, which does not select the conversion instruction class. This uses the same SDK correction mechanism as the BF16/FP16 conversion wrappers.

Build `test-dayan-mxfp4` and run it against an isolated cmodel server. It checks scale Z layout, partial transfers, VEC conversion, matrix tails, repeated resource reuse, shared casts, fused FFN, and floating-point output boundaries. `test-dayan-mxfp4 --host-only` checks packing without a device.

Before scheduling a llama graph, the backend rewrite inserts one matching BF16 or FP16 `CPY` for each unique F32 activation used by an eligible matrix multiplication. Matrix multiplications that share an activation and storage type also share the cast tensor. Direct ggml users that do not run the rewrite can still provide F32 activations; the Dayan graph executor converts each unique activation once.

The rewrite moves a cast before an exclusive metadata-only `VIEW`, `RESHAPE`, `PERMUTE`, or `TRANSPOSE` chain when this does not remove an F32 output or side consumer. It rebuilds the aliases in the low-precision type, and the graph compiler finds the MatMul producer and cast by dependency instead of node adjacency. This keeps the cast in the MatMul epilogue slot without producing both F32 and low-precision matrix outputs.

Each ggml Dayan arena has one Device allocation, and each tensor owns a Device view at its planned arena offset. Graph kernels pass these views directly to their consumers. Intermediate results remain on Device; Device-to-Host transfer occurs only through an explicit tensor read or a scheduler boundary.

URAM bypasses the CPU cache. CPU accesses to URAM need no cache flush or invalidate. LSU, MAT, and VEC dependencies still require the corresponding unit waits or Named Resources. Cache maintenance applies to cached DRAM, including shared kernel configurations.

The graph executor lowers a Dayan split to execution slots. Two adjacent matrix multiplications with the same activation use one pair slot and one activation upload. A matching `MUL_MAT + MUL_MAT + SWIGLU/GEGLU + CPY` sequence uses one fused slot and the dedicated FFN pipeline in `ggml-dayan-ffn.h`. Each K step loads gate and up weight rows into one B tile, then issues one MAT operation per slice. Each F32 accumulator row holds the gate columns followed by the up columns. N tails move the up half to the next 64-column boundary and reduce MAT_N. VEC hardware loops cover rows and 64-column chunks, evaluate SwiGLU or GeGLU, convert to BF16/FP16, and use Shuffle Compress with a half-vector store mask to write packed URAM output directly. LSU writes only valid columns to DRAM; no second URAM packing copy is needed.

Both one-core and four-core FFN entries use the same shape-dependent pipeline and a 1536 KiB URAM budget. Four-core FFN partitions N evenly and broadcasts A from core 0. For M<=128, each output tile has a fixed 128-row A allocation and two N slices of 192 output columns; each B slice holds 192 gate and 192 up columns, and A is double-buffered. For M>128, each output tile has two 256-row M slices; B is double-buffered with 128 gate and 128 up columns per buffer, shared by the active M slices. MAT and LSU process only actual rows, including M<32. Gate and Up loads share one Resource, and the buffered operand is released only after all active slices consume it. Each slice has one C allocation. A single packed output buffer is reused after LSU finishes its store. Down projection uses the following MatMul slot and returns F32. It has no GLU activation; matching residual ADD, RMSNorm, scale, and conversion operations use the subsequent normalization pipeline slot.

The MoE path has separate kernels for `MUL_MAT_ID`, `ADD_ID`, `ARGSORT`, and `TOP_K`. The graph compiler also recognizes `MUL_MAT_ID + MUL_MAT_ID + SWIGLU/GEGLU + CPY`. This fused slot selects both expert matrices and runs the same dedicated FFN pipeline for each selected expert. Expert ID views are read with their actual row strides, and routing rows support up to 4096 entries.

Graphs containing `MUL_MAT_ID` retain F32 residual storage. Rounding the residual stream to BF16 can change close router scores and select different experts in later layers.

When an `ARGSORT` result is used only by zero-offset prefix views, the slot selects and sorts only the largest required prefix. This changes routing selection from O(E^2) to O(E*K), retains the original row stride, and preserves the sorted order required by `ggml_argsort_top_k`. Full outputs and other consumers retain full sorting. `GET_ROWS` accepts strided I32/I64 indices, including these routing views. Sorting uses independent Procs with no child Threads or Named Resources.

F32 binary operators pack strided inputs with the existing Device copy kernel before arithmetic or broadcast. This covers the expert output views used in MoE aggregation. These copies are runtime staging within the binary slot. CLAMP accepts infinite bounds, including the lower-only clamp used to normalize selected routing weights.

The one-core and four-core MatMul entries use the same pipeline: two input loader Threads, the root MAT worker, and a post Thread. The general four-core loaders use DAPI Resource producer election to share A across columns and B across rows. Local Resources protect MAT inputs and accumulators; Shared Resources keep each source alive until the peer copy finishes. Each worker tracks Resource phases across batches and repeated expert products. LSU supports partial output rows, including the single-column shared expert gate.

The `matmul[_quantized|_silu]_inference_*` entries specialize small-M inference. For `1 <= M <= 512`, the graph executor selects the inference entry; pair, expert, and attention projection slots use the same layout internally. Four-core inference assigns core `c` the contiguous N interval `[N*c/4, N*(c+1)/4)`, so shard widths differ by at most one column. Core 0 loads each A tile once and broadcasts it through URAM, with three Shared Resource copy credits. Each core loads its own weight rows and computes the full M range. Cores with empty N shards still retire their A copy credits. Larger MatMuls retain the general 2x2 cooperative layout; FFN keeps its N partition.

For `M < 32`, both one-core and four-core inference use exactly M rows per A tile, including odd sizes. The A row stride is 512 bytes (`K_tile=256`, BF16/FP16), so the two A buffers occupy `2*M*512` bytes; decode with M=1 uses 1 KiB. No padding to 32 or 128 rows is needed. For the MatMul entries, the small A allocation leaves room for larger B slices: M<32 uses four 384-column slices on four cores, or four 192-column slices on one core. M>=32 uses four 96-column slices. Partial K and N dimensions go directly to MAT, and LSU stores only valid output bytes.

VisualSwizzle selects `M=7,B=2,S=3` for the 512-byte A/B row strides. Plain C uses linear storage at a 384-byte row stride and `M=7,B=2,S=3` at a 1536-byte row stride. Epilogues use linear C with a 640-byte or 1664-byte row stride. Small-M MatMul inference uses a linear C row stride of 896 bytes on one core or 1664 bytes on four cores. Dedicated FFN uses a 1664-byte linear accumulator row stride for 192+192 columns or a 1152-byte stride for 128+128 columns, including 32 padding floats, so MAT and VEC share the same layout. For full eight-row access groups, the selected FFN A/B mapping reduces peak Bank pressure from 4 to 1; linear FFN C already has peak pressure 1. Both cover all 32 Banks across four consecutive 32-byte blocks. The FFN buffers use 976 KiB per core for M<=128 and 1152 KiB for M>128, within the 1536 KiB budgets. These are address-mapping results, not measured hardware throughput.

Contiguous F32 residual adds and RMSNorm execute on Dayan. One parameterized normalization pipeline slot handles optional residual ADD, RMSNorm, scale multiplication, and BF16/FP16 quantization. Matrix multiplication remains a separate slot. The graph rewrite keeps every fused input live through the final logical node so the allocator cannot overlap a cast output with an input that another native Task still reads. VEC tensor dimensions expand complete 64-lane chunks into padded URAM scratch space, while LSU tail stores limit the final DRAM write to the valid byte count.

The attention operator package provides dedicated BF16 and FP16 QG, K, and V projection entry points. Q and K fuse low-precision matrix writeback, RMSNorm, RoPE or MRoPE, and output layout conversion. K and V also have indexed KV-cache variants that write directly to cache rows. Flash-Attention has native BF16 and FP16 entries, uses `dayan_uram_base()` directly, and has an explicit causal offset for KV-cache decode. It supports GQA and right-aligned causal masking. The output path fuses `attention * sigmoid(gate)`, low-precision quantization, and the output projection. The graph compiler recognizes the Qwen3.5 gate view, sigmoid, multiply, and output matrix multiplication pattern.

Flash-Attention uses 128-query tiles and 128-key tiles. Each core owns two Q slices; Q remains resident across the KV traversal. K and transposed V each have two buffers. Four-core mode splits Q tiles between cores and elects one active core to load each K or V tile, then copies its URAM tile to the other active cores. Every core retires its Shared Resource credit, including cores with no query rows. Single-core mode uses private KV loaders. The graph executor selects four-core mode for more than 64 query rows and single-core tasks for shorter queries.

The root issues QK MAT; six child Threads load Q, load K, load V, compute online softmax, issue PV MAT, and update/store O. Each Q slice has two Score/P, PV, and softmax-state buffers, allowing the compute stages to overlap across KV tiles as well as slices. Score and P share FP32 storage in each buffer; both Resource tokens must be free before QK can overwrite it. P is read directly as TF32 by the mixed TF32/BF16 or TF32/FP16 PV product. Each state generation has an O-update reader and, except for the last tile, a reader in the next softmax. Two alternating O version tokens serialize updates to one O allocation. There are 26 Local Resources and, in four-core mode, four Shared Resources plus 128 bytes of shared metadata.

Q becomes reusable after its last QK MAT read. K becomes reusable after all active slices finish their QK MAT reads and all peer copies finish. Neither waits for softmax or O update. PV MAT releases P and V when its reads complete; O update releases PV and its state credit when VEC completes. The final O token is released by the LSU store. Resource phases persist across query groups and batches, including slices with different causal tile counts.

Each core reserves 1536 KiB of URAM:

| Buffer | Count | KiB per buffer | Lifetime |
| --- | ---: | ---: | --- |
| Q | 2 | 64 | Load through last QK read for one query slice |
| Score/P | 4 | 64 | QK write through PV read |
| PV | 4 | 128 | PV write through O update |
| O | 2 | 128 | All KV tiles through output store |
| M, L, alpha | 4 each | 4 | Softmax write through next softmax and O update |
| Mask | 4 | 8 | One softmax tile, reserved with its state buffer |
| K, transposed V | 2 each | 64 | Local MAT and peer-copy consumers |

The buffers occupy 1488 KiB plus a 256-byte zero source. Maximum sizes support head dimensions 64, 128, 192, and 256. Q/K use identity swizzling for dimensions 64/192 and `(5,1,5)`/`(5,2,5)` for 128/256. Transposed V uses `(5,1,5)`; VEC-visible buffers use identity swizzling.

Online softmax follows `FlexVecAttentionSpec.scala`: each 64-lane VEC context processes eight rows, with eight lanes per row, two scans of 128 columns, native FP32 max, and XOR reductions by 4, 2, and 1. The unmasked block has 167 instructions: the test's 131 plus 32 scale multiplies, three constant broadcasts, and one alpha store. Causal/tail masks add 64 instructions. FP32 M/L remain in registers between the scans. The firmware SDK must provide `__tx_vec_v_fp32_eru` and `MAT_DATATYPE_PAIR` with mixed TF32 operand support.

For Qwen3.5 linear-attention layers, `SSM_CONV` followed by SiLU uses one dedicated device function. LSU gathers the F32 channel-major input and depthwise weights into tap-major URAM tiles. VEC performs one elementwise multiply per tap, accumulates the taps in F32, and optionally applies the common SiLU implementation before one output writeback. The kernel does not construct diagonal matrices or use MAT convolution. It supports kernel widths up to 63 and uses 64-token tiles. The convolution remains a separate operator from the delta-net core.

Gated Delta Net assigns heads and sequences to at most 16 root tasks, with up to four resident tasks per core. Decode and short prefill update state with VEC hardware loops: DIM2/DIM3 provide 16 independent contexts per head, each processing eight state rows. Scalar-gate prefill with at least 16 tokens and one final state snapshot uses 64-token chunks. MAT computes block correlations, output contributions, and state updates; VEC computes gate prefixes and triangular corrections with 128 independent value rows. Only the complete FP32 state crosses chunk boundaries. Vector gates (KDA) and multiple state snapshots use the recurrent VEC path.

Chunk MAT uses `MAT_DATATYPE_PAIR(DT_MATIN_TF32, DT_MATIN_TF32)`. The inputs have FP32 storage and TF32 precision; accumulation and VEC arithmetic remain FP32. This requires firmware headers and cmodel with the TF32 operand encoding enabled. The chunk layout uses 345.5 KiB within a 352 KiB task reservation, or 1408 KiB for four resident tasks per core. Head and chunk row strides are 576 and 320 bytes, so LSU transposes remain 64-byte aligned. All units use identity swizzling; VisualSwizzle reports peak Bank pressure 1 and no extra requests for these layouts. Padding and the triangular mask handle chunk tails, and unused gate differences are masked before exponentiation.

## Profiling

Device operators contain no custom cycle counters or performance buffers. Use the firmware profiler for scheduling traces:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock "$DAYAN_ROOT/build/host/dyperf/dyperf" -o /tmp/dayan-trace.json -- ./build-dayan/bin/test-dayan-matmul
```

Use `llama-bench` for actual host elapsed time and tokens per second on cmodel. Record the model, prompt/generation lengths, batch sizes, repetitions, and simulator configuration. Keep the profiler disabled for these wall-clock measurements and run one client at a time.

For a limited TestCluster operator sample with an existing server and dyperf, use [limited-perf.py](../../../tools/dayan-trace/limited-perf.md). It reports measured host wall time and profiler cycle spans by shape, plus an explicit Qwen3.5-0.8B prefill/decode extrapolation, without loading a whole model or running a shape sweep.

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

To run cmodel alongside TestCluster, isolate its FUSE mount, Runtime socket, emulator IPC sockets, and DRAM/URAM shared-memory names. Separate mount and Runtime socket paths alone are insufficient. The stock `run_cmodel_server.sh` cleans the default IPC and shared-memory files, so do not use it while TestCluster owns those resources. Use an isolated emulator/QEMU pair and a launcher whose cleanup touches only that instance's paths. Clients select the instance through `DAYAN_SERVER_SOCKET`.

Replace `test` with `support` to check whether Dayan supports the selected cases without running their numerical comparison. `perf` runs the performance case set, which has different shapes and may require a different `-p` filter; host timings under simulation are not hardware throughput measurements.

## Dedicated tests

The standalone BF16 matrix multiplication smoke test defaults to `M=1`, `N=8064`, and `K=1024`. Pass three dimensions to override the default shape:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-matmul
```

The focused graph test checks BF16/FP16 SWIGLU/GEGLU fused matrix multiplication variants, low-precision output quantization, the following matrix multiplication, and the Qwen3.5 0.8B `M=8, N=3584, K=1024` FFN shape:

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

The `glu` selector checks generic and inference 1c/4c SWIGLU and GEGLU entries in BF16/FP16. Cases cover M=1,2,3,7,8,15,16,31,32,128,129,257,512,513, N shards with different widths or no columns, K tails, repeated output tiles, repeated launches, and broadcast batches with padded row and batch strides in both slice regimes. N=769 exercises one or two active small-M N slices across different cores. M=513 exercises two M slices followed by one active tail slice. Output padding and trailing guards must remain unchanged. The MatMul test also checks the M=512/513 dispatch boundary and packed broadcast batches above 512 rows.

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-attention-projection glu
```

The attention projection test runs QG, K, and V projection, indexed KV-cache writes, causal Flash-Attention, gate fusion, low-precision quantization, and output projection as one numerical data flow for BF16 and FP16 model weights:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-attention-projection
```

The Gated Delta Net test compares outputs and state snapshots with the CPU reference. It covers decode, 16/17/64/65/128/1024-token prefill, weak and strong gates, shared Q/K heads and sequences, repeated task work, KDA, and multiple state snapshots. Chunk tests check TF32 error separately for outputs and final state; recurrent tests retain FP32 tolerances.

```sh
cmake --build build-dayan --target test-dayan-gated-delta-net -j4
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-gated-delta-net
```

The standalone Flash-Attention test checks GQA with a right-aligned causal mask where `q_len < kv_len`:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-flash-attention
```

The direct attention pipeline test covers both launch modes, BF16/FP16, all supported head dimensions, GQA, strided KV, causal/tail masks, buffer reuse across query groups, and changing online maxima. It compares with an FP32 host reference using `abs_error <= 0.001 + 0.01 * abs(reference)` and rejects nonfinite output. Pass one printed case name to select it:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-attention-pipeline
```

The Qwen3.5 dense test builds the minimum four-layer hybrid cycle with three gated DeltaNet layers and one causal full-attention layer. Every layer also contains RMSNorm, dense SWIGLU FFN, and residual paths. It compares Dayan outputs and final logits against a CPU graph with the same low-precision boundaries. Gated Delta Net uses a 128-element head so the test exercises the device kernel instead of CPU fallback:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-qwen35-dense
```

The MoE smoke test covers 512 experts, a narrow `ARGSORT` view, `TOP_K`, fused BF16 SWIGLU expert projections, the down expert projection, `ADD_ID`, and expert-weighted aggregation:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-moe
```

Both matrix launch modes use four consumer slices. `MUL_MAT_ID` still requires the expert output width to be a multiple of 16. Arbitrary attention masks, attention sinks, and Flash-Attention soft caps are not implemented. Flash-Attention has native BF16 and FP16 entries and requires Q/K/V to share one storage type. Full-model inference requires enough Device memory for all persistent weights and graph arenas; `run_cmodel_server.sh` defaults to a 4096 MiB memory window. A llama model using Dayan rejects any non-metadata model graph node assigned to another backend. Sampling stays on CPU and is not inserted into the Dayan graph.
