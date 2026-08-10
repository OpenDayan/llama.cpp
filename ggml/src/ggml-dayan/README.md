# Dayan backend

The backend supports contiguous two-dimensional `GGML_OP_MUL_MAT` with BF16 or FP16 weights. Activations are stored with the same type as the weights, matrix accumulation uses F32, and an unfused matrix multiplication returns the F32 result required by ggml.

Before scheduling a llama graph, the backend rewrite inserts one matching BF16 or FP16 `CPY` for each unique F32 activation used by an eligible matrix multiplication. Matrix multiplications that share an activation and storage type also share the cast tensor. Direct ggml users that do not run the rewrite can still provide F32 activations; the Dayan graph executor converts each unique activation once.

The graph executor lowers a Dayan split to execution slots. Two adjacent matrix multiplications with the same activation use one pair slot and one activation upload. A matching `MUL_MAT + MUL_MAT + SWIGLU/GEGLU + CPY` sequence uses one fused slot. The common matrix multiplication template accepts a compile-time epilogue function pointer, evaluates the activation with VEC before the final writeback, and quantizes the result to the weight type.

The MoE path has separate kernels for `MUL_MAT_ID`, `ADD_ID`, `ARGSORT`, and `TOP_K`. The graph compiler also recognizes `MUL_MAT_ID + MUL_MAT_ID + SWIGLU/GEGLU + CPY`. This fused slot selects both expert matrices, accumulates both products in F32 URAM buffers, calls the same compile-time VEC epilogue function pointer, and uses the common low-precision writeback. Expert ID views are packed with their actual strides, and routing rows support up to 4096 entries.

Contiguous F32 residual adds and RMSNorm execute on Dayan. The graph compiler recognizes `RMS_NORM + MUL + CPY + MUL_MAT` and `ADD + RMS_NORM + MUL + CPY + MUL_MAT`, and runs normalization, scaling, BF16/FP16 quantization, and matrix multiplication in one device function. VEC tensor dimensions expand complete 64-lane chunks into padded URAM scratch space, while LSU tail stores limit the final DRAM write to the valid byte count.

The attention operator package provides dedicated BF16 and FP16 QG, K, and V projection entry points. Q and K fuse low-precision matrix writeback, RMSNorm, RoPE or MRoPE, and output layout conversion. K and V also have indexed KV-cache variants that write directly to cache rows. The migrated BF16 Flash-Attention implementation is maintained in this backend, uses `dayan_uram_base()` directly, and has an explicit causal offset for KV-cache decode. It supports GQA and right-aligned causal masking. The output path fuses `attention * sigmoid(gate)`, low-precision quantization, and the output projection. The graph compiler recognizes the Qwen3.5 gate view, sigmoid, multiply, and output matrix multiplication pattern.

For Qwen3.5 linear-attention layers, `SSM_CONV` followed by SiLU uses one dedicated device function. It packs the depthwise convolution into BF16 diagonal kernels with 256-channel main slices, 128-channel tail slices, and 64-token tiles. The first tap uses `tx.mat.conv2d`, later taps use `tx.mat.conv2da`, and VEC applies SiLU before the result is written back. The convolution remains a separate operator from the delta-net core.

Configure the source SDK and instruction headers explicitly:

```sh
cmake -S . -B build-dayan -DGGML_DAYAN=ON -DDAYAN_ROOT=/path/to/npu-firmware -DDAYAN_NPU_KERNELS_DIR=/path/to/npu-kernels
cmake --build build-dayan --target test-backend-ops -j
```

Start `dayan_server`, set `DAYAN_SERVER_SOCKET` when it does not use the default socket, and run one supported smoke case:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-backend-ops -b DAYAN -o MUL_MAT -p 'type_a=bf16,type_b=f32,m=16,n=1,k=256,bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\]'
```

The standalone BF16 matrix multiplication smoke test defaults to `M=1`, `N=8064`, and `K=1024`. Pass three dimensions to override the default shape:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-matmul
```

The focused graph test checks all BF16/FP16 SWIGLU/GEGLU fused matrix multiplication variants, including both matrix producer paths, low-precision output quantization, the following matrix multiplication, and the Qwen3.5 0.8B `M=8, N=3584, K=1024` FFN shape:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-graph
```

The same test also checks the Qwen3.5 attention output gate plus projection pattern, fused RMSNorm matrix multiplication paths, residual adds, and the fused SSM convolution plus SiLU path.

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

The current matrix kernel uses one consumer slice because repeated fused launches can deadlock with the three-slice cmodel pipeline. It also requires the weight row count to be a multiple of 16. The MoE prototype uploads the complete expert weight tensor for each launch, so it is suitable for functional smoke tests but not a full Qwen3.5 397B run. Selected-expert or chunked weight transfer is still required. Batched ordinary matrix multiplication, device-resident weights, arbitrary attention masks, attention sinks, and Flash-Attention soft caps are not implemented. Flash-Attention consumes BF16 Q/K/V; the FP16 attention data-flow test converts projected Q/K/V to BF16 at the Flash boundary. A llama model using Dayan rejects any non-metadata model graph node assigned to another backend. Sampling stays on CPU and is not inserted into the Dayan graph.
