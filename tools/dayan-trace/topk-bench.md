# Dayan Top-8 comparison

`topk-bench.cpp` compares three Device implementations of sorted Top-8 selection:

- `scalar-direct`: the RISC-V core reads the original DRAM input, keeps eight scores on its stack, and writes indices directly to DRAM. It invalidates the input/output cache ranges before use and flushes the output afterward. It allocates no URAM or tensor units.
- `vec-radix`: LSU loads the row into URAM. VEC builds 64-bin atomic histograms, scans each histogram with six shuffle/add/mask steps, and selects the next six key bits. Once at most `candidate_limit` values remain above the threshold, VEC emits lane masks and the scalar core orders the remaining candidates. Ties can exceed that limit and are resolved by input index.
- `uram-insertion`: the existing backend's insertion selection, including its LSU input and output transfers. URAM is accessed without cache operations.

This is an isolated experiment. It does not change backend dispatch. All three return the same descending indices, with smaller indices first for equal scores. The output row stride is the input width, matching the ARGSORT prefix used by MoE routing. Only the first eight indices are written; the host checks the remaining output against a sentinel.

The radix experiment accepts finite nonnegative FP32 values without negative zero, widths from 64 to 4096 in multiples of 64, and K=8. It relies on the bitwise ordering of nonnegative FP32 values. It is not a general replacement for ggml ARGSORT/TOP_K.

## Build

From the llama.cpp checkout, with the Dayan firmware/compiler built:

```sh
${DAYAN_ROOT}/host/dycc/dycc -O2 -Wall -Wextra -Wno-unused-function \
  -Iggml/src/ggml-dayan -Xdevice -Iggml/src/ggml-dayan/device_include \
  --save-temps=build-dayan/topk-bench-temps \
  tools/dayan-trace/topk-bench.cpp -o build-dayan/bin/topk-bench
```

## Run

Start a Dayan server backed by TestCluster, then run one client at a time:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock build-dayan/bin/topk-bench 256 1 3 softmax 32 scalar-direct
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock build-dayan/bin/topk-bench 256 1 3 softmax 32 vec-radix
```

Arguments are `width rows repetitions pattern candidate_limit mode`. The patterns are `softmax`, `ascending`, `descending`, `ties`, or a path to an FP32 binary input. A file supplies the first `width * rows` elements without normalization. The generated Softmax input uses deterministic logits uniformly distributed in [-4, 4]. The default `compare` mode runs the direct scalar and VEC radix implementations. Mode `all` also runs the URAM baseline. A candidate limit of 8 refines to the exact threshold; 32 allows an earlier switch to scalar selection.

Each kernel uses one Proc on one Core and handles rows sequentially. This measures single-row latency and sequential batching, not concurrent throughput. Each sample uses `dayan_get_time()` around the complete operation, including required cache operations, tensor setup, DMA, waits, histogram clearing, bucket selection, and final index output. Kernel launch, module loading, input upload, timing-record writes, and output download are outside that interval. Sample zero is reported separately and excluded from the median. For an even number of measured samples, the summary reports the upper middle sample. Repeated samples reuse the input values; the direct scalar path invalidates its input on every sample.

Use TestCluster cycle counts to compare implementations. Cmodel validates functionality but does not provide RTL timing. Its current SparseMem backing is separate from scalar shared-memory accesses; the experiment transfers histogram prefix results and masks through URAM so both backends can observe them.
