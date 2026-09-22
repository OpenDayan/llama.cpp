# Dayan backend

MXFP4 dense, BF16 dense, and MXFP4 MoE models use the same `ggml-dayan` backend and kernel module. Enabling `GGML_DAYAN` compiles the BF16, FP16, and MXFP4 operator paths together; runtime dispatch selects operators by tensor type. Switching model groups requires no format-specific CMake flags or rebuild.

## Common commands

Run commands from the llama.cpp repository root. The examples cover these three model groups:

| Model group | GGUF | Operator preset |
| --- | --- | --- |
| MXFP4 dense | Qwen3.5-0.8B, including MXFP4 embedding/LM head | `qwen35-0.8B-mxfp4-vocab` |
| BF16 dense | Qwen3.5-0.8B | `qwen35-0.8B-bf16` |
| MXFP4 MoE | Qwen3.6-35B-A3B, BF16 attention/shared experts and F32 routers | `qwen36-35B-A3B-mxfp4-moe-bf16` |

Set the SDK, Runtime socket, and model paths in each client terminal:

```sh
export DAYAN_ROOT="$HOME/npu-firmware"
export DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock
export DAYAN_MODEL_MXFP4=/share/models/Qwen3.5-0.8B-MXFP4/Qwen3.5-0.8B-MXFP4-Vocab-MXFP4.gguf
export DAYAN_MODEL_BF16=/share/models/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf
export DAYAN_MODEL_MOE=/share/models/Qwen3.6-35B-A3B-MXFP4_MOE_BF16.gguf
```

For the dense MXFP4 GGUF with a BF16 vocabulary head, select `Qwen3.5-0.8B-MXFP4.gguf` and the `qwen35-0.8B-mxfp4` operator preset instead.

### Build all three model paths

Build on Linux with the Dayan SDK's `dycc`, host and Device API headers, and `libDayanRuntime.a`. One build supports all three model groups:

```sh
make -C "$DAYAN_ROOT" -j6 host-driver dycc dyperf
cmake -S . -B build-dayan \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_DAYAN=ON \
    -DDAYAN_ROOT="$DAYAN_ROOT" \
    -DDAYAN_DYCC="$DAYAN_ROOT/host/dycc/dycc" \
    -DDAYAN_INCLUDE_DIR="$DAYAN_ROOT/host/include" \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_SERVER=ON
cmake --build build-dayan -j6
```

Executables are written to `build-dayan/bin`. To rebuild the tools used below after a change:

```sh
cmake --build build-dayan -j6 --target \
    llama-cli llama-server llama-bench llama-dayan-trace \
    llama-dayan-operator-bench llama-dayan-moe-bench \
    test-backend-ops test-dayan-topology test-dayan-matmul test-dayan-graph \
    test-dayan-mxfp4 test-dayan-moe test-dayan-qwen35-dense \
    test-dayan-attention-projection test-dayan-flash-attention
```

### Start cmodel and the Runtime server

Rebuild cmodel after changing its source or instruction implementations. The firmware launcher builds firmware and the Runtime server, but does not rebuild cmodel:

```sh
make -C "$HOME/rvtng-cmodel" -j6
```

| Model group | Example memory capacity |
| --- | --- |
| MXFP4 dense, 0.8B | 4096 MiB |
| BF16 dense, 0.8B | 4096 MiB |
| MXFP4 MoE, 35B-A3B | 32768 MiB |

In a separate terminal with `DAYAN_ROOT` and `DAYAN_SERVER_SOCKET` set, start one four-core server. This 32 GiB configuration can be reused for all three groups:

```sh
NPU_CMODEL_HARTS=4 NPU_CMODEL_MEMORY_MIB=32768 \
    bash "$DAYAN_ROOT/run_cmodel_server.sh"
```

Wait for `Dayan cmodel server is ready`. Run one inference or operator client at a time, and stop the launcher with Ctrl-C after testing. A single-core operator run uses `NPU_CMODEL_HARTS=1`; mega requires one complete four-core Cluster. If using an existing server, select its socket instead of starting another simulator.

### Run model inference

Run the command for the desired model. Mega is enabled by default when the graph and topology match:

```sh
# MXFP4 dense
./build-dayan/bin/llama-cli -m "$DAYAN_MODEL_MXFP4" \
    --device DAYAN -ngl 99 -fa on -c 512 -b 128 -ub 128 \
    -n 16 --no-warmup --single-turn -p "Hello"

# BF16 dense
./build-dayan/bin/llama-cli -m "$DAYAN_MODEL_BF16" \
    --device DAYAN -ngl 99 -fa on -c 512 -b 128 -ub 128 \
    -n 16 --no-warmup --single-turn -p "Hello, Introduce yourself"

# MXFP4 MoE
./build-dayan/bin/llama-cli -m "$DAYAN_MODEL_MOE" \
    --device DAYAN -ngl 99 -fa on -c 512 -b 128 -ub 128 \
    -n 16 --no-warmup --single-turn -p "Hello, Introduce yourself"
```

For HTTP serving, choose one model from this table, then run the shared server command:

| Model group | Selection |
| --- | --- |
| MXFP4 dense | `export DAYAN_MODEL="$DAYAN_MODEL_MXFP4"` |
| BF16 dense | `export DAYAN_MODEL="$DAYAN_MODEL_BF16"` |
| MXFP4 MoE | `export DAYAN_MODEL="$DAYAN_MODEL_MOE"` |

```sh
./build-dayan/bin/llama-server -m "$DAYAN_MODEL" \
    --device DAYAN -ngl 99 -fa on -c 512 -b 128 -ub 128 \
    --parallel 1 --no-warmup --host 127.0.0.1 --port 8080
```

Set `GGML_DAYAN_MEGA=0` before any of these commands to use standalone operators, or `GGML_DAYAN_MEGA=1 GGML_DAYAN_MEGA_DEBUG=1` to enable mega and print slot sequences and launch counts. Restart a running inference process after changing these settings. A slot trace or slot filter callback disables mega. Both initial graphs and subsequent decode graphs without the two state SCALE nodes are supported.

### Check standalone operators before mega

Check topology selection and MXFP4 packing without a simulator:

```sh
ctest --test-dir build-dayan -R '^test-dayan-topology$' --output-on-failure
./build-dayan/bin/test-dayan-mxfp4 --host-only
```

With the Runtime server ready, select the model group:

```sh
# MXFP4 dense: packing, conversions, MatMul, fused FFN, tails, and repeated launches
GGML_DAYAN_MEGA=0 ./build-dayan/bin/test-dayan-mxfp4

# BF16 dense: MatMul and graph fusion, including FP16 variants
GGML_DAYAN_MEGA=0 ./build-dayan/bin/test-dayan-matmul 3 481 513
GGML_DAYAN_MEGA=0 ./build-dayan/bin/test-dayan-graph

# MXFP4 MoE: routing regression and exact FlexVec radix-sort indices
GGML_DAYAN_MEGA=0 ./build-dayan/bin/test-dayan-moe
GGML_DAYAN_MEGA=0 ./build-dayan/bin/test-backend-ops test -b DAYAN -o ARGSORT \
    -p 'ne=\[(63|64|65|256|4096),3,1,1\]'
GGML_DAYAN_MEGA=0 ./build-dayan/bin/test-backend-ops test -b DAYAN -o TOP_K \
    -p 'ne=\[256,5,1,1\],k=8'
```

`test-dayan-moe` uses BF16 expert weights. The MXFP4 expert products and TF32 routers are checked by the MoE operator preset in [Operator benchmarks](#operator-benchmarks-with-limited-perfpy); run that preset before the MoE mega test. Shared attention, recurrence, memory, and RoPE selectors are listed under [Dedicated tests](#dedicated-tests) and [Test a specific operator](#test-a-specific-operator).

### Check mega and decode graphs

These commands use different references. Dense MXFP4 uses a trained-GGUF smoke run plus the standalone CPU checks above. Dense BF16 uses a random layer against CPU. MoE compares a random two-layer mega graph with standalone Dayan; its CPU operator checks remain separate.

MoE mega reuses the BF16 attention segment and includes routing, MXFP4 experts, shared experts, and the F32 residual in one launch per layer. Parameter offsets are 64-bit for weights beyond 4 GiB. Benchmark clients can set `GGML_DAYAN_MEGA_TAG` to override the profiler name.

#### MXFP4 dense

Run the trained model with launch diagnostics. Several generated tokens exercise the first and later decode graphs:

```sh
GGML_DAYAN_MEGA=1 GGML_DAYAN_MEGA_DEBUG=1 \
./build-dayan/bin/llama-cli -m "$DAYAN_MODEL_MXFP4" \
    --device DAYAN -ngl 99 -fa on -c 512 -b 128 -ub 128 \
    -n 3 --no-warmup --single-turn -p "Hello"
```

Inspect the mega launch count; successful inference alone does not prove fusion. There is no dedicated dense MXFP4 mega preset in `limited-perf.py`.

#### BF16 dense

Run the default random dense regressions, or reproduce just the single-token linear case that checks fused BF16 input lifetimes:

```sh
GGML_DAYAN_MEGA=1 ./build-dayan/bin/test-dayan-qwen35-dense

mkdir -p build-dayan/mega-bf16
cat > build-dayan/mega-bf16/linear-decode.case.json <<'JSON'
{"op":"mega","dtype":"bf16","mode":"linear","layers":1,"m":1,"seed":35,"repeats":1,"warmup":0,"cmp":true}
JSON
GGML_DAYAN_MEGA=1 GGML_DAYAN_MEGA_DEBUG=1 \
./build-dayan/bin/test-dayan-qwen35-dense \
    build-dayan/mega-bf16/linear-decode.case.json \
    build-dayan/mega-bf16/linear-decode.result.json dense-linear
```

The case protocol requires both a CPU comparison pass and a mega launch. Disabling mega deliberately makes that protocol fail its launch check, even if the numerical comparison passes. The default suite also checks four-layer BF16/FP16 hybrid graphs.

#### MXFP4 MoE

Generate a random two-layer fixture with the source widths, 256 experts, one linear-attention layer, one full-attention layer, and a small vocabulary. It is about 953 MiB and avoids loading the full trained weight payloads:

```sh
python3 tools/dayan-trace/create-qwen35-moe-fixture.py \
    --source-model "$DAYAN_MODEL_MOE" \
    --expert-type mxfp4 --layers 2 --full-attention-interval 2 --small-vocab \
    --output build-dayan/moe-validation/hybrid.gguf
export DAYAN_MOE_MODEL="$PWD/build-dayan/moe-validation/hybrid.gguf"

cat > build-dayan/moe-validation/decode.case.json <<'JSON'
{"op":"mega","dtype":"mxfp4","mode":"hybrid","layers":2,"m":1,"phase":"decode","decode_steps":2,"seed":35,"repeats":1,"warmup":0,"cmp":true}
JSON
GGML_DAYAN_MEGA_DEBUG=1 ./build-dayan/bin/llama-dayan-moe-bench \
    build-dayan/moe-validation/decode.case.json \
    build-dayan/moe-validation/decode.result.json moe-decode
```

The same client accepts a dense `qwen35` model or fixture through `DAYAN_MEGA_MODEL`, with `m` up to 1024 and `layers` set to the model layer count. It uses the model graph builder and requires one mega launch per layer at every step.

This checks an initial single token followed by two decode steps. Each step compares logits and requires two mega launches, including graphs where state SCALE nodes disappear. The client switches between standalone and mega internally. Use the MoE mega preset below to check both prefill and initial-single-token sequences. See [MoE validation](../../../tools/dayan-trace/qwen35-moe.md) for the full-model smoke result and quantization limits.

### Operator benchmarks with limited-perf.py

`--model` selects operator shapes, not a GGUF file. These three presets use the same client and run CPU comparisons on stored BF16/MXFP4 inputs or TF32-rounded inputs. Each MoE expert case already executes all eight selected experts; its layer factor is 40, not 320.

Preview all three groups without contacting a server:

```sh
for preset in qwen35-0.8B-mxfp4-vocab qwen35-0.8B-bf16 qwen36-35B-A3B-mxfp4-moe-bf16; do
    python3 tools/dayan-trace/limited-perf.py plan \
        --model "$preset" --prefill 4 --context 16 --cmp --cmp-seed 35
done
```

With cmodel ready, run the desired presets sequentially. Remove names from the loop to test only one model group. Each run has its own trace and output directory:

```sh
for preset in qwen35-0.8B-mxfp4-vocab qwen35-0.8B-bf16 qwen36-35B-A3B-mxfp4-moe-bf16; do
    "$DAYAN_ROOT/build/host/dyperf/dyperf" \
        --socket "$DAYAN_SERVER_SOCKET" --follow -o "/tmp/dayan-$preset.json" -- \
        python3 tools/dayan-trace/limited-perf.py run \
            --model "$preset" --client build-dayan/bin/llama-dayan-operator-bench \
            --simulator cmodel --socket "$DAYAN_SERVER_SOCKET" \
            --perf-trace "/tmp/dayan-$preset.json" --output "build-dayan/perf-$preset" \
            --prefill 4 --context 16 --cmp --cmp-seed 35
    python3 tools/dayan-trace/limited-perf.py report --output "build-dayan/perf-$preset"
done
```

The profiler flushes its trace after the client exits. The initial `run` can return 2 for missing cycle records; the following `report` reads the completed trace without rerunning operators. Inspect both comparison status and cycle records. Output includes `report.md`, `report.json`, `operators.csv`, and per-case inputs, results, and logs.

For a short run, add one of these filters to `plan` and `run`, using a separate output directory:

| Model group | Example filter |
| --- | --- |
| MXFP4 dense | `--cases decode-ffn decode-down` |
| BF16 dense | `--cases decode-ffn decode-down` |
| MXFP4 MoE | `--cases decode-router decode-top8 decode-expert-ffn decode-expert-down` |

`--cmp` covers the selected cases, not every model operator. Row reads/writes, RoPE, layouts, and small gates need the separate tests above. Hardware MXFP4 conversion differs from GGML's nearest-value quantizer. For larger samples, adjust `--prefill` and `--context`; use `--warmup 1 --repeats 3` for repeated operator measurements. `--resume` requires the original trace, output directory, binaries, and parameters. See [limited-perf.md](../../../tools/dayan-trace/limited-perf.md) for prediction formulas and exit statuses.

Top-8 routing uses four measured prefill rows by default and scales only that sort term to the requested `--prefill` in the report. Use `--sort-rows 0` when a full-size sorting measurement is required; this can be very slow on cmodel. Other MoE operator cases still use the requested prefill size.

### Mega benchmarks

| Model group | Benchmark path |
| --- | --- |
| MXFP4 dense | Use the GGUF `llama-bench` commands below; no separate mega preset |
| BF16 dense | `qwen35-mega-bf16`, random linear/full layers against CPU |
| MXFP4 MoE | `qwen36-35B-A3B-moe-mega`, two-layer fixture against standalone Dayan |

For BF16 dense, benchmark both attention modes in prefill and decode:

```sh
"$DAYAN_ROOT/build/host/dyperf/dyperf" \
    --socket "$DAYAN_SERVER_SOCKET" --follow -o /tmp/dayan-mega-bf16.json -- \
    python3 tools/dayan-trace/limited-perf.py run \
        --model qwen35-mega-bf16 --client build-dayan/bin/test-dayan-qwen35-dense \
        --simulator cmodel --socket "$DAYAN_SERVER_SOCKET" \
        --perf-trace /tmp/dayan-mega-bf16.json --output build-dayan/perf-mega-bf16 \
        --prefill 4 --context 16 --cmp --cmp-seed 35
python3 tools/dayan-trace/limited-perf.py report --output build-dayan/perf-mega-bf16
```

For MXFP4 MoE, use `DAYAN_MOE_MODEL` from the fixture step:

```sh
"$DAYAN_ROOT/build/host/dyperf/dyperf" \
    --socket "$DAYAN_SERVER_SOCKET" --follow -o /tmp/dayan-mega-moe.json -- \
    python3 tools/dayan-trace/limited-perf.py run \
        --model qwen36-35B-A3B-moe-mega --client build-dayan/bin/llama-dayan-moe-bench \
        --simulator cmodel --socket "$DAYAN_SERVER_SOCKET" \
        --perf-trace /tmp/dayan-mega-moe.json --output build-dayan/perf-mega-moe \
        --prefill 4 --context 2 --cmp --cmp-seed 35
python3 tools/dayan-trace/limited-perf.py report --output build-dayan/perf-mega-moe
```

Mega presets require cmodel, `--repeats 1`, and `--warmup 0` (the defaults). The MoE sample measures one `llama_decode` call on the two-layer fixture; it is not a 40-layer model estimate. cmodel wall times and counters do not establish hardware or RTL performance.

### Benchmark full-model inference

Stop the profiler, then run each desired model sequentially. These commands measure real GGUF prefill and decode with mega enabled:

```sh
# MXFP4 dense
GGML_DAYAN_MEGA=1 ./build-dayan/bin/llama-bench -m "$DAYAN_MODEL_MXFP4" \
    --device DAYAN -ngl 99 -fa on -p 128 -n 16 -b 128 -ub 128 -r 1 --no-warmup

# BF16 dense
GGML_DAYAN_MEGA=1 ./build-dayan/bin/llama-bench -m "$DAYAN_MODEL_BF16" \
    --device DAYAN -ngl 99 -fa on -p 128 -n 16 -b 128 -ub 128 -r 1 --no-warmup

# MXFP4 MoE, with the 32 GiB server
GGML_DAYAN_MEGA=1 ./build-dayan/bin/llama-bench -m "$DAYAN_MODEL_MOE" \
    --device DAYAN -ngl 99 -fa on -p 128 -n 16 -b 128 -ub 128 -r 1 --no-warmup
```

Use `GGML_DAYAN_MEGA=0` for standalone comparisons. `--no-warmup` keeps smoke runs short; omit it and increase repetitions for warm measurements. On cmodel, wall time and tokens/s describe simulator execution.

## Runtime topology and task dimensions

The backend queries `npu_runtime_get_topology` once when it opens the device. The SDK headers, Runtime library, and `dayan_server` must support this API. The query returns physical topology and the Server's enabled Cluster/Core masks. Initialization fails if the query is unavailable or invalid.

A cooperative kernel requires all four cores of an enabled Cluster. `TaskDim.x` stays 4, and `TaskDim.y` specifies the number of logical four-core groups. The Server places each group within one Cluster; logical group IDs are not physical Cluster IDs. `TaskDim.z` stays 1.

| Enabled topology | Cooperative kernel | Non-cooperative kernel |
| --- | --- | --- |
| 1 core | 1c, `(1,1,1)` | `(1,1,1)` |
| 1 complete Cluster, 4 cores | 4c, `(4,1,1)` | `(4,1,1)` |
| 2 complete Clusters, 8 cores | 4c, `(4,2,1)` | `(8,1,1)` |
| 4 complete Clusters, 16 cores | 4c, `(4,4,1)` | `(16,1,1)` |
| 2 Clusters with 3 enabled cores each | 1c, `(6,1,1)` | `(6,1,1)` |

MatMul, fused SiLU, paired projections, SWIGLU/GEGLU FFN, MoE, MXFP4, and gated output projections have non-cooperative fallbacks. FlashAttention uses its general 1c entry when four-core prefill or split-KV decode is unavailable. Standalone GLU support still requires the existing graph fusion pattern; topology selection does not add a generic standalone SWIGLU kernel.

Independent workers and cooperative groups partition output columns; attention projections keep whole heads within each partition. MXFP4 partitions output tiles, and FlashAttention partitions heads and batches. Gated output projections reserve separate staging storage for each worker or group. Native operators use the enabled core count, with Gated Delta Net and SSM convolution also limiting tasks to available work.

## Backend behavior

The backend supports contiguous `GGML_OP_MUL_MAT` with BF16 or FP16 weights, including broadcastable `ne2` and `ne3` batches. Activations are stored with the same type as the weights, matrix accumulation uses F32, and an unfused matrix multiplication returns the F32 result required by ggml.

Dense MXFP4 MatMul and FFN use `ggml-dayan-mxfp4.h`. MXFP4 weights must have one matrix batch, contiguous rows, and K divisible by 32. Activation batches are flattened into M. Attention operators and KV storage keep their existing floating-point paths. MXFP4 expert matrices use the same payload and scale layout, with one product per selected expert. Resource phases persist across these products, and partial scale stores write only the active rows.

The GGUF representation remains the standard 17-byte block: one E8M0 scale followed by 32 FP4 values. Upload splits this into consecutive low-nibble-first FP4 payload and a separate scale view stored in `tensor->extra`. Readback restores the GGUF representation, including partial transfers. The arena allocation includes payload row padding and all scale planes, so the graph allocator owns both lifetimes.

`GET_ROWS` accepts MXFP4 source weights and writes F32, BF16, or FP16 embeddings. It uses the same packed payload and Z-scale allocation as MatMul, so a tied embedding/LM head can remain one MXFP4 tensor. Each task loads up to 1024 selected-row elements plus aligned 32-byte scale packs. One VEC block expands FP4 nibbles and applies E8M0 scales; BF16/FP16 conversion and packing remain in registers. Dim2 covers 64-element groups, and LSU stores the complete output tile. The 8 KiB task scratch supports four resident Procs. I32/I64 indices, repeated selections, batched indices, and K32 tails are supported. Eligible residual/norm consumers retain the existing BF16/FP16 storage rewrite.

For either operand, let `r` be the A row or B column and `g` the K32 group. The scale offset is `(g / 4) * scale_stride + r * 4 + g % 4`, with `scale_stride = align_up(rows_or_columns, 8) * 4`. Eight lanes occupy 32 bytes in each K128 plane. Scales are unswizzled; only matrix payload follows the selected MAT input swizzle. The load-a and load-b threads load both payload and scales before publishing their Resource. The paired gate/up products share A and release its Resource after both MAT reads.

MXFP4 tiling follows `npu-firmware/demo/matmul_fp4_4c.cpp`: each core keeps two A buffers and splits the macro tile into two N slices, each with its own B and F32 accumulator buffers. A is reused across both N slices and, for FFN, both gate/up products. The four cores own independent output macro tiles. `DayanMxfp4Tile` accounts for payload, scale planes, accumulators, alignment, packed output, and a 64-byte guard within 1536 KiB per core:

| Large-M path | Macro M x N x K | MAT slice M x N x K | URAM used | Capacity used |
| --- | --- | --- | --- | --- |
| MXFP4 MatMul, F32 output | 512 x 512 x 640 | 512 x 256 x 640 | 1534.06 KiB | 99.87% |
| MXFP4 MatMul with output conversion | 512 x 512 x 512 | 512 x 256 x 512 | 1500.06 KiB | 97.66% |
| MXFP4 gate/up + activation + conversion | 256 x 512 x 512 | 256 x 256 x 512, twice | 1466.06 KiB | 95.45% |

M capacity is aligned to 32 rows, capped at 512 for MatMul and 256 for paired FFN, and shrinks for smaller inputs. Below 128 rows, N targets one quarter of the output columns, rounded up to 256 and capped at 3072; MXFP4 K grows to 1024, 2048, or 4096 when useful and within budget. BF16/FP16 producers retain K=512 and reduce M in 32-row steps to fit their wider inputs. Actual tail instructions use valid M/N/K, including single rows and K32 tails.

The post thread submits one VEC block for the entire MAT slice. For MXFP4 output, Dim2 spans physical 128-element blocks and Dim3 spans rows: 1024 independent iterations for a 512 x 256 MatMul slice and 512 for a 256 x 256 FFN slice. Each iteration reads its whole C block and any gate/up inputs before overwriting that C block. Activation, BF16 rounding, and MXFP4 conversion stay in VEC registers. The C swizzle only permutes 128-byte groups within rows, so each K32 scale group remains intact. A 64-byte VEC channel stride leaves each group's 16-byte payload and scale at offsets 0 and 64 in its original 128-byte C region. BF16/FP16 output uses 64-element iterations and leaves 64 packed bytes in each original 128-byte group.

Two LSU URAM copies gather the entire MXFP4 slice into a compact payload buffer and a Z-packed scale buffer, removing the C swizzle. After those copies complete, one LSU instruction writes all payload rows and one writes all scale planes. A full 512 x 256 slice writes 64 KiB of payload and 4 KiB of scales; a 256 x 256 slice writes 32 KiB and 2 KiB. Tail stores include only allocated K128 payload padding and eight-row scale-owner padding. The C Resource remains held through LSU completion, and the shared packed buffers are reused only after the previous store completes. BF16/FP16 output stores the whole slice directly from C; a partial K32 column group uses one additional tail instruction.

Standalone conversion uses 128 x 512 tiles, one VEC block per tile, and the same whole-tile payload/scale stores. FP16 widening and BF16 rounding also stay in registers before MXFP4 conversion. Each Proc reserves 296 KiB, allowing all four independent Procs to reside within 1536 KiB. URAM A scale planes use `align_up(M * 4, 64)` as their K128 stride so LSU's 64-byte tail zero-fill cannot overwrite the next plane; the external Z layout is unchanged.

All MXFP4 pipeline buffer bases are at least 32-byte aligned. The host arena and payload rows use 64-byte alignment, and external K128 scale planes use 32-byte alignment. URAM A/B/C buffers have 1024-byte-aligned bases and sizes; scale buffers, packed payload, and packed scales meet 32-byte alignment. Device entry points check the URAM buffer bases. Individual packed scale bytes keep their Z layout.

VisualSwizzle was evaluated with 32 banks, 32-byte words, eight rows per cycle, actual buffer offsets, and FP4 64-byte or floating-point 32-byte row accesses. A/B strides 256 use M/B/S=(7,1,3), 320 uses (9,1,1), 512 uses (7,2,3), 1024 uses (7,3,3), and 2048 uses (7,3,4). C uses the same rule for its power-of-two row-stride factor; the default 1024-byte C stride uses (7,3,3). All selected layouts have peak bank pressure 1 and zero excess requests. Four-block coverage is 28/32 for the 320-byte input stride and 32/32 for the other MAT layouts. The 320-byte mapping stays within 1024-byte windows; 32-row M alignment preserves its buffer boundaries even for tails. LSU input writes use the same mapping as MAT reads. VEC processes physical C groups in place, and LSU gathers or stores restore logical column order.

The rewrite shares activation casts and fuses eligible MatMul or SwiGLU/GeGLU producers with an MXFP4 output cast. Dependencies on the final fused node keep input payload and scale planes live through the post stage. Without these dependencies, the compiler leaves the operations separate. BF16/FP16 producers can also write MXFP4, and MXFP4 MatMul can write BF16/FP16 at an attention boundary. The post thread converts each output tile before its DRAM store. A standalone VEC conversion handles inputs whose producers cannot be fused. BF16 groups use the hardware MXFP4 conversion directly; F32 and FP16 inputs first round to BF16. Hardware conversion uses the smallest E8M0 scale that covers the finite normal group maximum and truncates E2M1 magnitudes. This differs from GGML's generic MXFP4 activation quantizer.

The SDK must include the MXFP4 correction in `dayan_intrinsics.h`: `__tx_vec_cvt_mxfp4_bf16<0, 0, 0>()` must encode `0x8027800b`. The older Clang builtin emits `0x8023800b`, which does not select the conversion instruction class. This uses the same SDK correction mechanism as the BF16/FP16 conversion wrappers.

Build `test-dayan-mxfp4` and run it against an isolated cmodel server. It checks scale Z layout, partial transfers, VEC conversion, matrix tails, repeated resource reuse, shared casts, fused FFN, and floating-point output boundaries. `test-dayan-mxfp4 --host-only` checks packing without a device. `test-dayan-mxfp4 --get-rows` checks quantized embeddings and the low-precision storage rewrite. The limited operator profiler also accepts `--cmp` for CPU gold comparisons; see [limited-perf.md](../../../tools/dayan-trace/limited-perf.md).

Before scheduling a llama graph, the backend rewrite inserts one matching BF16 or FP16 `CPY` for each unique F32 activation used by an eligible matrix multiplication. Matrix multiplications that share an activation and storage type also share the cast tensor. Direct ggml users that do not run the rewrite can still provide F32 activations; the Dayan graph executor converts each unique activation once.

The rewrite moves a cast before an exclusive metadata-only `VIEW`, `RESHAPE`, `PERMUTE`, or `TRANSPOSE` chain when this does not remove an F32 output or side consumer. It rebuilds the aliases in the low-precision type, and the graph compiler finds the MatMul producer and cast by dependency instead of node adjacency. This keeps the cast in the MatMul epilogue slot without producing both F32 and low-precision matrix outputs.

Each ggml Dayan arena has one Device allocation, and each tensor owns a Device view at its planned arena offset. Graph kernels pass these views directly to their consumers. Intermediate results remain on Device; Device-to-Host transfer occurs only through an explicit tensor read or a scheduler boundary.

URAM bypasses the CPU cache. CPU accesses to URAM need no cache flush or invalidate. LSU, MAT, and VEC dependencies still require the corresponding unit waits or Named Resources. Cache maintenance applies to cached DRAM, including shared kernel configurations.

The graph executor lowers a Dayan split to execution slots. Two adjacent matrix multiplications with the same activation use one pair slot and one activation upload. A matching `MUL_MAT + MUL_MAT + SWIGLU/GEGLU + CPY` sequence uses one fused slot and the dedicated FFN pipeline in `ggml-dayan-ffn.h`. Each K step loads gate and up weight rows into one B tile, then issues one MAT operation per slice. Each F32 accumulator row holds the gate columns followed by the up columns. N tails move the up half to the next 64-column boundary and reduce MAT_N. VEC hardware loops cover rows and 64-column chunks, evaluate SwiGLU or GeGLU, convert to BF16/FP16, and use Shuffle Compress with a half-vector store mask to write packed URAM output directly. LSU writes only valid columns to DRAM; no second URAM packing copy is needed.

Both one-core and four-core FFN entries use the same shape-dependent pipeline and a 1536 KiB URAM budget. Four-core FFN partitions N evenly and broadcasts A from core 0. For M<=128, each output tile has a fixed 128-row A allocation and two N slices of 192 output columns; each B slice holds 192 gate and 192 up columns, and A is double-buffered. For M>128, each output tile has two 256-row M slices; B is double-buffered with 128 gate and 128 up columns per buffer, shared by the active M slices. MAT and LSU process only actual rows, including M<32. Gate and Up loads share one Resource, and the buffered operand is released only after all active slices consume it. Each slice has one C allocation. A single packed output buffer is reused after LSU finishes its store. Down projection uses the following MatMul slot and returns F32. It has no GLU activation; matching residual ADD, RMSNorm, scale, and conversion operations use the subsequent normalization pipeline slot.

The MoE path has separate kernels for `MUL_MAT_ID`, `ADD_ID`, `ARGSORT`, and `TOP_K`. The graph compiler also recognizes `MUL_MAT_ID + MUL_MAT_ID + SWIGLU/GEGLU + CPY`. This fused slot selects both expert matrices and runs the same dedicated FFN pipeline for each selected expert. Expert ID views are read with their actual row strides, and routing rows support up to 4096 entries.

F32 matrix products, including the expert router and scalar shared gate, execute on MAT as TF32 inputs with F32 accumulation and output. The TF32 tile is M=32, N=64, K=256 with 1152-byte input strides and a 384-byte output stride. VisualSwizzle checks of A, B, and C give peak bank multiplicity 1 with identity swizzle.

Graphs containing `MUL_MAT_ID` retain F32 residual storage. Rounding the residual stream to BF16 can change close router scores and select different experts in later layers.

`ARGSORT` and `TOP_K` use a stable 32-bit FlexVec radix sort. VEC transforms float keys, computes prefix sums and bucket totals with Shuffle, and scatters keys and indices into SparseMem. Prefix carries stay in URAM and return directly to VEC; scalar code only controls loops. Signed zero is canonicalized and equal keys retain index order. Prefix consumers store only the requested indices with the original row stride. `GET_ROWS` accepts these strided I32/I64 views. Sorting uses independent Procs with no child Threads or Named Resources.

F32 binary operators read strided rows directly, including the expert output views used in MoE aggregation. Inputs with a noncontiguous first dimension still use Device copy staging. CLAMP accepts infinite bounds, including the lower-only clamp used to normalize selected routing weights.

The one-core and four-core MatMul entries use the same pipeline: two input loader Threads, the root MAT worker, and a post Thread. The general four-core loaders use DAPI Resource producer election to share A across columns and B across rows. Local Resources protect MAT inputs and accumulators; Shared Resources keep each source alive until the peer copy finishes. Each worker tracks Resource phases across batches and repeated expert products. LSU supports partial output rows, including the single-column shared expert gate.

The `matmul[_quantized|_silu]_inference_*` entries specialize small-M inference. For `1 <= M <= 512`, the graph executor selects the inference entry; pair, expert, and attention projection slots use the same layout internally. Four-core inference assigns core `c` the contiguous N interval `[N*c/4, N*(c+1)/4)`, so shard widths differ by at most one column. Core 0 loads each A tile once and broadcasts it through URAM, with three Shared Resource copy credits. Each core loads its own weight rows and computes the full M range. Cores with empty N shards still retire their A copy credits. Larger MatMuls retain the general 2x2 cooperative layout; FFN keeps its N partition.

For `M < 32`, both one-core and four-core inference use exactly M rows per A tile, including odd sizes. The A row stride is 512 bytes (`K_tile=256`, BF16/FP16), so the two A buffers occupy `2*M*512` bytes; decode with M=1 uses 1 KiB. No padding to 32 or 128 rows is needed. For the MatMul entries, the small A allocation leaves room for larger B slices: M<32 uses four 384-column slices on four cores, or four 192-column slices on one core. M>=32 uses four 96-column slices. Partial K and N dimensions go directly to MAT, and LSU stores only valid output bytes.

VisualSwizzle selects `M=7,B=2,S=3` for the 512-byte A/B row strides. Plain C uses linear storage at a 384-byte row stride and `M=7,B=2,S=3` at a 1536-byte row stride. Epilogues use linear C with a 640-byte or 1664-byte row stride. Small-M MatMul inference uses a linear C row stride of 896 bytes on one core or 1664 bytes on four cores. Dedicated FFN uses a 1664-byte linear accumulator row stride for 192+192 columns or a 1152-byte stride for 128+128 columns, including 32 padding floats, so MAT and VEC share the same layout. For full eight-row access groups, the selected FFN A/B mapping reduces peak Bank pressure from 4 to 1; linear FFN C already has peak pressure 1. Both cover all 32 Banks across four consecutive 32-byte blocks. The FFN buffers use 976 KiB per core for M<=128 and 1152 KiB for M>128, within the 1536 KiB budgets. These are address-mapping results, not measured hardware throughput.

Contiguous F32 residual adds and RMSNorm execute on Dayan. One parameterized normalization pipeline slot handles optional residual ADD, RMSNorm, scale multiplication, and BF16/FP16 quantization. Matrix multiplication remains a separate slot. The graph rewrite keeps every fused input live through the final logical node so the allocator cannot overlap a cast output with an input that another native Task still reads. VEC tensor dimensions expand complete 64-lane chunks into padded URAM scratch space, while LSU tail stores limit the final DRAM write to the valid byte count.

The attention operator package provides dedicated BF16 and FP16 QG, K, and V projection entry points. Q and K fuse low-precision matrix writeback, RMSNorm, RoPE or MRoPE, and output layout conversion. K and V also have indexed KV-cache variants that write directly to cache rows. Flash-Attention has native BF16 and FP16 entries, uses `dayan_uram_base()` directly, and has an explicit causal offset for KV-cache decode. It supports GQA and right-aligned causal masking. The output path fuses `attention * sigmoid(gate)`, low-precision quantization, and the output projection. The graph compiler recognizes the Qwen3.5 gate view, sigmoid, multiply, and output matrix multiplication pattern.

Flash-Attention uses 128-query tiles and 128-key tiles. Each core owns two Q slices; Q remains resident across the KV traversal. K and transposed V each have two buffers. Four-core mode splits Q tiles between cores and elects one active core to load each K or V tile, then copies its URAM tile to the other active cores. Every core retires its Shared Resource credit, including cores with no query rows. Single-core mode uses private KV loaders. The graph executor selects four-core mode for more than 64 query rows when a complete Cluster is enabled, and single-core tasks otherwise.

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

When convolution consumes `concat(history, transpose(qkv))` with a final history copy, the graph compiler folds both memory operations into `ssm_conv_split_f32_1c`. The kernel loads raw QKV rows directly into URAM and writes the final history. This applies to standalone and mega execution; graphs with other concat consumers keep the ordinary path. General concat copies merge contiguous dimensions and use LSU Transpose for F32/F16/BF16 matrix transposes.

Gated Delta Net assigns heads and sequences to independent single-core tasks, limited by the enabled core count. Scalar-gate prefill with at least 16 tokens and one final state snapshot uses 64-token chunks. Each task has a loader Thread, a precompute Thread, and the root state-update loop. Two input slots and two prepared slots use four Local Resources for backpressure. Precompute builds the gate coefficients and jointly solves U and W with 256 independent RHS rows. The root computes the corrected values as U - W S, then produces outputs and updates the FP32 state. MAT, VEC, and LSU work from different chunks can overlap. Resource phases continue across heads and sequences; there is no separate limit on how far a producer can advance.

Chunk MAT uses `MAT_DATATYPE_PAIR(DT_MATIN_TF32, DT_MATIN_TF32)`. The inputs have FP32 storage and TF32 precision; accumulation and VEC arithmetic remain FP32. This requires firmware headers and cmodel with the TF32 operand encoding enabled. The pipeline uses 758.5 KiB within a 768 KiB task reservation, with up to two resident tasks per core. Head and chunk row strides are 576 and 320 bytes, so LSU transposes remain 64-byte aligned. All units use identity swizzling; VisualSwizzle reports peak Bank pressure 1 and no extra requests for these layouts. Padding and the triangular mask handle chunk tails, and unused gate differences are masked before exponentiation. Prepared Q becomes output storage after its last MAT read, and the prepared Resource stays held until the output DMA completes.

Decode, short prefill, vector gates (KDA), and multiple state snapshots use the FP32 recurrent VEC path with a 96 KiB reservation and up to four resident tasks per core. It forms the gated key once per token and folds state decay into the state update, avoiding a separate full-state read/write pass. DIM2/DIM3 provide 16 independent contexts per head, each processing eight state rows. Both entries can distribute independent tasks across the available cores; they do not split one head across a cooperating four-core group.

## Profiling

Device operators contain no custom cycle counters or performance buffers. The [operator benchmark commands](#operator-benchmarks-with-limited-perfpy) collect firmware scheduling traces for MXFP4 dense, BF16 dense, and MXFP4 MoE. The [full-model commands](#benchmark-full-model-inference) measure GGUF inference with the profiler stopped. Record the model, prompt/generation lengths, batch sizes, repetitions, and simulator configuration.

For an existing RTL server, operator presets also accept `--simulator testcluster` with that server's socket. This labels the report; it does not start TestCluster. Mega presets currently require cmodel. See [limited-perf.md](../../../tools/dayan-trace/limited-perf.md) for measurement boundaries and extrapolation factors.

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

The backend selects kernels from the cores enabled in `dayan_server`; core mask `0xf` permits four-core cooperation, and a partial mask selects non-cooperative kernels. The same client commands work with a TestCluster-backed server.

To run cmodel alongside TestCluster, isolate its FUSE mount, Runtime socket, emulator IPC sockets, and DRAM/URAM shared-memory names. Separate mount and Runtime socket paths alone are insufficient. The stock `run_cmodel_server.sh` cleans the default IPC and shared-memory files, so do not use it while TestCluster owns those resources. Use an isolated emulator/QEMU pair and a launcher whose cleanup touches only that instance's paths. Clients select the instance through `DAYAN_SERVER_SOCKET`.

Replace `test` with `support` to check whether Dayan supports the selected cases without running their numerical comparison. `perf` runs the performance case set, which has different shapes and may require a different `-p` filter; host timings under simulation are not hardware throughput measurements.

## Dedicated tests

`test-dayan-topology` checks launch selection for 1, 4, 8, and 16 cores, partial masks, and output partition boundaries without a device. `test-dayan-attention-projection launch` checks independent workers and multiple four-core groups when available. The attention pipeline cases `groups-prefill-bf16`, `groups-decode-fp16`, and `tasks-prefill-fp16` cover group and worker sharding. Multiple logical groups can run in sequence on one physical Cluster; this does not validate physical multi-Cluster concurrency.

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
| `ssm-split` | Raw QKV convolution, final state writeback, two rounds, and concat fanout |
| `concat` | Contiguous and transposed inputs, all axes, F32/BF16/F16 |
| `gated-output` | Gated attention output projection |
| `copy` | Device copy paths |
| `get-rows` | Indexed row reads |
| `embedding-norm` | Embedding lookup and normalization |
| `memory` | Memory and layout operators |
| `empty` | Empty Scale, memory operators, and BF16/FP16/MXFP4 output MatMul |

The standalone test executables below have matching CMake build targets, for example `cmake --build build-dayan --target test-dayan-attention-projection -j`.

The `glu` selector checks generic and inference 1c/4c SWIGLU and GEGLU entries in BF16/FP16. Cases cover M=1,2,3,7,8,15,16,31,32,128,129,257,512,513, N shards with different widths or no columns, K tails, repeated output tiles, repeated launches, and broadcast batches with padded row and batch strides in both slice regimes. N=769 exercises one or two active small-M N slices across different cores. M=513 exercises two M slices followed by one active tail slice. Output padding and trailing guards must remain unchanged. The MatMul test also checks the M=512/513 dispatch boundary and packed broadcast batches above 512 rows.

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-attention-projection glu
```

The attention projection test runs QG, K, and V projection, indexed KV-cache writes, causal Flash-Attention, gate fusion, low-precision quantization, and output projection as one numerical data flow for BF16 and FP16 model weights:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-attention-projection
```

The Gated Delta Net test compares outputs and state snapshots with the CPU reference. It covers decode, 16/17/64/65/128/129/257/1024-token prefill, weak and strong gates, shared Q/K heads and sequences, odd chunk counts across repeated task work, KDA, and multiple state snapshots. Chunk tests check TF32 error separately for outputs and final state; recurrent tests retain FP32 tolerances.

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

The Qwen3.5 dense test includes the BF16 single-token linear regression with seed 35, then four-layer BF16/FP16 hybrid cycles with three gated DeltaNet layers and one full-attention layer. Every layer also contains RMSNorm, dense SWIGLU FFN, and residual paths. It compares Dayan outputs and final logits against a CPU reference. Gated Delta Net uses a 128-element head so the test exercises the device kernel instead of CPU fallback. For the single-case JSON command, see [BF16 dense mega checks](#bf16-dense).

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-qwen35-dense
```

The BF16 MoE routing regression covers 512 experts, a narrow `ARGSORT` view, `TOP_K`, fused SWIGLU expert projections, the down expert projection, `ADD_ID`, and expert-weighted aggregation. For MXFP4 expert and TF32 router CPU comparisons, use the [MoE operator preset](#operator-benchmarks-with-limited-perfpy); for the random hybrid graph, use [MXFP4 MoE mega checks](#mxfp4-moe).

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-moe
```

Both matrix launch modes use four consumer slices. `MUL_MAT_ID` still requires the expert output width to be a multiple of 16. Arbitrary attention masks, attention sinks, and Flash-Attention soft caps are not implemented. Flash-Attention has native BF16 and FP16 entries and requires Q/K/V to share one storage type. Full-model inference requires enough Device memory for all persistent weights and graph arenas. The launcher defaults to 4096 MiB; the 35B-A3B MoE commands above use 32768 MiB. A llama model using Dayan rejects any non-metadata model graph node assigned to another backend. Sampling stays on CPU and is not inserted into the Dayan graph.
