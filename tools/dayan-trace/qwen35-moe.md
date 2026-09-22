# Qwen3.5/3.6 MoE on Dayan

The tested source is `Qwen3.6-35B-A3B-MXFP4_MOE_BF16.gguf`, architecture `qwen35moe`: 40 layers, hidden width 2048, 256 experts, Top-8, expert/shared FFN width 512, 30 linear-attention layers and 10 full-attention layers. Attention and shared expert weights are BF16; routed experts are MXFP4; both routers are F32.

The backend supports these operators. F32 matrix products use MAT TF32 with F32 accumulation. Routed products use the existing W4A4 pipeline and actual expert-ID row strides. Standalone sorting uses a stable four-pass scalar radix sort in each Proc's private URAM, including partial output views. It reserves 96 KiB of URAM and no Atomic Cache. MoE mega selects experts from logits in URAM and builds per-core dispatch tables in a reusable device workspace, without shared atomic counters. Its separate `qwen35_moe_mega` entry combines a BF16 attention segment with routing, experts, shared experts, and the F32 residual. It accepts initial and later decode graphs, including omitted state SCALE slots and contiguous ADD slots lowered as residual adds.

## cmodel and fixture

Use cmodel for functional checks. TestCluster and RTL performance measurements are separate.

```sh
NPU_CMODEL_HARTS=4 NPU_CMODEL_MEMORY_MIB=32768 \
DAYAN_SERVER_SOCKET=/tmp/dayan-moe.sock \
bash ~/npu-firmware/run_cmodel_server.sh
```

Confirm 34359738368 shared DRAM bytes in the startup log. Runtime offsets and mega parameters use 64-bit addresses. Create a smaller random fixture without reading the source weight payloads:

```sh
python3 tools/dayan-trace/create-qwen35-moe-fixture.py \
  --source-model /share/models/Qwen3.6-35B-A3B-MXFP4_MOE_BF16.gguf \
  --expert-type mxfp4 --layers 2 --full-attention-interval 2 --small-vocab \
  --output build-dayan/moe-validation/hybrid.gguf
cmake --build build-dayan --target llama-dayan-operator-bench llama-dayan-moe-bench
```

This fixture keeps the real layer dimensions and 256 experts. It has one linear and one full-attention layer, a 512-token vocabulary, and is about 953 MiB. Using both attention kinds exercises the model's hybrid memory inputs. Random weights test execution, not language quality.

## Standalone operators first

Run the operator preset under firmware `dyperf`:

```sh
~/npu-firmware/build/host/dyperf/dyperf --socket /tmp/dayan-moe.sock --follow -o /tmp/dayan-moe-operators-perf.json -- \
python3 tools/dayan-trace/limited-perf.py run \
  --model qwen36-35B-A3B-mxfp4-moe-bf16 --simulator cmodel \
  --socket /tmp/dayan-moe.sock --perf-trace /tmp/dayan-moe-operators-perf.json \
  --prefill 4 --context 16 --cmp --cmp-seed 35 \
  --output build-dayan/moe-operators
python3 tools/dayan-trace/limited-perf.py report --output build-dayan/moe-operators
```

The profiler writes the trace when the client exits. Regenerate the report after that write; the initial report can return code 2 because cycle records are not available yet.

The preset includes TF32 router/shared-gate products, Top-8 sorting, selected expert gate/up/SwiGLU and down products, BF16 shared experts, attention, recurrence, convolution, normalization, activation conversion, and a vocabulary proxy. Each expert case already executes all eight selected experts; the model factor is 40, not 320. Attention projection and vocabulary terms are width proxies. Small routing reductions, gates, layouts, and MTP are excluded from the estimate.

CPU comparisons decode the stored inputs before F32 computation. TF32 reference inputs truncate the low 13 mantissa bits. MXFP4 activation conversion first rounds to BF16, selects the smallest E8M0 scale covering the group maximum, and truncates E2M1 magnitudes. GGML's generic nearest-value quantizer has different semantics. Do not equate agreement with this reference with agreement to unquantized F32 activations.

## Mega and consecutive decode

After standalone checks pass, use the separate mega client:

```sh
DAYAN_MOE_MODEL=$PWD/build-dayan/moe-validation/hybrid.gguf \
~/npu-firmware/build/host/dyperf/dyperf --socket /tmp/dayan-moe.sock --follow -o /tmp/dayan-moe-mega-perf.json -- \
python3 tools/dayan-trace/limited-perf.py run \
  --model qwen36-35B-A3B-moe-mega \
  --client build-dayan/bin/llama-dayan-moe-bench --simulator cmodel \
  --socket /tmp/dayan-moe.sock --perf-trace /tmp/dayan-moe-mega-perf.json \
  --prefill 4 --context 2 --cmp --cmp-seed 35 \
  --output build-dayan/moe-mega
python3 tools/dayan-trace/limited-perf.py report --output build-dayan/moe-mega
```

Each case runs standalone Dayan and mega from fresh contexts. The prefill case checks the prompt and two later single-token steps. The decode case starts with one token and checks two consecutive decode steps. Every step must launch one mega block per layer. A silent fallback fails the test. Logits are compared at every step. This is a fusion regression against standalone Dayan; CPU operator checks remain separate.

The sample measures one `llama_decode` call, including embedding, final norm/head, host graph work, and two mega launches. Validation and extra decode steps are excluded from that sample. The preset reports this two-layer fixture only; it does not extrapolate a 40-layer model. Its decode sample has two cached tokens. cmodel wall times describe simulator runtime, and its cycle counters are not hardware performance evidence. An exit code of 2 can mean numerical checks passed but usable cycle records are unavailable; inspect the result JSON.

## Validation record (2026-09-19)

On four-core cmodel with 32 GiB DRAM:

| Check | Result |
| --- | --- |
| Operator preset, prefill=4 and decode=1 | 29/29 CPU comparisons passed |
| Grouped GDN, Q/K=16 heads and V=32 heads | Prefill=4 and decode=1 passed; maximum absolute error below 5e-9 |
| MXFP4 expert gate/up/SwiGLU and down, 256 experts and Top-8 | Exact stored-input CPU agreement, including repeated launches |
| TF32 router and shared gate | Maximum absolute error below 1e-8 against TF32-input CPU reference |
| Radix sort, widths 63/64/65/256/4096, three rows, both orders | 10/10 exact index comparisons passed |
| Hybrid mega, prefill=4 then two decode steps | Two mega launches per step; logits bitwise equal to standalone |
| Hybrid mega, initial single token then two decode steps | Two mega launches per step; logits bitwise equal to standalone |
| Full 40-layer trained GGUF, one input token | 40 mega launches, 2476 compute nodes on Dayan, 248320 finite logits |
| Existing MXFP4, BF16 MoE, and Top-K regressions | Passed |
| Existing BF16 full-attention dense mega, one token | Passed |
| BF16 linear dense, one token and seed 35 | CPU comparison passed with mega enabled and disabled after the input-lifetime fix |
| limited-perf Python regressions | 23 passed |

Artifacts are under `build-dayan/moe-validation/`, including `limited-operators-final/report.json`, `limited-mega/report.json`, and per-case results. The initial graphs contain two state SCALE slots; both subsequent decode graphs omit them. The saved Device assembly in `device-review/router-sort.asm` contains MAT MM/MMA for the router and VEC Shuffle/SparseMem scatter for sorting. Bucket totals and prefix carries use VEC loads and stores, with no scalar load of the counter.

The end-to-end runs preceded the final change that moved bucket totals entirely into VEC. After that change, sorting, Top-K, and BF16 MoE regressions passed again; the final operator preset was rerun. End-to-end inference was not repeated.

For the five-token random fixture, standard GGML CPU inference without the Dayan activation rewrites gives final-logit relative L2 error 0.30619 and cosine 0.95259. Both routed FFN stages match exactly when the CPU uses the stored activation values and hardware MXFP4 output conversion. This distinguishes W4A4 quantization error from kernel correctness. These tests do not assess trained-model language quality.

The full GGUF test used token ID 3 and a 21032.24 MiB model buffer. Dayan and standard CPU both selected token ID 16. Final-logit relative L2 error was 0.24337 and cosine was 0.97256; the different activation precision remains visible. The result is saved in `build-dayan/moe-validation/full-model/result.json`. This checks complete-model execution and mega addresses beyond 4 GiB, not language quality.

The dense linear fixture (BF16, one token, seed 35) exposed early buffer reuse in fused FFN operations. Its BF16 input, SwiGLU output, and down-projection output shared an address because input lifetime protection covered only MXFP4. Extending that protection to all supported matmul storage types fixes both standalone and mega execution, without changing slot signatures. The graph has no REPEAT slots. Layer-output maximum error falls from 11.4961 to 0.08724 and logit maximum error from 3.04512 to 0.02247, within the test tolerances. This case is now included in the default dense regression. See `dense-linear-fixed-mega1.result.json` and `dense-linear-fixed-mega0.result.json`.

The standalone diagnostic intentionally disables mega. Its numerical comparison passes, but the mega-only benchmark protocol reports a missing mega launch. `dense-linear-fixed-official.result.json` records the normal benchmark run.
