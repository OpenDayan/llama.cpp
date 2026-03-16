# NPU Backend Migration Acceptance Checklist

## 0. Meta
- Source branch: `npu-backend-ng` (custom repo)
- Target branch: `npu-backend-ng-migrate` (migrated repo)
- Target base: latest upstream official `master`
- Scope:
  - `ggml`
  - `gguf-py`
  - `include`
  - `src`
  - `tests`
  - `tools`
  - `batched-bench.sh`
  - `convert_json.py`
  - `run_npu.sh`

## 1. Branch and History
- [ ] Migration branch is created from migrated workspace HEAD.
- [ ] Source custom commits are audited against migration branch.
- [ ] No required source commit is missing from migration branch.
- [ ] Branch naming/versioning convention is documented.

Evidence:
- `git log --oneline --decorate -n 50`
- `git branch --show-current`
- `git merge-base --is-ancestor <src_commit> HEAD`

## 2. Build and Link
- [ ] `GGML_RNPU=OFF` build succeeds.
- [ ] `GGML_RNPU=ON` build succeeds.
- [ ] Required runtime/kernel paths are configurable and documented.
- [ ] No unresolved symbol/linker error for RNPU backend.

Suggested commands:
```bash
cmake -S . -B build -DGGML_RNPU=OFF
cmake --build build -j
cmake -S . -B build-rnpu -DGGML_RNPU=ON -DRNPU_RT_PATH=/workspace/npu-runtime -DRNPU_KERNEL_PATH=/workspace/npu-kernels
cmake --build build-rnpu -j
```

## 3. Runtime Smoke Tests
- [ ] `llama-batched` runs successfully with RNPU enabled.
- [ ] `llama-batched-bench` runs successfully with `-ngl 28`.
- [ ] `llama-batched-bench` runs successfully with `-ngl 29`.
- [ ] No crash/assert in KV-cache, graph build, or backend dispatch path.

Suggested commands:
```bash
./build/bin/llama-batched -m <model.gguf> -p "hello" -np 1 -ctk bf16 -ctv bf16 -ngl 29 -fa off
./build/bin/llama-batched-bench -m <model.gguf> -npp 6 -ntg 5 -npl 4 -s 1 -ngl 28 -ctk bf16 -ctv bf16 -fa off -c 4096
./build/bin/llama-batched-bench -m <model.gguf> -npp 6 -ntg 5 -npl 4 -s 1 -ngl 29 -ctk bf16 -ctv bf16 -fa off -c 4096
```

## 4. Functional Correctness
- [ ] Same prompt on CPU and RNPU has no obvious semantic regression.
- [ ] No repeated-garbage output pattern under RNPU in short generation.
- [ ] KV-cache write/read path behaves correctly for non-flash attention.
- [ ] Multi-sequence bench path does not trigger stream/layout assertions.

Evidence:
- Save logs for at least 3 prompts and 2 bench configurations.
- Attach diffs/snippets for output comparison.

## 5. RNPU Backend Coverage
- [ ] Core ops are covered and dispatched correctly:
  - [ ] `GET_ROWS`
  - [ ] `SET_ROWS`
  - [ ] `MUL_MAT` (BF16 path)
  - [ ] `PERMUTE` (view semantics respected)
  - [ ] `ROPE`, `SOFTMAX`, `RMS_NORM`, `SILU`
- [ ] Unsupported type paths fail clearly with actionable message.
- [ ] Backend init gracefully handles no-device / runtime load failure.

Evidence:
- `ggml/src/ggml-rnpu/*.cpp`
- operator runtime logs

## 6. Data Type and Quantization Compatibility
- [ ] BF16 main path validated end-to-end.
- [ ] Legacy `E4M3_Q`/`MXFP8` behavior documented for upstream compatibility.
- [ ] Any temporary fallback or TODO is tracked as issue.

Notes:
- If upstream type system differs from custom branch, mark gap explicitly and provide migration plan.

## 7. Performance and Stability
- [ ] Prompt and decode throughput recorded (`t/s`).
- [ ] Memory footprint recorded (model buffer, KV buffer, compute buffer).
- [ ] No progressive slowdown or resource leak in repeated runs.
- [ ] Long run (>= 3 repeated invocations) stability is validated.

## 8. Repo Hygiene
- [ ] `git status` clean before release tag.
- [ ] Debug-only code/logs are reviewed.
- [ ] Artifacts not meant for version control are excluded.
- [ ] Commit message explains migration scope and known gaps.

## 9. Release Gate
- [ ] All mandatory checks above are marked complete.
- [ ] Known gaps are listed with owner and follow-up milestone.
- [ ] Branch pushed to custom remote and visible to reviewers.

Approval:
- Reviewer:
- Date:
- Decision: `PASS` / `PASS_WITH_GAPS` / `FAIL`
- Follow-up issues:
