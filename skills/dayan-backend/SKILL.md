---
name: dayan-backend
description: Guidance for working on the Dayan backend in llama.cpp.
---

# Dayan backend agent context

This directory is a progressive context package for work on the llama.cpp Dayan backend. It summarizes the framework-specific knowledge needed for graph and operator optimization without loading the full repository into an agent context.

The documents describe two different things:

- A contract or invariant is behavior that an implementation must preserve.
- A current implementation note describes this checkout and must be checked against source before editing.

Do not treat this directory as a replacement for reading the code being changed. Use it to select the small set of source files that matters for a task.

## Reading routes

Always start with [quick-context.md](references/quick-context.md).

For graph rewrite or fusion work, then read:

1. [framework-and-memory.md](references/framework-and-memory.md)
2. [graph-optimization.md](references/graph-optimization.md)
3. [qwen35-dense.md](references/qwen35-dense.md) when the pattern comes from Qwen3.5
4. [validation-and-performance.md](references/validation-and-performance.md)

For a Device operator or kernel, then read:

1. [operator-and-kernel.md](references/operator-and-kernel.md)
2. The operator-specific header under `ggml/src/ggml-dayan/`
3. [validation-and-performance.md](references/validation-and-performance.md)

For FlashAttention, read:

1. [flash-attention.md](references/flash-attention.md)
2. [operator-and-kernel.md](references/operator-and-kernel.md)
3. [qwen35-dense.md](references/qwen35-dense.md)
4. [validation-and-performance.md](references/validation-and-performance.md)

Use [source-map.md](references/source-map.md) to find exact source symbols and tests.

## Working rule

Before changing an operator, write down its tensor contract in this order:

1. GGML operation and source indices.
2. Type of every source and output.
3. `ne[]`, `nb[]`, view offset, contiguity, and alias rules.
4. Prefill, decode, batch, head, and tail shapes that are supported.
5. Host config fields and their units.
6. Device memory layout, URAM budget, task topology, and completion waits.
7. CPU reference and tolerance used by an existing test.

If any item is unknown, read the relevant source before editing. A successful Device launch is not proof that the contract is correct.

## Repository constraints

- Follow the root `AGENTS.md` and `CONTRIBUTING.md`.
- Keep code and comments ASCII-only.
- Prefer existing backend infrastructure and existing tests.
- Keep comments short and explain only non-obvious invariants.
- Do not add a new dependency for Device instruction encodings. The public headers in `npu-firmware/device/include/api` are the source of truth.
- Dayan-emu is for functional and relative before/after comparison. It is not RTL or hardware timing.

## Keeping this package current

When a change alters a stable contract, update the relevant document. When only a helper name, tile size, or implementation strategy changes, update the source map or current implementation note without expanding `quick-context.md`.
