# Dayan TestCluster status

This file records the Dayan decode validation and performance results available on 2026-08-13. No failed or blocked result below is a valid estimate of correct inference performance.

## Test scope

- llama.cpp revision: `8507e57904baf8a78e8832f04111fe827b07056a`
- npu-firmware revision: `004cd8c80d16695fbf01081a2adc5470952c8650`
- Model: Qwen3.5 0.8B BF16
- Phase: one-token decode after a 128-token prefill
- Trace: `/tmp/dayan-decode-current.json`
- Trace SHA256: `87afa502d81874194ed89c491b0825c1dd1bb654ba89be3c16c8efb04dc3692b`
- Trace contents: 50 unique signatures, 695 calls, and 21 slot kinds
- TestCluster configuration: `--cluster-count 1 --core-mask 0xf`
- Device profile source: `dyperf`
- Profile clock: `cycle_ns=1`

One launch duration is measured from the earliest root Task `RUNNING` state on the four Harts to the last root Task `WAIT_EXIT` state. Child Proc activity is part of that wall-clock span and is not added again.

Cycles from a numerical failure or a failed launch are diagnostic values only. They must not be added to a correct decode estimate.

## Deduplicated Host replay baseline

The trace replay executed each unique signature once and used its observed count to estimate the complete operator total. Output hashes matched between the cold and warm rounds.

| Round | Setup, us | Unique replay wall, us | Measured unique slots, us | Estimated operator total, us | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Cold | 12,148 | 15,940,098 | 15,781,530 | 68,991,890 | Pass |
| Warm | 9,913 | 15,593,415 | 15,421,291 | 76,535,893 | Pass |

The warm baseline is about `76.536 s/decode`. This is a Host slot timer that includes Host preparation, configuration upload or cache lookup, kernel launch, and device wait. It is not a TestCluster RTL cycle result.

The warm Host estimate by slot kind is:

| Slot kind | Trace calls | Estimated Host time, us |
| --- | ---: | ---: |
| `concat` | 18 | 274,518 |
| `copy` | 115 | 2,174,805 |
| `elementwise_mul` | 42 | 1,050,582 |
| `flash_attention` | 6 | 2,465,016 |
| `gated_delta_net` | 18 | 6,630,408 |
| `gated_output_projection` | 6 | 2,054,256 |
| `get_rows` | 74 | 593,734 |
| `l2_norm` | 36 | 2,747,124 |
| `mul_mat` | 121 | 31,544,904 |
| `mul_mat_pair` | 6 | 828,654 |
| `mul_mat_swiglu` | 24 | 11,936,328 |
| `norm_pipeline` | 73 | 9,581,562 |
| `residual_add` | 18 | 411,210 |
| `rms_norm` | 6 | 258,066 |
| `rope` | 12 | 1,593,120 |
| `scale` | 36 | 0 |
| `set_rows` | 12 | 376,506 |
| `sigmoid` | 18 | 299,916 |
| `silu` | 18 | 471,384 |
| `softplus` | 18 | 302,508 |
| `ssm_conv_silu` | 18 | 941,292 |
| Total | 695 | 76,535,893 |

## TestCluster cases that pass correctness

| Operator | Case | Device cycles |
| --- | --- | ---: |
| ADD | F32, 8 elements | 49,373 |
| MUL | F32, 8 elements | 48,803 |
| Sigmoid | F32, 1024 elements | 53,100 |
| SiLU | F32, 1024 elements | 52,222 |
| Softplus | F32, 1024 elements | 53,609 |
| GET_ROWS | Small memory case | 56,435 |
| GET_ROWS | Exact Trace width 262144 | 69,515 |
| CONCAT | Small memory case | 37,222 |

Empty SCALE, GET_ROWS, and COPY slots short-circuit on the Host and use zero Device cycles. The Trace contains 36 empty calls of each kind, or 108 empty calls in total.

An earlier grouped memory run also recorded COPY at 42,581 cycles and SET_ROWS at 42,911 cycles. Later independent tests found numerical failures in both operators, so those grouped-run cycles are retained only as raw historical profiles. The GET_ROWS and CONCAT results from that run pass.

## Partial Trace-weighted TestCluster estimate

The following calculation applies the passing case cycles to the corresponding Trace call counts. Some entries use a small-shape proxy instead of every exact Trace shape.

| Operator | Mapping | Weighted cycles |
| --- | --- | ---: |
| GET_ROWS | 18 exact-width calls plus 20 small-case proxy calls | 2,379,970 |
| CONCAT | 18 calls | 669,996 |
| ADD | 18 calls | 888,714 |
| Softplus | 18 calls | 964,962 |
| MUL | 42 calls | 2,049,726 |
| Sigmoid | 18 calls | 955,800 |
| SiLU | 18 calls | 939,996 |
| Valid subset | 170 Device launches | 8,849,164 |

Including the 108 Host-short-circuited empty calls, this maps 278 of the 695 Trace calls. At `cycle_ns=1`, the valid subset numerically equals about `8.849 ms`. It is a partial proxy, not a lower bound, an upper bound, or a complete decode estimate.

## Confirmed failures and blocked cases

| Operator or path | Result | Device cycles |
| --- | --- | ---: |
| COPY | Independent F32 to F16 test, NMSE `5.369444733` | 107,984 |
| SET_ROWS | Independent F32 to F16 test, NMSE `0.210624640` | 48,996 |
| FlashAttention | `batch=2`, `q_heads=4`, `kv_heads=2`, `q_len=17`, `kv_len=23`, `head_dim=64`, `causal_offset=6`; `max_abs=0.160728`, `max_rel=754.232` | 131,717 |
| Gated Delta Net | `heads=2`, `tokens=1`, `sequences=1`, 33,024 elements; `max_abs=0.027083`, `max_rel=15702.9` | 55,570 |
| SSM Conv | `kernel=1`, `tokens=5`, `channels=128`, `sequences=1`; `max_abs=0.576492` | 50,523 |
| L2Norm | F32 shape `64x5x4x3`; `ERR=0.979991602` | 151,625 |
| RMSNorm | F32 shape `64x5x4x3`; `ERR=0.990382536` | 154,834 |
| RoPE | `mode=0`, `width=130`, `heads=2`, `tokens=3`, `n_dims=64`; `max_abs=0.943548` | 60,243 |
| MatMul | `M=1`, `N=16`, `K=1024`; launch status `-22` | 38,845 diagnostic cycles |
| Norm pipeline | Continuous graph reports ADD `ERR=0.998007101` | 48,496 / 37,329 / 36,297 raw launch cycles |

The MatMul calibration graph also recorded a preceding activation cast at 55,059 cycles. The MatMul launch failed, so neither value forms a valid MatMul performance result.

The gated output projection test is blocked by the first QG MatMul launch with status `-22`. The fused gate and projection body has not received a valid correctness or cycle result.

The real Trace replay fails at its first GET_ROWS launch with:

```text
PROC_CREATE request mismatch
version=0 size=0 state=3 name='get-rows'
```

The launch returns status `-71`. There is therefore no complete real-replay Device cycle count.

## Failure-weighted diagnostic totals

These totals apply the failed case cycles to Trace counts only to show the amount of profiled work. They are not valid inference estimates.

| Operator | Mapped Trace calls | Diagnostic weighted cycles |
| --- | ---: | ---: |
| COPY | 79 non-empty calls | 8,530,736 |
| SET_ROWS | 12 calls | 587,952 |
| L2Norm | 36 calls | 5,458,500 |
| RMSNorm | 6 calls | 929,004 |
| RoPE | 12 calls | 722,916 |
| Gated Delta Net | 18 calls | 1,000,260 |
| SSM Conv | 18 calls | 909,414 |
| FlashAttention | 6 calls | 790,302 |
| Failed numerical subset | 187 calls | 18,929,084 |

The exact MatMul calibration shape occurs 36 times in the Trace. Applying its failed-launch span gives another 1,398,420 diagnostic cycles. Combining the passing proxy subset, failed numerical subset, and this failed MatMul launch gives 29,176,668 raw mapped cycles. This number must not be reported as correct decode latency.

## Current bug assessment

- COPY and SET_ROWS still fail when launched independently. Their errors are not explained only by contamination from a preceding locked Task.
- Independent ADD passes, while ADD inside the continuous normalization graph fails. This points to a locked-task lifetime, cleanup, or state-isolation problem between launches.
- FlashAttention, Gated Delta Net, SSM Conv, L2Norm, RMSNorm, and RoPE complete but produce incorrect numerical results.
- MatMul returns `-22` before a valid result. This also blocks MatMul pair, MatMul SWIGLU, and gated output projection validation.
- Full replay is blocked by the Runtime `PROC_CREATE` request mismatch before useful Device timing can be collected.

## Overall performance estimate

There is no reliable complete TestCluster decode estimate yet.

The only correctness-qualified Device number is the partial proxy of 8,849,164 cycles for 170 Device launches, plus 108 zero-cycle Host short circuits. The 29,176,668-cycle raw mapped total includes known-wrong results and a failed MatMul launch, so it is diagnostic only.

The available full-decode reference remains the warm Host estimate of 76,535,893 us. Its dominant unresolved paths are:

| Path | Host estimate, us | Share of Host total |
| --- | ---: | ---: |
| `mul_mat` | 31,544,904 | 41.22% |
| `mul_mat_swiglu` | 11,936,328 | 15.60% |
| `norm_pipeline` | 9,581,562 | 12.52% |
| `mul_mat_pair` | 828,654 | 1.08% |
| Subtotal | 53,891,448 | 70.41% |

These four paths account for about 70.41% of the Host estimate and do not yet have correct TestCluster cycles. A defensible complete RTL decode estimate requires correct MatMul and normalization results, exact-shape cycles for the remaining signatures, and a successful full Trace replay.
