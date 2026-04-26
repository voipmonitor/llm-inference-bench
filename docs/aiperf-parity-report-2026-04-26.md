# AIPerf Parity Report - 2026-04-26

This note documents what `llm_decode_bench.py` measures, how it aligns with
AIPerf-style finite request bursts, and the current numeric parity checks.

## Methodology

`llm_decode_bench.py` reports three separate benchmark layers. They should not
be mixed in one comparison table because they answer different questions.

| Layer | How to enable | Headline formula | What it answers |
|---|---|---|
| Prefill | default Phase 1, or selected with `--prefill-contexts` | `prompt_tokens / TTFT` | How fast the engine ingests a cold long-context prompt |
| Sustained Decode | default Phase 2, controlled by `--duration N` | OpenAI stream usage `completion_tokens` per measured window; client chunk fallback only if continuous usage is unavailable | How much decode throughput the engine sustains after warmup at a requested concurrency |
| Burst / E2E Decode | optional Phase 3 with `--run-burst`, or primary mode with `--request-count N` | `sum(completion_tokens) / profiling_wall_time` | What a finite client-facing request burst sees end-to-end |

`--request-count N` is the Burst / E2E-only mode. It exists for direct AIPerf
comparisons. For full reports, `--run-burst` is preferred because it keeps the
Sustained Decode matrix and appends a clearly labeled Burst / E2E section.

Client latency metrics use the same formulas as AIPerf in both modes:

| Metric | Formula |
|---|---|
| TTFT | request start to first streamed content token |
| TTST | first streamed content token to second streamed content token |
| Request latency | request start to last streamed content token |
| ITL | `(request_latency - TTFT) / (output_tokens - 1)` |
| tok/s/user | `1 / ITL` |
| Prefill tok/s | `prompt_tokens / TTFT` |

Prometheus is now treated as validation for prefill and as scheduler/KV state
for Sustained Decode. It is not required for `--request-count` Burst / E2E-only
mode, and it is not the default Sustained Decode headline source.

Important interpretation rule: Sustained Decode is the kernel/scheduler tuning
signal; Burst / E2E is the client-facing finite-batch signal. If a config
improves one but hurts the other, both numbers are real but they measure
different parts of the serving path.

## Kimi vLLM Test Environment

Server:

```text
Image: voipmonitor/vllm:kimi-k26-mtp-upstream-stack-pcie-env-test-20260424-parallel-metrics-20260426
Model: moonshotai/Kimi-K2.6
Served name: Kimi-K2.6
Port: 8000
DCP: 1
Speculative: lightseekorg/kimi-k2.5-eagle3-mla, eagle3, 3 tokens, TRITON_MLA, fp8 draft KV
KV cache: 343,664 tokens
NCCL: 2.29.7
PCIE allreduce: enabled
```

## Kimi Burst / E2E-Only Comparison

The following compares `--request-count` Burst / E2E-only mode with AIPerf
using the same prompt text, same server-reported input length (`75` tokens),
same requested output length (`64` tokens), same `ignore_eos=true`, and the same
Kimi vLLM server on port `8000`.

For this parity path, `llm_decode_bench.py` requests only final OpenAI usage
chunks, matching AIPerf's finite request-burst payload shape. Continuous usage
chunks remain enabled for duration-based Sustained Decode only, where the tool
must measure inside an open time window.

| Tool | C | measured requests | prompt tok | output tok/request | aggregate tok/s | diff vs AIPerf | TTFT p50 ms | TTST p50 ms | latency p50 ms | ITL p50 ms | tok/s/user p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llm-decode-bench v0.4.0 | 1 | 12 | 75 | 64 | 169.1 | -0.3% | 58.7 | 17.8 | 369.9 | 4.95 | 202.1 |
| AIPerf 0.7.0 | 1 | 12 | 75 | 64 | 169.6 | baseline | 58.1 | 17.7 | 368.8 | 4.93 | 203.0 |
| llm-decode-bench v0.4.0 | 4 | 24 | 75 | 64 | 358.0 | -3.6% | 115.8 | 32.5 | 687.5 | 9.10 | 109.9 |
| AIPerf 0.7.0 | 4 | 24 | 75 | 64 | 371.2 | baseline | 103.7 | 31.5 | 674.5 | 8.92 | 112.1 |

This is close enough for regression use. C=1 is effectively identical. C=4 is
within a few percent; the remaining gap is consistent with small client
scheduling/wall-clock differences in finite request bursts.

## Sustained Decode Sanity Check

Sustained Decode remains the default. A short Kimi duration smoke test used:

```bash
python3 llm_decode_bench.py \
  --port 8000 \
  --model Kimi-K2.6 \
  --skip-prefill \
  --contexts 0 \
  --concurrency 1 \
  --duration 2 \
  --max-tokens 16
```

Result:

| layer | aggregate tok/s | server output tokens | client streamed chunks | completed requests |
|---|---:|---:|---:|---:|
| Sustained Decode | 116.3 | 235 | 78 | 55 |

This is why Sustained Decode should not use client streamed chunks as the
primary aggregate source: speculative decoding can deliver multiple tokens per
stream chunk, so chunk count undercounts real output tokens.

## v0.4.24 All-Layers Smoke Test

This smoke test verifies the new report layout and JSON naming. It intentionally
uses one short cell, not a performance benchmark:

```bash
python3 llm_decode_bench.py \
  --port 8000 \
  --model Kimi-K2.6 \
  --skip-prefill \
  --contexts 0 \
  --concurrency 1 \
  --duration 1 \
  --max-tokens 16 \
  --run-burst \
  --burst-request-count 3 \
  --burst-warmup-request-count 1
```

Result:

| Layer | context | C | aggregate tok/s | requests |
|---|---:|---:|---:|---:|
| Sustained Decode | 0 | 1 | 115.5 | duration window |
| Burst / E2E Decode | 0 | 1 | 94.7 | 3/3 completed |

The JSON export has `metadata.primary_decode_layer="sustained_decode"`,
one `results` entry, one `burst_results` entry, `aggregate_source` per result,
and a `methodology` object that spells out the formula for each layer.

## Prefill Parity

Prefill headline throughput is now client-side and AIPerf-style:

| Model/engine | Tool | context | prompt tok | TTFT p50/median s | client prefill tok/s | server validation |
|---|---|---:|---:|---:|---:|---:|
| GLM-5/SGLang | llm-decode-bench v0.4.21 | 8k | 8,196 | 3.114 | 2,632 | 2,663 tok/s |
| GLM-5/SGLang | AIPerf 0.7.0 | 8k | 8,197 | 3.073 | 2,668 | n/a |
| Kimi-K2.6/vLLM | llm-decode-bench v0.4.22 | 8k | 8,189 | 1.111 | 7,375 | 7,645 tok/s |
| Kimi-K2.6/vLLM | AIPerf 0.7.0 | 8k | 8,200 | 1.161 | 7,064 | n/a |

## Current Limit

The request-count formula and the tested prompt payload now align with AIPerf
for the simple single-turn case above. Remaining differences can still appear
from client scheduling details, connection pooling, and exact request issue
timing, especially at higher concurrency.

The next parity step, if exact replay is needed for arbitrary workloads, is to
add one of these:

- `--dataset-jsonl` compatible with AIPerf single-turn JSONL.
- `--export-prompts` and `--import-prompts` so both tools can replay the exact
  same messages across all context lengths.

## How To Read A Final Report

Use the sections in this order:

1. Prefill: compare long-context ingest speed only. The headline is
   `prompt_tokens / TTFT`. Optional server Prometheus counters are validation,
   not a replacement for the client-visible headline.
2. Sustained Decode: compare engine throughput once the requested concurrency is
   admitted and warmed up. This is the right table for kernel, NCCL, DCP, MTP,
   scheduler, and KV-cache regression work.
3. Burst / E2E Decode: compare finite client-facing request bursts. This is the
   right table when matching AIPerf or answering "what throughput does a batch of
   N submitted requests observe?"

The report intentionally keeps Sustained Decode and Burst / E2E Decode separate.
Sustained Decode restarts streams during the measured window to keep the engine
saturated. Burst / E2E sends a finite measured request set and waits for it to
finish. Both are useful, but they are not interchangeable.

## Artifacts

| Artifact | Path |
|---|---|
| Kimi Burst / E2E-only C1 v0.4.0 | `/tmp/llmbench_parity_c1_final.json` |
| Kimi Burst / E2E-only C4 v0.4.0 | `/tmp/llmbench_parity_c4_final.json` |
| Kimi Burst / E2E-only smoke | `/tmp/llmbench_request_count_smoke.json` |
| Kimi Sustained Decode smoke | `/tmp/llmbench_duration_smoke.json` |
| Kimi prefill v0.4.22 | `/tmp/llmbench_kimi_prefill_v0422.json` |
| Kimi all-layers smoke v0.4.24 | `/tmp/llmbench_all_layers_smoke.json` |
| AIPerf Kimi C1 same-prompt parity | `/tmp/aiperf_llmbench_parity_c1/profile_export_aiperf.json` |
| AIPerf Kimi C4 same-prompt parity | `/tmp/aiperf_llmbench_parity_c4/profile_export_aiperf.json` |
| AIPerf Kimi smoke | `/tmp/aiperf_kimi_request_count_smoke/profile_export_aiperf.json` |
