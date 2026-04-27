# Changelog

## 0.4.3 - 2026-04-27

### Reverse proxies

- Fixed OpenAI-compatible reverse proxies that forward `/v1/*` but return 502 or non-Prometheus bodies for `/version`, `/get_server_info`, and `/metrics`.
- Such endpoints are now detected as `openai_proxy`, Prometheus diagnostics are disabled, and sustained decode warmup uses client stream activity instead of waiting 60 seconds for nonexistent scheduler metrics.

### Live prefill progress

- Added live prefill ETA/progress text for long-prefill models.
- The ETA uses the nearest completed prefill sample, including the observed tokenizer-token ratio, so long contexts show an approximate remaining time instead of only a static "prefill" status.
- When no prefill baseline exists yet, the dashboard explicitly says it is waiting for the first completed prefill sample.

## 0.4.2 - 2026-04-26

### Graceful Quit

- Fixed early `q` / Ctrl-C final reports losing already measured prefill rows.
- Partial prefill results are now snapshotted whenever an integrated scout, scout-only prefill, or standalone prefill row completes.
- Primary Summary now includes `Prefill tok/s` on interrupted runs as soon as any prefill row has been measured.

## 0.4.1 - 2026-04-26

### Metrics optionality

- SGLang without `--enable-metrics` is no longer fatal.
- Missing `/metrics` now produces a visible warning and the benchmark continues using OpenAI stream metrics for headline throughput.
- Scheduler/effective-concurrency, KV auto-detection from Prometheus, and Prometheus validation are marked unavailable when metrics are disabled.
- Duration warmup falls back to client stream activity when scheduler metrics are unavailable, avoiding the old 60 second wait for `running_reqs`.
- Request-count Burst / E2E mode also skips repeated `/metrics` scrapes when metrics are unavailable.

### Live dashboard

- Improved the mid-width hardware panel: it now gets more horizontal space and uses a real GPU table before falling back to the ultra-compact layout.

## 0.4.0 - 2026-04-26

### Measurement methodology

- Added three clearly separated benchmark layers: integrated prefill, sustained decode, and optional Burst / E2E decode.
- Kept duration-based Sustained Decode as the default tuning/regression signal. The benchmark waits for the server to admit the requested concurrency and pass warmup before measuring.
- Added request-count Burst / E2E-only mode with `--request-count` and `--warmup-request-count`, matching the finite request-burst style used by tools such as AIPerf.
- Added optional post-sustained Burst / E2E matrix via `--run-burst`, `--burst-request-count`, `--burst-warmup-request-count`, and `--burst-requests-per-concurrency`.
- Switched aggregate decode throughput to OpenAI stream usage where available, using continuous `completion_tokens` during the measured window. Prometheus generation counters remain as validation and scheduler telemetry.
- Changed request-count Burst / E2E payloads to request final OpenAI usage only. `continuous_usage_stats` is reserved for duration-based Sustained Decode, where it is required to measure inside an open time window.
- Reworked client latency metrics to follow OpenAI streaming semantics: TTFT starts at request submission, request latency ends at the last content token, and ITL uses the interval between first and last received content tokens.
- Sustained decode now computes observed ITL and per-user decode throughput from partial streams stopped at the measurement boundary, without using cancel or HTTP close time as a synthetic last token.
- Full request latency remains completed-stream-only.

### Prefill

- Integrated default prefill measurement into decode scout requests, avoiding the old extra repeated prefill phase for normal runs.
- Added scout-only extra prefill contexts through `--prefill-contexts`; default prefill contexts include `8k,64k,128k` plus non-zero decode contexts.
- Added `--standalone-prefill` for the previous cold-prefill profile when debugging ingest behavior.
- Added `--prefill-metric` with client headline measurement and optional Prometheus validation.
- Fixed the old misleading baseline-subtraction prefill approach; headline prefill is now client `prompt_tokens / TTFT`.

### Live dashboard

- Reworked the Rich TUI into adaptive wide, medium, and narrow layouts.
- Added a live hardware panel with GPU SM utilization, memory-controller utilization, VRAM usage, power, temperature, clocks, PCIe rx/tx, and CPU utilization/frequency.
- Added an event log for warmup, readiness, skips, and cell completion.
- Made the aggregate decode panel compact instead of stretching across unused screen width.
- Added inline aggregate-cell latency details when horizontal space allows: `tok/s + TTFT/ITL`. Narrower layouts fall back to stacked or compact cells.
- Added a decode speed trace with fixed deviation scaling so small variance does not look like large jitter.
- Improved terminal ergonomics: `q` now behaves like Ctrl-C and prints partial results instead of hard-exiting.
- Improved narrow-terminal rendering so hardware and decode panels avoid Rich ellipsis in important numeric columns.

### Final report

- Reordered final output so primary prefill and aggregate decode summaries are repeated at the end.
- Replaced the misleading global mixed client distribution table with per-cell client matrices.
- Added per-cell request latency matrices while keeping sample counts and full request-level distributions in JSON.
- The aggregate decode matrix can include compact per-cell `TTFT/ITL` detail; per-request throughput and request latency are shown separately.
- Added explicit notes when Burst / E2E was not run.

### Scheduler, KV, and diagnostics

- Added effective-concurrency tracking from scheduler metrics where available.
- Marked cells that cannot fit the configured KV budget and kept exact deficit information in JSON.
- Added `--dcp-size` support for deriving DCP-adjusted KV budget when server-side introspection is not available.
- Added startup diagnostics to JSON: benchmark args, relevant `NCCL_`/`VLLM_`/`SGLANG_`/`CUDA_`/`OMP_` environment variables, `uname`, GPU query output, and `nvidia-smi topo -m`.
- Added hardware summaries per measured cell when hardware sampling is enabled.

### JSON and documentation

- Expanded JSON output with decode mode, primary decode layer, burst settings, prefill mode, startup diagnostics, event log, hardware summaries, request samples, and methodology metadata.
- Updated README with the new benchmark layers, options, methodology, client latency semantics, hardware panel, and JSON structure.
- Added this CHANGELOG for versioned methodology changes.

### Compatibility and validation

- Kept support for vLLM and SGLang auto-detection.
- Kept Prometheus optional: it is used for validation, scheduler/effective-concurrency state, and server-side diagnostics, while OpenAI streaming remains the primary portable data source.
- Added parity-oriented request-count mode so comparable workloads can be checked against AIPerf-style finite request-burst measurements.
