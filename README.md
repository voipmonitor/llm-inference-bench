# llm-inference-bench

LLM inference decode throughput benchmark with a Rich TUI dashboard.

Measures token generation speed across a matrix of **concurrency levels** and **context lengths**, giving you a full picture of how your serving engine scales under load.

Supports **SGLang** and **vLLM** engines (auto-detected). Works with any OpenAI-compatible API (OpenRouter, Together AI, etc.).

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)

![screenshot](screenshot.png)

## Features

- **Throughput matrix** — benchmarks every combination of concurrency (1, 2, 4, 8, ...) and context length (0K, 16K, 32K, 64K, 128K)
- **Server-side metrics** — scrapes Prometheus `/metrics` endpoint for accurate `gen_throughput` (tok/s) reported by the engine, with client-side fallback
- **Live TUI dashboard** — real-time progress, per-cell results, and aggregate stats via [Rich](https://github.com/Textualize/rich)
- **Prefill measurement** — duration-based TTFT measurement with actual token counts from server
- **Queue detection** — shows `(X/Y)` when server can't run all requested concurrent requests (KV cache full)
- **Dynamic warmup** — waits for all streams to start generating before measurement begins
- **JSON output** — structured results saved to `benchmark_results.json` for further analysis
- **Smart test skipping** — reads KV cache budget from the server, automatically skips over-capacity cells
- **Engine auto-detection** — automatically detects SGLang vs vLLM and adapts metric scraping
- **Auto-update** — checks GitHub for new versions on startup, offers one-click upgrade

## Installation

```bash
pip install httpx rich
```

## Usage

```bash
# Default: localhost:5000, tests concurrency 1-128, contexts 0K-128K
python3 llm_decode_bench.py

# Custom port and parameters
python3 llm_decode_bench.py --port 5199 --concurrency 1,2,4 --contexts 0,16384

# Custom max tokens and test duration
python3 llm_decode_bench.py --port 5001 --max-tokens 4096 --duration 60

# Remote API with authentication (OpenRouter, Together AI, etc.)
python3 llm_decode_bench.py --host https://openrouter.ai --api-key sk-or-... --model meta-llama/llama-3-70b

# Skip prefill phase for quick decode-only testing
python3 llm_decode_bench.py --skip-prefill --concurrency 1,2,4 --contexts 0

# Manual KV cache budget (for vLLM where auto-detection is unreliable)
python3 llm_decode_bench.py --port 5199 --kv-budget 692736
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--host` | `localhost` | Server hostname or full URL (e.g. `https://openrouter.ai`) |
| `--port` | `5000` | Server port (ignored when `--host` is a URL) |
| `--api-key` | | API key sent as `Authorization: Bearer` header |
| `--model` | `Qwen3.5` | Model name for API requests (auto-detected from server) |
| `--concurrency` | `1,2,4,8,16,32,64,128` | Comma-separated concurrency levels |
| `--contexts` | `0,16384,32768,65536,131072` | Comma-separated context lengths (tokens) |
| `--max-tokens` | `8192` | Max tokens to generate per request |
| `--duration` | `30` | Duration per decode test cell (seconds) |
| `--output` | `benchmark_results.json` | Output file path |
| `--kv-budget` | `0` | KV cache budget in tokens (0 = auto-detect) |
| `--skip-prefill` | | Skip prefill benchmark phase |

## Output

Results are saved as JSON with metadata and per-cell throughput data:

```json
{
  "metadata": {
    "version": "0.2.0",
    "engine": "vllm",
    "model": "Qwen3_5-397B-A17B-NVFP4",
    "timestamp": "2026-03-13T00:30:53",
    "concurrency_levels": [1, 2, 4, 8, 16, 32, 64, 128],
    "context_lengths": [0, 16384, 32768, 65536, 131072]
  },
  "prefill": { ... },
  "results": [ ... ],
  "summary_table": { ... }
}
```

## Additional tools

### `llm_cjk_watchdog.py` — CJK character leak detector

A standalone streaming watchdog that runs chat completions against any OpenAI-compatible endpoint and watches for unexpected Chinese / CJK Han ideographs in the output. Useful for catching model drift, KV-cache corruption, quantization damage, or other failure modes where an English task starts emitting Chinese tokens.

```bash
# single shot against local SGLang/vLLM on :5000
python3 llm_cjk_watchdog.py

# loop until the model leaks a Chinese character
python3 llm_cjk_watchdog.py --loop

# remote OpenAI-compatible endpoint
python3 llm_cjk_watchdog.py --host https://api.together.xyz \
    --api-key $TOGETHER_API_KEY --model meta-llama/llama-3-70b

# simulate a 40k-token input context
python3 llm_cjk_watchdog.py --context-tokens 40000 --max-tokens 2000
```

Features:

- **Loop mode** — runs indefinitely, aborts the stream the moment a CJK character appears
- **Two-row live overlay** pinned to the bottom of the terminal: row 1 shows the current iteration's live tok/s, tokens, elapsed time, and CJK counter; row 2 shows last-iteration and cumulative stats so they never scroll away
- **Precise tok/s** — uses `stream_options.continuous_usage_stats` so the live readout is the exact `completion_tokens` reported by the server, not an estimate from chunk counts
- **Padding context** — optional synthetic input of configurable token size to reproduce long-context failure modes
- **Exit code 2** when CJK characters are detected (scripting-friendly)

Requires only `requests`. See `python3 llm_cjk_watchdog.py --help` for the full CLI.

## License

MIT
