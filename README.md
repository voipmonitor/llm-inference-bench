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

## License

MIT
