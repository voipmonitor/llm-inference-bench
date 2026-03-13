# llm-inference-bench

LLM inference decode throughput benchmark with a Rich TUI dashboard.

Measures token generation speed across a matrix of **concurrency levels** and **context lengths**, giving you a full picture of how your serving engine scales under load.

Supports **SGLang** and **vLLM** engines (auto-detected).

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)

## Features

- **Throughput matrix** — benchmarks every combination of concurrency (1, 2, 4, 8, ...) and context length (0K, 16K, 32K, 64K, 128K)
- **Server-side metrics** — scrapes Prometheus `/metrics` endpoint for accurate `gen_throughput` (tok/s) reported by the engine
- **Live TUI dashboard** — real-time progress, per-cell results, and aggregate stats via [Rich](https://github.com/Textualize/rich)
- **Prefill measurement** — separate TTFT measurement for large context prefill
- **JSON output** — structured results saved to `benchmark_results.json` for further analysis
- **Engine auto-detection** — automatically detects SGLang vs vLLM and adapts metric scraping

## Installation

```bash
pip install httpx rich
```

## Usage

```bash
# Default: localhost:30000, tests concurrency 1-128, contexts 0K-128K
python3 llm_decode_bench.py

# Custom port and parameters
python3 llm_decode_bench.py --port 5199 --concurrency 1,2,4 --contexts 0,16384

# Custom max tokens and test duration
python3 llm_decode_bench.py --port 5001 --max-tokens 4096 --duration 60
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--host` | `localhost` | Server hostname |
| `--port` | `30000` | Server port |
| `--concurrency` | `1,2,4,8,16,32,64,128` | Comma-separated concurrency levels |
| `--contexts` | `0,16384,32768,65536,131072` | Comma-separated context lengths (tokens) |
| `--max-tokens` | `8192` | Max tokens to generate per request |
| `--duration` | `30` | Duration per test cell (seconds) |
| `--output` | `benchmark_results.json` | Output file path |

## Output

Results are saved as JSON with metadata and per-cell throughput data:

```json
{
  "metadata": {
    "engine": "vllm",
    "model": "Qwen3_5-397B-A17B-NVFP4",
    "timestamp": "2026-03-13T00:30:53",
    "concurrency_levels": [1, 2, 4, 8, 16, 32, 64, 128],
    "context_lengths": [0, 16384, 32768, 65536, 131072]
  },
  "results": { ... }
}
```

## License

MIT
