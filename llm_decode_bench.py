#!/usr/bin/env python3
"""
LLM Inference Benchmark with Rich TUI Dashboard.

Measures decode throughput across a matrix of concurrency levels and context lengths.
Auto-detects SGLang or vLLM engine and adapts metrics accordingly.

Usage:
    python3 llm_decode_bench.py
    python3 llm_decode_bench.py --port 5199 --concurrency 1,2,4 --contexts 0,16384
    python3 llm_decode_bench.py --port 5199 --kv-budget 692736
    python3 llm_decode_bench.py --port 5001 --max-tokens 4096
    python3 llm_decode_bench.py --host https://openrouter.ai --api-key sk-or-... --model meta-llama/llama-3-70b
    python3 llm_decode_bench.py --skip-prefill --concurrency 1,2,4 --contexts 0
"""

import argparse
import asyncio
import json
import os
import random
import re
import select
import string
import subprocess
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass, field, asdict
from datetime import datetime
from statistics import mean, median
from typing import Optional

import httpx
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERSION = "0.2.0"

CHARS_PER_TOKEN = 4

PADDING_SENTENCES = [
    "The history of European architecture spans thousands of years and encompasses a wide variety of styles and movements.",
    "From the ancient Greek temples to the Gothic cathedrals of the Middle Ages, each era has left its distinctive mark on the built environment.",
    "The Renaissance brought a renewed interest in classical forms, while the Baroque period introduced dramatic ornamentation and grandeur.",
    "In the modern era, architects have experimented with new materials such as steel, glass, and reinforced concrete.",
    "The development of skyscrapers in the late 19th century transformed urban landscapes around the world.",
    "Sustainable architecture has become increasingly important as societies grapple with climate change and resource depletion.",
    "The principles of good design include functionality, durability, and aesthetic appeal.",
    "Urban planning plays a crucial role in shaping how cities develop and how their inhabitants experience daily life.",
    "Public spaces such as parks, plazas, and waterfronts contribute significantly to the quality of urban living.",
    "The integration of technology into building design has opened up new possibilities for energy efficiency and comfort.",
    "Historical preservation efforts seek to maintain the cultural heritage embodied in older structures.",
    "The relationship between architecture and nature has been explored by many influential designers throughout history.",
    "Building codes and regulations ensure that structures meet minimum standards for safety and accessibility.",
    "The choice of materials in construction affects not only the appearance of a building but also its environmental impact.",
    "Innovative structural engineering techniques have made it possible to create buildings of unprecedented scale and complexity.",
    "The study of vernacular architecture reveals how different cultures have adapted their building practices to local conditions.",
    "Interior design complements architecture by addressing the arrangement and decoration of interior spaces.",
    "Landscape architecture deals with the design of outdoor areas, landmarks, and structures to achieve environmental or aesthetic outcomes.",
    "The concept of smart cities integrates information technology with urban infrastructure to improve efficiency and quality of life.",
    "Affordable housing remains one of the most pressing challenges facing urban planners and policymakers worldwide.",
]

GENERATION_PROMPT = (
    "Write an extremely detailed, comprehensive encyclopedia article about the complete "
    "history of mathematics from ancient Mesopotamia to 2025. Cover every civilization, "
    "every major mathematician, every theorem, proof, and breakthrough. Include detailed "
    "biographical information, historical context, and mathematical explanations. "
    "Do not summarize - provide maximum detail on every topic."
)

METRIC_RE = re.compile(r'^((?:sglang|vllm):\w+)(?:\{([^}]*)\})?\s+([\d.eE+-]+)')

# Engine types
ENGINE_SGLANG = "sglang"
ENGINE_VLLM = "vllm"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StreamResult:
    ttft: float = 0.0
    total_tokens: int = 0
    total_time: float = 0.0
    tokens_per_sec: float = 0.0
    error: Optional[str] = None


@dataclass
class CellResult:
    concurrency: int = 0
    context_tokens: int = 0
    aggregate_tps: float = 0.0
    per_request_avg_tps: float = 0.0
    ttft_avg: float = 0.0
    ttft_p50: float = 0.0
    ttft_p99: float = 0.0
    total_tokens: int = 0
    wall_time: float = 0.0
    num_completed: int = 0
    num_errors: int = 0
    server_gen_throughput: float = 0.0
    server_utilization: float = 0.0
    server_spec_accept_rate: float = 0.0
    # Queue / effective concurrency tracking
    avg_running_reqs: float = 0.0
    max_running_reqs: int = 0
    avg_queue_reqs: float = 0.0
    max_queue_reqs: int = 0
    queue_fraction: float = 0.0  # fraction of samples where queue > 0


@dataclass
class TUIState:
    # Overall
    engine: str = ENGINE_SGLANG
    model_name: str = ""
    server_url: str = ""
    total_tests: int = 0
    completed_tests: int = 0
    overall_start: float = 0.0
    # Current cell
    current_concurrency: int = 0
    current_context: int = 0
    cell_start: float = 0.0
    cell_duration: float = 20.0
    cell_tokens: int = 0
    cell_live_tps: float = 0.0
    cell_running: bool = False
    cell_warmup: bool = False  # True during prefill ramp-up before measurement
    cell_measurement_start: float = 0.0  # when actual measurement begins (after warmup)
    # Server metrics
    srv_gen_throughput: float = 0.0
    srv_running_reqs: int = 0
    srv_queue_reqs: int = 0
    srv_utilization: float = 0.0
    srv_spec_accept_rate: float = 0.0
    srv_spec_accept_length: float = 0.0
    # Results
    results: dict = field(default_factory=dict)  # (ctx, conc) -> aggregate_tps
    errors: dict = field(default_factory=dict)   # (ctx, conc) -> num_errors
    queue_info: dict = field(default_factory=dict)  # (ctx, conc) -> (avg_running, avg_queue)
    concurrency_levels: list = field(default_factory=list)
    context_lengths: list = field(default_factory=list)
    # Prefill results: ctx -> {ttft, tok_per_sec}
    prefill_results: dict = field(default_factory=dict)
    prefill_contexts: list = field(default_factory=list)
    prefill_phase: bool = False
    # Server limits
    kv_cache_budget: int = 0
    max_running_requests: int = 0
    skipped_cells: int = 0
    max_tokens: int = 0
    # Timing
    cell_times: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_token_value(s: str) -> int:
    """Parse token value with optional k/K suffix: '16384', '16k', '128K' → int."""
    s = s.strip()
    if s.lower().endswith("k"):
        return int(float(s[:-1]) * 1024)
    return int(s)


def generate_padding_text(target_tokens: int) -> str:
    target_chars = target_tokens * CHARS_PER_TOKEN
    lines = []
    current_chars = 0
    idx = 0
    while current_chars < target_chars:
        sentence = PADDING_SENTENCES[idx % len(PADDING_SENTENCES)]
        lines.append(sentence)
        current_chars += len(sentence) + 1
        idx += 1
    return " ".join(lines)


def build_messages(context_tokens: int, context_text: str) -> list:
    messages = []
    if context_tokens > 0 and context_text:
        messages.append({
            "role": "user",
            "content": (
                "Below is a large reference document. Read it carefully, "
                "then answer the question that follows.\n\n"
                "--- BEGIN REFERENCE DOCUMENT ---\n"
                f"{context_text}\n"
                "--- END REFERENCE DOCUMENT ---"
            )
        })
        messages.append({
            "role": "assistant",
            "content": "I have read the entire reference document. Please ask your question."
        })
    messages.append({"role": "user", "content": GENERATION_PROMPT})
    return messages


def percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def format_context(ctx: int) -> str:
    if ctx == 0:
        return "0"
    elif ctx >= 1024:
        return f"{ctx // 1024}k"
    return str(ctx)


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


# ---------------------------------------------------------------------------
# Metrics scraping
# ---------------------------------------------------------------------------

async def scrape_metrics(client: httpx.AsyncClient, base_url: str) -> dict:
    metrics = {}
    try:
        resp = await client.get(f"{base_url}/metrics", timeout=5.0)
        for line in resp.text.splitlines():
            if line.startswith("#"):
                continue
            m = METRIC_RE.match(line)
            if m:
                name, labels, value = m.group(1), m.group(2) or "", float(m.group(3))
                # Only take tp_rank=0 metrics to avoid duplicates
                if "tp_rank=" in labels and 'tp_rank="0"' not in labels:
                    continue
                key = f"{name}|{labels}" if labels else name
                metrics[key] = value
    except Exception:
        pass
    return metrics


def extract_metric(metrics: dict, name: str, label_filter: str = "") -> float:
    for key, val in metrics.items():
        if key.startswith(name):
            if label_filter and label_filter not in key:
                continue
            return val
    return 0.0


def extract_label(metrics: dict, metric_name: str, label: str) -> str:
    """Extract a label value from a labeled Prometheus metric."""
    for key in metrics:
        if key.startswith(metric_name):
            m = re.search(rf'{label}="([^"]*)"', key)
            if m:
                return m.group(1)
    return ""


def metric_name(engine: str, key: str) -> str:
    """Map a logical metric key to engine-specific Prometheus metric name."""
    names = {
        ENGINE_SGLANG: {
            "gen_throughput": "sglang:gen_throughput",
            "running_reqs": "sglang:num_running_reqs",
            "queue_reqs": "sglang:num_queue_reqs",
            "utilization": "sglang:utilization",
            "spec_accept_rate": "sglang:spec_accept_rate",
            "spec_accept_length": "sglang:spec_accept_length",
            "gen_tokens_total": "sglang:generation_tokens_total",
        },
        ENGINE_VLLM: {
            "gen_throughput": "vllm:avg_generation_throughput_toks_per_s",
            "running_reqs": "vllm:num_requests_running",
            "queue_reqs": "vllm:num_requests_waiting",
            "utilization": "vllm:kv_cache_usage_perc",
            "spec_accept_rate": "",
            "spec_accept_length": "",
            "gen_tokens_total": "vllm:generation_tokens_total",
        },
    }
    return names.get(engine, names[ENGINE_SGLANG]).get(key, "")


# ---------------------------------------------------------------------------
# Streaming request
# ---------------------------------------------------------------------------

async def stream_one_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    index: int,
    cancel_event: asyncio.Event,
    shared_token_count: list,
    shared_active_streams: list = None,
) -> StreamResult:
    """Stream requests in a loop until cancel_event. When a request finishes
    (hits max_tokens or EOS), immediately start a new one to keep concurrency
    saturated. Returns aggregate stats across all iterations."""
    result = StreamResult()
    t_start = time.monotonic()
    t_first = None
    total_chunks = 0
    total_usage_tokens = 0
    iterations = 0
    active_signaled = False

    while not cancel_event.is_set():
        usage_tokens = None
        try:
            async with client.stream("POST", url, json=payload, timeout=httpx.Timeout(600.0, connect=30.0)) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    result.error = f"HTTP {resp.status_code}: {body.decode()[:200]}"
                    break

                async for line in resp.aiter_lines():
                    if cancel_event.is_set():
                        break

                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Check for usage in final chunk (stream_options.include_usage)
                    usage = data.get("usage")
                    if usage and "completion_tokens" in usage:
                        usage_tokens = usage["completion_tokens"]

                    if "choices" not in data or len(data["choices"]) == 0:
                        continue

                    delta = data["choices"][0].get("delta", {})
                    text = ""

                    reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                    if reasoning:
                        text += reasoning

                    content = delta.get("content")
                    if content:
                        text += content

                    if text:
                        if t_first is None:
                            t_first = time.monotonic()
                        if not active_signaled and shared_active_streams is not None:
                            shared_active_streams[0] += 1
                            active_signaled = True
                        total_chunks += 1
                        shared_token_count[0] += 1

        except httpx.ReadTimeout:
            result.error = "ReadTimeout"
            break
        except httpx.ConnectError as e:
            result.error = f"ConnectError: {e}"
            break
        except httpx.RemoteProtocolError as e:
            result.error = f"ProtocolError: {e}"
            break
        except asyncio.CancelledError:
            break
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            break

        if usage_tokens is not None:
            total_usage_tokens += usage_tokens
        iterations += 1

    t_end = time.monotonic()
    result.total_tokens = total_usage_tokens if total_usage_tokens > 0 else total_chunks
    result.total_time = t_end - t_start
    if t_first is not None:
        result.ttft = t_first - t_start
    if result.total_tokens > 0 and result.total_time > 0:
        result.tokens_per_sec = result.total_tokens / result.total_time

    return result


# ---------------------------------------------------------------------------
# Run one cell
# ---------------------------------------------------------------------------

async def run_one_cell(
    client: httpx.AsyncClient,
    base_url: str,
    concurrency: int,
    context_tokens: int,
    context_text: str,
    duration: float,
    max_tokens: int,
    model: str,
    state: TUIState,
    live: Live,
    engine: str = ENGINE_SGLANG,
    auth_headers: dict = None,
) -> CellResult:
    messages = build_messages(context_tokens, context_text)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "stream_options": {"include_usage": True},
    }

    url = f"{base_url}/v1/chat/completions"
    cancel_event = asyncio.Event()
    shared_token_count = [0]
    shared_active_streams = [0]  # how many streams have received first token

    # Fresh client per cell — avoids stale keepalive connections from previous cells
    cell_limits = httpx.Limits(
        max_connections=concurrency + 10,
        max_keepalive_connections=concurrency + 5,
    )
    cell_client = httpx.AsyncClient(limits=cell_limits, headers=auth_headers or {})

    # Update TUI state
    state.current_concurrency = concurrency
    state.current_context = context_tokens
    state.cell_start = time.monotonic()
    state.cell_duration = duration
    state.cell_tokens = 0
    state.cell_live_tps = 0.0
    state.cell_running = True
    state.cell_warmup = True

    # Scout request: ensure prefix cache is warm before launching full concurrency.
    # Send one request with max_tokens=1 to populate/refresh prefix cache,
    # then all C requests will get cache hits instead of competing for prefill.
    if context_tokens > 0:
        scout_payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": 1,
        }
        try:
            async with client.stream("POST", url, json=scout_payload,
                                     timeout=httpx.Timeout(600.0, connect=30.0)) as resp:
                async for line in resp.aiter_lines():
                    if line and line.startswith("data: [DONE]"):
                        break
        except Exception:
            pass
        live.update(build_display(state))

    # Launch all streams on fresh client (no stale keepalive connections)
    tasks = [
        asyncio.create_task(
            stream_one_request(cell_client, url, payload, i, cancel_event,
                               shared_token_count, shared_active_streams)
        )
        for i in range(concurrency)
    ]

    # Monitor loop — collect server gen_throughput samples for accurate measurement
    metrics_interval = 1.0
    min_warmup_seconds = 2.0   # minimum warmup (CUDA graph etc.)
    max_warmup_seconds = 60.0  # give up waiting for queue to drain
    last_metrics_time = 0.0
    gen_throughput_samples = []
    # For vLLM: compute throughput rate from generation_tokens counter
    prev_gen_tokens = None
    prev_gen_time = None
    # Queue tracking: collect running/queue samples after warmup
    running_reqs_samples = []
    queue_reqs_samples = []
    # Dynamic warmup: wait until all requests are in decode (queue == 0)
    warmup_done = False
    warmup_stable_since = None  # time when queue first hit 0 after min_warmup
    measurement_start = None    # reset timer after warmup for full duration measurement
    measurement_tokens_start = 0  # token count at measurement start (for client-side fallback)

    while True:
        await asyncio.sleep(0.5)
        now = time.monotonic()
        elapsed = now - state.cell_start

        # Update token counts from client-side estimate (for TUI only)
        state.cell_tokens = shared_token_count[0]
        state._active_streams = shared_active_streams[0]

        # Scrape server metrics periodically
        if now - last_metrics_time > metrics_interval:
            metrics = await scrape_metrics(client, base_url)

            # Throughput: SGLang has a gauge, vLLM needs rate from counter
            if engine == ENGINE_SGLANG:
                state.srv_gen_throughput = extract_metric(metrics, metric_name(engine, "gen_throughput"))
            else:
                # Try vLLM v0 gauge first
                tp = extract_metric(metrics, metric_name(engine, "gen_throughput"))
                if tp > 0:
                    state.srv_gen_throughput = tp
                else:
                    # vLLM v1: compute rate from generation_tokens counter
                    gen_total = extract_metric(metrics, metric_name(engine, "gen_tokens_total"))
                    if gen_total > 0 and prev_gen_tokens is not None:
                        dt = now - prev_gen_time
                        if dt > 0.1:
                            state.srv_gen_throughput = (gen_total - prev_gen_tokens) / dt
                    prev_gen_tokens = gen_total
                    prev_gen_time = now

            state.srv_running_reqs = int(extract_metric(metrics, metric_name(engine, "running_reqs")))
            state.srv_queue_reqs = int(extract_metric(metrics, metric_name(engine, "queue_reqs")))
            state.srv_utilization = extract_metric(metrics, metric_name(engine, "utilization"))
            state.srv_spec_accept_rate = extract_metric(metrics, metric_name(engine, "spec_accept_rate"))
            state.srv_spec_accept_length = extract_metric(metrics, metric_name(engine, "spec_accept_length"))

            # Dynamic warmup: wait for min_warmup AND queue==0 AND tokens flowing
            # All three conditions ensure:
            #   - CUDA graphs / JIT compiled (min_warmup)
            #   - all requests past prefill (queue==0)
            #   - tokens actually being generated (server OR client-side detection)
            if not warmup_done:
                if elapsed >= min_warmup_seconds:
                    # All streams must have received their first token before measurement.
                    # This prevents measuring before slow-to-prefill requests start decoding.
                    all_streams_active = shared_active_streams[0] >= concurrency
                    generating = state.srv_gen_throughput > 0 or shared_token_count[0] > 0
                    server_ready = state.srv_queue_reqs == 0 and generating and all_streams_active
                    if server_ready:
                        if warmup_stable_since is None:
                            warmup_stable_since = now
                        # Require 1s of stable conditions to avoid transient dips
                        elif now - warmup_stable_since >= 1.0:
                            warmup_done = True
                            state.cell_warmup = False
                            measurement_start = now
                            state.cell_measurement_start = now
                            measurement_tokens_start = shared_token_count[0]
                    else:
                        warmup_stable_since = None
                    # Give up after max_warmup — queue never drained (real capacity issue)
                    if elapsed >= max_warmup_seconds:
                        warmup_done = True
                        state.cell_warmup = False
                        measurement_start = now
                        state.cell_measurement_start = now
                        measurement_tokens_start = shared_token_count[0]

            # Collect samples only after warmup is done
            if warmup_done:
                if state.srv_gen_throughput > 0:
                    gen_throughput_samples.append(state.srv_gen_throughput)
                running_reqs_samples.append(state.srv_running_reqs)
                queue_reqs_samples.append(state.srv_queue_reqs)
            last_metrics_time = now

        # Use server gen_throughput for live display; fall back to client-side estimate
        if state.srv_gen_throughput > 0:
            state.cell_live_tps = state.srv_gen_throughput
        elif measurement_start:
            client_elapsed = now - measurement_start
            measurement_tokens = shared_token_count[0] - measurement_tokens_start
            if client_elapsed > 0.5 and measurement_tokens > 0:
                state.cell_live_tps = measurement_tokens / client_elapsed

        # Update TUI
        live.update(build_display(state))

        # Check duration (measured from after warmup completes)
        measure_elapsed = (now - measurement_start) if measurement_start else 0
        if measurement_start and measure_elapsed >= duration:
            cancel_event.set()
            break

        # Check skip key
        if _skip_event.is_set():
            _skip_event.clear()
            cancel_event.set()
            return CellResult(concurrency=concurrency, context_tokens=context_tokens, aggregate_tps=-2)

        # Check if all tasks already done
        if all(t.done() for t in tasks):
            break

    # Wait for tasks to finish (grace period)
    done, pending = await asyncio.wait(tasks, timeout=30.0)
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.wait(pending, timeout=5.0)

    # Collect results
    wall_time = time.monotonic() - state.cell_start
    stream_results = []
    for t in tasks:
        try:
            stream_results.append(t.result())
        except (asyncio.CancelledError, Exception):
            stream_results.append(StreamResult(error="cancelled", total_time=wall_time))

    # Final metrics scrape
    metrics = await scrape_metrics(client, base_url)
    if engine == ENGINE_SGLANG:
        final_gen_throughput = extract_metric(metrics, metric_name(engine, "gen_throughput"))
    else:
        final_gen_throughput = extract_metric(metrics, metric_name(engine, "gen_throughput"))
        if final_gen_throughput == 0 and prev_gen_tokens is not None:
            gen_total = extract_metric(metrics, metric_name(engine, "gen_tokens_total"))
            dt = time.monotonic() - prev_gen_time if prev_gen_time else 0
            if gen_total > 0 and dt > 0.1:
                final_gen_throughput = (gen_total - prev_gen_tokens) / dt
    if final_gen_throughput > 0:
        gen_throughput_samples.append(final_gen_throughput)

    # Use server-side gen_throughput as the primary metric (median for robustness)
    avg_gen_throughput = median(gen_throughput_samples) if gen_throughput_samples else 0.0

    # Client-side stats
    successful = [r for r in stream_results if r.error is None]
    total_tokens = sum(r.total_tokens for r in stream_results)

    # Client-side fallback: only when server metrics give 0 (vLLM V1 missing gauge
    # and counter not updating in real-time).
    # shared_token_count tracks exact token count (1 SSE chunk = 1 token).
    if avg_gen_throughput == 0:
        measure_duration = (time.monotonic() - measurement_start) if measurement_start else wall_time
        measurement_tokens = shared_token_count[0] - measurement_tokens_start
        if measure_duration > 0 and measurement_tokens > 0:
            avg_gen_throughput = measurement_tokens / measure_duration

    # Derive per-request from aggregate for consistency
    per_req_tps = avg_gen_throughput / concurrency if concurrency > 0 else 0.0
    ttfts = [r.ttft for r in successful if r.ttft > 0]

    # Queue stats
    avg_running = mean(running_reqs_samples) if running_reqs_samples else 0.0
    max_running = max(running_reqs_samples) if running_reqs_samples else 0
    avg_queue = mean(queue_reqs_samples) if queue_reqs_samples else 0.0
    max_queue = max(queue_reqs_samples) if queue_reqs_samples else 0
    queued_count = sum(1 for q in queue_reqs_samples if q > 0)
    queue_frac = queued_count / len(queue_reqs_samples) if queue_reqs_samples else 0.0

    cell = CellResult(
        concurrency=concurrency,
        context_tokens=context_tokens,
        aggregate_tps=avg_gen_throughput,
        per_request_avg_tps=per_req_tps,
        ttft_avg=mean(ttfts) if ttfts else 0.0,
        ttft_p50=percentile(ttfts, 50) if ttfts else 0.0,
        ttft_p99=percentile(ttfts, 99) if ttfts else 0.0,
        total_tokens=total_tokens,
        wall_time=wall_time,
        num_completed=len(successful),
        num_errors=len(stream_results) - len(successful),
        server_gen_throughput=median(gen_throughput_samples) if gen_throughput_samples else 0.0,
        server_utilization=extract_metric(metrics, metric_name(engine, "utilization")),
        server_spec_accept_rate=extract_metric(metrics, metric_name(engine, "spec_accept_rate")),
        avg_running_reqs=round(avg_running, 1),
        max_running_reqs=max_running,
        avg_queue_reqs=round(avg_queue, 1),
        max_queue_reqs=max_queue,
        queue_fraction=round(queue_frac, 3),
    )

    state.cell_running = False
    state.results[(context_tokens, concurrency)] = cell.aggregate_tps
    state.errors[(context_tokens, concurrency)] = cell.num_errors
    state.queue_info[(context_tokens, concurrency)] = (cell.avg_running_reqs, cell.avg_queue_reqs)

    await cell_client.aclose()
    return cell


# ---------------------------------------------------------------------------
# TUI rendering
# ---------------------------------------------------------------------------

def build_display(state: TUIState) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="middle", size=10),
        Layout(name="results", ratio=1, minimum_size=8),
        Layout(name="footer", size=3),
    )
    layout["middle"].split_row(
        Layout(name="current_test", ratio=1),
        Layout(name="server_metrics", ratio=1),
    )

    # Header
    header_text = Text()
    engine_label = state.engine.upper() if state.engine else "Benchmark"
    header_text.append(f"{engine_label} Benchmark", style="bold cyan")
    header_text.append(f"  {state.model_name} @ {state.server_url}")
    header_text.append(f"  |  {state.total_tests} tests  |  {state.cell_duration:.0f}s each")
    if state.kv_cache_budget > 0 or state.max_running_requests > 0:
        header_text.append("  |  ", style="dim")
        if state.kv_cache_budget > 0:
            header_text.append(f"KV: {state.kv_cache_budget:,}", style="cyan")
        if state.max_running_requests > 0:
            if state.kv_cache_budget > 0:
                header_text.append("  ", style="dim")
            header_text.append(f"MaxReqs: {state.max_running_requests}", style="cyan")
        if state.skipped_cells > 0:
            header_text.append(f"  ({state.skipped_cells} skipped)", style="yellow")
    layout["header"].update(Panel(header_text, style="bold"))

    # Current test panel
    if state.cell_running:
        elapsed = time.monotonic() - state.cell_start
        if state.prefill_phase:
            cell_text = (
                f"[bold magenta]PREFILL[/bold magenta]  ctx={format_context(state.current_context)}\n"
                f"  Elapsed: {elapsed:.1f}s\n"
                f"  Populating radix cache...\n"
                f"  Test [bold]{state.completed_tests + 1}[/bold] of {state.total_tests}"
            )
        else:
            if state.cell_warmup:
                # During ramp-up: show elapsed warmup time and why we're waiting
                if state.srv_gen_throughput == 0 and state.cell_tokens == 0:
                    wait_reason = "waiting for token generation (JIT compile?)"
                elif state.srv_queue_reqs > 0:
                    wait_reason = "waiting for queue→0 (prefill ramp-up)"
                elif hasattr(state, '_active_streams') and state._active_streams < state.current_concurrency:
                    wait_reason = f"waiting for all streams ({state._active_streams}/{state.current_concurrency})"
                else:
                    wait_reason = "stabilizing..."
                cell_text = (
                    f"[bold]DECODE[/bold]  C={state.current_concurrency}, ctx={format_context(state.current_context)}"
                    f"  [magenta]RAMP-UP[/magenta]\n"
                    f"  {wait_reason} {elapsed:.0f}s\n"
                    f"  Server: [bold yellow]{state.cell_live_tps:.1f}[/bold yellow] tok/s  "
                    f"running={state.srv_running_reqs} queue={state.srv_queue_reqs}\n"
                    f"  Test [bold]{state.completed_tests + 1}[/bold] of {state.total_tests}"
                )
            else:
                # Measurement phase: progress bar from measurement_start
                measure_elapsed = (time.monotonic() - state.cell_measurement_start) if state.cell_measurement_start > 0 else elapsed
                pct = min(measure_elapsed / state.cell_duration, 1.0) if state.cell_duration > 0 else 0
                bar_width = 30
                filled = int(pct * bar_width)
                bar = "[green]" + "=" * filled + ">" + "[/green]" + " " * (bar_width - filled - 1)

                cell_text = (
                    f"[bold]DECODE[/bold]  C={state.current_concurrency}, ctx={format_context(state.current_context)}\n"
                    f"  [{bar}] {measure_elapsed:.0f}/{state.cell_duration:.0f}s\n"
                    f"  Server: [bold yellow]{state.cell_live_tps:.1f}[/bold yellow] tok/s ({state.engine} throughput)\n"
                    f"  Test [bold]{state.completed_tests + 1}[/bold] of {state.total_tests}"
                )
    else:
        cell_text = "[dim]Waiting...[/dim]"
    layout["current_test"].update(Panel(cell_text, title="Current Test", border_style="cyan"))

    # Server metrics panel
    srv_table = Table(show_header=False, box=None, padding=(0, 1))
    srv_table.add_column("Metric", style="dim")
    srv_table.add_column("Value", style="bold")
    srv_table.add_row("gen_throughput", f"{state.srv_gen_throughput:.1f} tok/s")
    srv_table.add_row("running_reqs", str(state.srv_running_reqs))
    srv_table.add_row("queue_reqs", str(state.srv_queue_reqs))
    srv_table.add_row("utilization", f"{state.srv_utilization:.2%}")
    srv_table.add_row("spec_accept_rate", f"{state.srv_spec_accept_rate:.2%}")
    srv_table.add_row("spec_accept_len", f"{state.srv_spec_accept_length:.1f}")
    layout["server_metrics"].update(Panel(srv_table, title="Server Metrics", border_style="magenta"))

    # Results table
    results_table = Table(title="Aggregate Throughput (tok/s)", border_style="green", expand=True)
    results_table.add_column("ctx \\ conc", style="bold cyan", min_width=8)
    for conc in state.concurrency_levels:
        results_table.add_column(str(conc), justify="right", min_width=10)

    # Determine color thresholds from existing results (exclude skipped=-1)
    all_values = [v for v in state.results.values() if v > 0]
    if all_values:
        p25 = percentile(all_values, 25)
        p75 = percentile(all_values, 75)
    else:
        p25, p75 = 0, 0

    for ctx in state.context_lengths:
        row = [format_context(ctx)]
        for conc in state.concurrency_levels:
            key = (ctx, conc)
            if key in state.results:
                val = state.results[key]
                if val == -2:
                    row.append("[yellow]skipped[/yellow]")
                    continue
                if val < 0:
                    needed = conc * (ctx + state.max_tokens)
                    row.append(f"[dim]N/A ({needed // 1024}k)[/dim]")
                    continue
                errs = state.errors.get(key, 0)
                if val > p75 and p75 > 0:
                    style = "bold green"
                elif val < p25 and p25 > 0:
                    style = "red"
                else:
                    style = "yellow"
                cell = f"{val:.1f}"
                if errs > 0:
                    cell += f" [red]({errs}e)[/red]"
                # Show running/requested when queuing detected
                qi = state.queue_info.get(key)
                if qi and qi[1] > 0:
                    avg_run, avg_q = qi
                    cell += f" [magenta]({avg_run:.0f}/{conc})[/magenta]"
                row.append(f"[{style}]{cell}[/{style}]")
            else:
                row.append("[dim]...[/dim]")
        results_table.add_row(*row)

    # Prefill table (shown alongside decode results)
    if state.prefill_contexts:
        prefill_table = Table(title="Prefill (C=1)", border_style="magenta", expand=True)
        prefill_table.add_column("ctx", style="bold cyan")
        prefill_table.add_column("TTFT", justify="right")
        prefill_table.add_column("tok/s", justify="right")
        prefill_table.add_column("N", justify="right")
        for ctx in state.prefill_contexts:
            if ctx in state.prefill_results:
                pr = state.prefill_results[ctx]
                if pr.get("skipped"):
                    prefill_table.add_row(format_context(ctx), "[yellow]skip[/yellow]", "", "")
                else:
                    prefill_table.add_row(
                        format_context(ctx),
                        f"{pr['ttft']:.2f}s",
                        f"[bold green]{pr['tok_per_sec']:,.0f}[/bold green]",
                        str(pr.get('samples', '')),
                    )
            else:
                prefill_table.add_row(format_context(ctx), "[dim]...[/dim]", "[dim]...[/dim]", "")

        results_layout = Layout()
        results_layout.split_row(
            Layout(Panel(prefill_table), size=38),
            Layout(Panel(results_table)),
        )
        layout["results"].update(results_layout)
    else:
        layout["results"].update(Panel(results_table))

    # Footer - overall progress
    if state.total_tests > 0:
        overall_pct = state.completed_tests / state.total_tests
        elapsed_total = time.monotonic() - state.overall_start if state.overall_start > 0 else 0
        if state.cell_times:
            avg_cell = mean(state.cell_times)
            remaining = (state.total_tests - state.completed_tests) * avg_cell
            eta_str = format_time(remaining)
        else:
            eta_str = "calculating..."

        bar_width = 50
        filled = int(overall_pct * bar_width)
        bar = "[green]" + "=" * filled + ">" + "[/green]" + " " * max(0, bar_width - filled - 1)
        footer_text = (
            f"  [{bar}]  "
            f"{state.completed_tests}/{state.total_tests}  "
            f"Elapsed: {format_time(elapsed_total)}  "
            f"ETA: {eta_str}  "
            f"[dim]s=skip  q=quit[/dim]"
        )
    else:
        footer_text = "Initializing..."
    layout["footer"].update(Panel(footer_text, style="bold"))

    return layout


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

async def run_benchmark(args):
    concurrency_levels = [int(x) for x in args.concurrency.split(",")]
    context_lengths = [parse_token_value(x) for x in args.contexts.split(",")]
    if args.host.startswith("http://") or args.host.startswith("https://"):
        base_url = args.host.rstrip("/")
        # Append --port if explicitly provided and URL doesn't already contain one
        if args.port is not None:
            from urllib.parse import urlparse
            parsed = urlparse(base_url)
            if parsed.port is None:
                base_url = f"{parsed.scheme}://{parsed.hostname}:{args.port}{parsed.path}"
    else:
        port = args.port if args.port is not None else 5000
        base_url = f"http://{args.host}:{port}"
    server_label = base_url.replace("http://", "").replace("https://", "")
    auth_headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    console = Console()

    # --kv-budget overrides --max-total-tokens
    if args.kv_budget > 0:
        args.max_total_tokens = args.kv_budget
        console.print(f"[cyan]KV cache budget (manual):[/cyan] {args.max_total_tokens:,} tokens")

    # --- Step 1: Connect to server, detect engine, and read limits ---
    server_context_length = 0
    max_running = None
    engine = ENGINE_SGLANG
    async with httpx.AsyncClient(headers=auth_headers) as check_client:
        try:
            resp = await check_client.get(f"{base_url}/v1/models", timeout=10.0)
            models = resp.json()
            model_ids = [m['id'] for m in models.get('data', [])]
            # Auto-detect model name from server if using default
            if args.model == "Qwen3.5" and model_ids:
                args.model = model_ids[0]
            # Get max_model_len (works for both SGLang and vLLM)
            model_data = models.get("data", [])
            if model_data:
                server_context_length = model_data[0].get("max_model_len", 0) or 0
        except Exception as e:
            console.print(f"[red]Cannot connect to server at {base_url}: {e}[/red]")
            console.print("Make sure SGLang or vLLM is running and the port is correct.")
            return [], {}, engine

        # Detect engine: try SGLang's /get_server_info first, then vLLM's /version
        try:
            resp = await check_client.get(f"{base_url}/get_server_info", timeout=10.0)
            server_info = resp.json()
            if "max_total_num_tokens" in server_info:
                engine = ENGINE_SGLANG
                console.print(f"[green]Engine: SGLang {server_info.get('version', '?')}[/green]  Models: {model_ids}")
                if args.max_total_tokens == 0:
                    kv_budget = server_info.get("max_total_num_tokens", 0)
                    if kv_budget:
                        args.max_total_tokens = int(kv_budget)
                        console.print(f"[cyan]KV cache budget:[/cyan] {args.max_total_tokens:,} tokens")
                max_running = server_info.get("max_running_requests")
                if max_running:
                    max_running = int(max_running)
                server_context_length = server_info.get("context_length") or server_context_length
                # Verify that --enable-metrics is active
                try:
                    metrics_resp = await check_client.get(f"{base_url}/metrics", timeout=5.0)
                    if "sglang:" not in metrics_resp.text[:2000]:
                        console.print(
                            "[bold red]ERROR: SGLang server does not have metrics enabled.[/bold red]\n"
                            "This benchmark requires Prometheus metrics for accurate server-side throughput measurement.\n"
                            "Restart SGLang with [bold]--enable-metrics[/bold] flag.\n\n"
                            "Example:\n"
                            "  python -m sglang.launch_server --model ... --enable-metrics"
                        )
                        sys.exit(1)
                except httpx.HTTPError:
                    console.print(
                        "[bold red]ERROR: Cannot reach SGLang /metrics endpoint.[/bold red]\n"
                        "Restart SGLang with [bold]--enable-metrics[/bold] flag."
                    )
                    sys.exit(1)
            else:
                raise ValueError("Not SGLang")
        except Exception:
            # Not SGLang — try vLLM
            try:
                resp = await check_client.get(f"{base_url}/version", timeout=10.0)
                version_info = resp.json()
                if "version" in version_info:
                    engine = ENGINE_VLLM
                    console.print(f"[green]Engine: vLLM {version_info['version']}[/green]  Models: {model_ids}")
                else:
                    raise ValueError("No version")
            except Exception:
                # Fallback: check /metrics prefix
                try:
                    resp = await check_client.get(f"{base_url}/metrics", timeout=5.0)
                    if "vllm:" in resp.text[:2000]:
                        engine = ENGINE_VLLM
                        console.print(f"[green]Engine: vLLM (detected from metrics)[/green]  Models: {model_ids}")
                    else:
                        engine = ENGINE_SGLANG
                        console.print(f"[green]Engine: SGLang (assumed)[/green]  Models: {model_ids}")
                except Exception:
                    console.print(f"[yellow]Could not detect engine. Assuming SGLang.[/yellow]  Models: {model_ids}")

        # vLLM: KV budget cannot be reliably derived from metrics (block_size includes
        # non-KV state for MoE/hybrid models). Rely on --kv-budget or queue detection.
        if engine == ENGINE_VLLM and args.max_total_tokens == 0:
            console.print(
                "[yellow]vLLM: KV cache budget not available from metrics. "
                "Use --kv-budget to skip over-capacity cells, or rely on queue detection.[/yellow]"
            )

        # Handle max_running_requests
        if max_running:
            over = [c for c in concurrency_levels if c > max_running]
            if over:
                concurrency_levels = [c for c in concurrency_levels if c <= max_running]
                console.print(f"[cyan]Max running requests:[/cyan] {max_running} (dropped C={','.join(str(c) for c in over)})")
        if server_context_length:
            console.print(f"[cyan]Model context length:[/cyan] {server_context_length:,} tokens")

    # --- Step 2: Build prefill context list (up to 200k or model limit) ---
    PREFILL_CANDIDATES = [8192, 16384, 32768, 65536, 131072]
    max_prefill = min(131072, server_context_length - 64) if server_context_length > 0 else 131072
    prefill_contexts = [c for c in PREFILL_CANDIDATES if c <= max_prefill]
    if not prefill_contexts and max_prefill > 0:
        prefill_contexts = [max_prefill]
    console.print(f"[cyan]Prefill tests:[/cyan] {[format_context(c) for c in prefill_contexts]}")

    # --- Step 3: Generate padding text calibrated to actual token counts ---
    # Calibrate chars_per_token ratio once on a small context (~8k), then extrapolate.
    # This avoids sending huge requests (128k+) just for calibration.
    all_ctx_sizes = sorted(set(prefill_contexts + [c for c in context_lengths if c > 0]))
    max_ctx = max(all_ctx_sizes) if all_ctx_sizes else 0
    context_cache = {}
    run_id = ''.join(random.choices(string.ascii_lowercase, k=12))
    if max_ctx > 0:
        console.print(f"[bold]Calibrating padding text (run={run_id}, up to {format_context(max_ctx)})...[/bold]")
        base_text = generate_padding_text(int(max_ctx * 1.5))

        # Calibrate on small context: send ~8k tokens, measure actual, derive ratio
        calibrated_cpt = CHARS_PER_TOKEN
        cal_ctx = min(8192, max_ctx)
        async with httpx.AsyncClient(headers=auth_headers) as cal_client:
            prefix = f"[BENCH_{run_id}_CAL] "
            for attempt in range(3):
                target_chars = int(cal_ctx * calibrated_cpt)
                cal_text = (prefix + base_text)[:target_chars]
                msgs = build_messages(cal_ctx, cal_text)
                payload = {
                    "model": args.model, "messages": msgs,
                    "stream": False, "max_tokens": 1,
                    "stream_options": {"include_usage": True},
                }
                try:
                    resp = await cal_client.post(
                        f"{base_url}/v1/chat/completions",
                        json=payload,
                        timeout=httpx.Timeout(120.0, connect=30.0),
                    )
                    data = resp.json()
                    actual = data.get("usage", {}).get("prompt_tokens", 0)
                    if actual > 0:
                        calibrated_cpt = len(cal_text) / actual
                        if abs(actual - cal_ctx) / cal_ctx < 0.05:
                            break
                except Exception:
                    break
        console.print(f"  Calibrated: {calibrated_cpt:.2f} chars/token (from {format_context(cal_ctx)} probe)")

        # Generate text for each context size using calibrated ratio
        for ctx in all_ctx_sizes:
            prefix = f"[BENCH_{run_id}_CTX_{ctx}] "
            target_chars = int(ctx * calibrated_cpt)
            text = (prefix + base_text)[:target_chars]
            context_cache[ctx] = text
            est_tokens = int(len(text) / calibrated_cpt)
            console.print(f"  {format_context(ctx)}: {len(text):,} chars (~{est_tokens:,} tokens)")
    context_cache[0] = ""
    console.print("[green]Done.[/green]\n")

    # --- Step 4: Initialize TUI state ---
    state = TUIState(
        engine=engine,
        model_name=args.model,
        server_url=server_label,
        total_tests=len(concurrency_levels) * len(context_lengths),
        concurrency_levels=concurrency_levels,
        context_lengths=context_lengths,
        overall_start=time.monotonic(),
    )
    if max_running:
        state.max_running_requests = int(max_running)
    state.max_tokens = args.max_tokens
    state.prefill_contexts = prefill_contexts

    # Mark skipped decode cells
    max_run = state.max_running_requests or 0

    def _should_skip(ctx, conc):
        if args.max_total_tokens <= 0:
            return False
        # Never skip if concurrency is within server's max_running_requests
        # and context is 0 (server manages KV allocation dynamically)
        if ctx == 0 and max_run > 0 and conc <= max_run:
            return False
        return conc * (ctx + args.max_tokens) > args.max_total_tokens

    if args.max_total_tokens > 0:
        state.kv_cache_budget = args.max_total_tokens

        runnable = sum(
            1 for ctx in context_lengths for conc in concurrency_levels
            if not _should_skip(ctx, conc)
        )
        skipped = state.total_tests - runnable
        state.skipped_cells = skipped
        for ctx in context_lengths:
            for conc in concurrency_levels:
                if _should_skip(ctx, conc):
                    state.results[(ctx, conc)] = -1

    # Add prefill tests to total count (unless skipped)
    if args.skip_prefill:
        prefill_contexts = []
        state.prefill_contexts = []
    state.total_tests += len(prefill_contexts)

    # Run benchmark
    global _partial_results
    all_results = []
    max_conc = max(concurrency_levels)
    limits = httpx.Limits(max_connections=max_conc + 20, max_keepalive_connections=max_conc + 10)

    async def measure_ttft(client, messages):
        """Send one streaming request with max_tokens=1, return (TTFT, prompt_tokens).
        prompt_tokens is None if not available from server."""
        payload = {
            "model": args.model,
            "messages": messages,
            "stream": True,
            "max_tokens": 1,
            "stream_options": {"include_usage": True},
        }
        t0 = time.monotonic()
        ttft = None
        prompt_tokens = None
        try:
            async with client.stream(
                "POST", f"{base_url}/v1/chat/completions",
                json=payload,
                timeout=httpx.Timeout(600.0, connect=30.0),
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    # Capture usage (comes in final chunk)
                    usage = data.get("usage")
                    if usage and "prompt_tokens" in usage:
                        prompt_tokens = usage["prompt_tokens"]
                    if ttft is None and "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                            ttft = time.monotonic() - t0
        except Exception:
            pass
        if ttft is None:
            ttft = time.monotonic() - t0
        return ttft, prompt_tokens

    async with httpx.AsyncClient(limits=limits, headers=auth_headers) as client:
        with Live(build_display(state), refresh_per_second=2, console=console) as live:

            # === Phase 1: Prefill benchmark (C=1, max_tokens=1) ===
            if prefill_contexts:
                # Warmup: trigger CUDA graph compilation before measurements.
                state.prefill_phase = True
                state.current_context = 0
                state.cell_running = True
                state.cell_start = time.monotonic()
                live.update(build_display(state))

                # Warmup 1: short decode (triggers decode CUDA graphs / JIT compilation)
                warmup_payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": "Count from 1 to 20."}],
                    "stream": True, "max_tokens": 64,
                }
                try:
                    async with client.stream(
                        "POST", f"{base_url}/v1/chat/completions",
                        json=warmup_payload,
                        timeout=httpx.Timeout(300.0, connect=30.0),
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if line and line.startswith("data: [DONE]"):
                                break
                except Exception:
                    pass

                # Warmup 2: prefill with smallest test context (triggers prefill CUDA graphs)
                warmup_ctx = prefill_contexts[0]
                warmup_prefix = f"[WARMUP_{run_id}] "
                warmup_text = warmup_prefix + (context_cache.get(warmup_ctx, "") or generate_padding_text(warmup_ctx))
                warmup_text = warmup_text[:warmup_ctx * CHARS_PER_TOKEN]
                warmup_msgs = build_messages(warmup_ctx, warmup_text)
                await measure_ttft(client, warmup_msgs)

                # Measure baseline TTFT (ctx=0) to subtract overhead
                baseline_msgs = [{"role": "user", "content": "Say OK."}]
                baseline_samples = []
                for _ in range(5):
                    t, _ = await measure_ttft(client, baseline_msgs)
                    baseline_samples.append(t)
                baseline_ttft = median(baseline_samples)
                state.cell_running = False

                # Measure each prefill context: repeat for a fixed duration to collect
                # many samples. Each request uses a unique prefix to avoid cache hits.
                PREFILL_DURATION = 10.0  # seconds per context size

                for ctx in prefill_contexts:
                    state.current_concurrency = 1
                    state.current_context = ctx
                    state.cell_running = True
                    state.cell_start = time.monotonic()
                    state.cell_duration = PREFILL_DURATION
                    live.update(build_display(state))

                    ttft_samples = []
                    actual_prompt_tokens = None
                    orig_prefix_len = len(f"[BENCH_{run_id}_CTX_{ctx}] ")
                    r = 0
                    skipped = False
                    while True:
                        elapsed = time.monotonic() - state.cell_start
                        if elapsed >= PREFILL_DURATION and len(ttft_samples) >= 2:
                            break
                        if _skip_event.is_set():
                            _skip_event.clear()
                            skipped = True
                            break
                        # Unique prefix per iteration → no prefix cache hit
                        prefix = f"[BENCH_{run_id}_CTX_{ctx}_R{r}] "
                        variant_text = prefix + context_cache[ctx][orig_prefix_len:]
                        msgs = build_messages(ctx, variant_text)
                        t, pt = await measure_ttft(client, msgs)
                        ttft_samples.append(t)
                        if pt is not None and actual_prompt_tokens is None:
                            actual_prompt_tokens = pt
                        r += 1
                        live.update(build_display(state))

                    if skipped or not ttft_samples:
                        state.prefill_results[ctx] = {
                            "ttft": 0, "prefill_time": 0, "tok_per_sec": 0,
                            "baseline": baseline_ttft, "samples": 0,
                            "prompt_tokens": 0, "skipped": True,
                        }
                    else:
                        raw_ttft = median(ttft_samples)
                        prefill_time = max(raw_ttft - baseline_ttft, 0.001)
                        token_count = actual_prompt_tokens if actual_prompt_tokens else ctx
                        tok_per_sec = token_count / prefill_time

                        state.prefill_results[ctx] = {
                            "ttft": raw_ttft,
                            "prefill_time": prefill_time,
                            "tok_per_sec": tok_per_sec,
                            "baseline": baseline_ttft,
                            "samples": len(ttft_samples),
                            "prompt_tokens": token_count,
                        }

                    state.cell_running = False
                    state.completed_tests += 1
                    cell_time = time.monotonic() - state.cell_start
                    state.cell_times.append(cell_time)
                    live.update(build_display(state))
                    await asyncio.sleep(1.0)

            # === Warmup probe: ensure CUDA graphs / JIT compiled before decode ===
            # This runs even with --skip-prefill. Sends a short request to trigger
            # kernel compilation so the first decode cell isn't contaminated.
            if not prefill_contexts:
                state.prefill_phase = True
                state.current_context = 0
                state.cell_running = True
                state.cell_start = time.monotonic()
                live.update(build_display(state))

                warmup_payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": "Count from 1 to 20."}],
                    "stream": True, "max_tokens": 64,
                }
                try:
                    async with client.stream(
                        "POST", f"{base_url}/v1/chat/completions",
                        json=warmup_payload,
                        timeout=httpx.Timeout(300.0, connect=30.0),
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if line and line.startswith("data: [DONE]"):
                                break
                except Exception:
                    pass
                state.cell_running = False
                live.update(build_display(state))

            # === Phase 2: Decode benchmark (cached prefill, pure decode speed) ===
            # Cache warming per context is handled by scout request in run_one_cell.
            # Order: 1) first column (C=first, all ctx) — baseline
            #        2) first row (ctx=first, all C) — concurrency scaling
            #        3) rest row by row
            state.prefill_phase = False
            first_conc = concurrency_levels[0]
            first_ctx = context_lengths[0]
            test_order = []
            # 1) First column: C=first, all contexts
            for ctx in context_lengths:
                test_order.append((ctx, first_conc))
            # 2) First row: ctx=first, remaining C
            for conc in concurrency_levels[1:]:
                test_order.append((first_ctx, conc))
            # 3) Rest: row by row, skip already added
            done_set = set(test_order)
            for ctx in context_lengths[1:]:
                for conc in concurrency_levels[1:]:
                    if (ctx, conc) not in done_set:
                        test_order.append((ctx, conc))

            for ctx, conc in test_order:
                    # Skip cells that exceed token budget
                    if args.max_total_tokens > 0 and _should_skip(ctx, conc):
                        state.results[(ctx, conc)] = -1  # mark as skipped
                        cell = CellResult(concurrency=conc, context_tokens=ctx, aggregate_tps=-1)
                        all_results.append(cell)
                        _partial_results = all_results
                        state.completed_tests += 1
                        live.update(build_display(state))
                        continue

                    cell_start = time.monotonic()

                    try:
                        result = await run_one_cell(
                            client=client,
                            base_url=base_url,
                            concurrency=conc,
                            context_tokens=ctx,
                            context_text=context_cache[ctx],
                            duration=args.duration,
                            max_tokens=args.max_tokens,
                            model=args.model,
                            state=state,
                            live=live,
                            engine=engine,
                            auth_headers=auth_headers,
                        )
                        if result.aggregate_tps == -2:
                            state.results[(ctx, conc)] = -2
                        all_results.append(result)
                        _partial_results = all_results
                    except Exception as e:
                        console.print(f"[red]Cell C={conc} ctx={format_context(ctx)} failed: {e}[/red]")
                        cell = CellResult(concurrency=conc, context_tokens=ctx)
                        all_results.append(cell)
                        _partial_results = all_results
                        state.results[(ctx, conc)] = 0.0
                        state.errors[(ctx, conc)] = conc

                    cell_time = time.monotonic() - cell_start
                    state.cell_times.append(cell_time)
                    state.completed_tests += 1
                    live.update(build_display(state))

                    # Brief pause between cells to let server settle
                    await asyncio.sleep(2.0)

    return all_results, state.prefill_results, engine


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------

def print_final_results(results: list, concurrency_levels: list, context_lengths: list,
                        console: Console, prefill_results: dict = None):
    console.print("\n")

    # Prefill table
    if prefill_results:
        baseline = next(iter(prefill_results.values()), {}).get("baseline", 0)
        pt = Table(title=f"Prefill Speed (C=1, baseline TTFT={baseline:.3f}s subtracted)",
                   border_style="magenta")
        pt.add_column("Context", style="bold cyan")
        pt.add_column("Tokens", justify="right")
        pt.add_column("TTFT (s)", justify="right")
        pt.add_column("Prefill (s)", justify="right")
        pt.add_column("Prefill tok/s", justify="right")
        pt.add_column("N", justify="right")
        for ctx in sorted(prefill_results.keys()):
            pr = prefill_results[ctx]
            ptok = pr.get('prompt_tokens', ctx)
            pt.add_row(
                format_context(ctx),
                f"{ptok:,}" if ptok != ctx else f"~{ctx:,}",
                f"{pr['ttft']:.2f}",
                f"{pr.get('prefill_time', pr['ttft']):.2f}",
                f"{pr['tok_per_sec']:,.0f}",
                str(pr.get('samples', '?')),
            )
        console.print(pt)
        console.print()

    # Aggregate throughput table
    table = Table(title="Aggregate Throughput (tok/s)", border_style="green")
    table.add_column("ctx \\ conc", style="bold cyan")
    for conc in concurrency_levels:
        table.add_column(str(conc), justify="right")

    result_map = {(r.context_tokens, r.concurrency): r for r in results}
    any_queued = any(r.avg_queue_reqs > 0 for r in results if r.aggregate_tps >= 0)

    for ctx in context_lengths:
        row = [format_context(ctx)]
        for conc in concurrency_levels:
            r = result_map.get((ctx, conc))
            if r and r.aggregate_tps < 0:
                row.append("skip")
            elif r:
                val = f"{r.aggregate_tps:.1f}"
                if r.num_errors > 0:
                    val += f" ({r.num_errors}e)"
                if r.avg_queue_reqs > 0:
                    val += f" ({r.avg_running_reqs:.0f}/{conc})"
                row.append(val)
            else:
                row.append("-")
        table.add_row(*row)

    console.print(table)
    if any_queued:
        console.print("[dim](X/Y) = avg running / requested concurrency — requests were queued[/dim]")

    # Per-request avg tok/s table
    table2 = Table(title="Per-Request Avg Throughput (tok/s)", border_style="blue")
    table2.add_column("ctx \\ conc", style="bold cyan")
    for conc in concurrency_levels:
        table2.add_column(str(conc), justify="right")

    for ctx in context_lengths:
        row = [format_context(ctx)]
        for conc in concurrency_levels:
            r = result_map.get((ctx, conc))
            if r and r.aggregate_tps < 0:
                row.append("skip")
            elif r and r.per_request_avg_tps > 0:
                val = f"{r.per_request_avg_tps:.1f}"
                if r.avg_queue_reqs > 0:
                    val += f" ({r.avg_running_reqs:.0f}/{conc})"
                row.append(val)
            else:
                row.append("-")
        table2.add_row(*row)

    console.print(table2)

    # TTFT table
    table3 = Table(title=f"Avg TTFT (seconds)  [dim]llm-decode-bench v{VERSION}[/dim]",
                   border_style="yellow")
    table3.add_column("ctx \\ conc", style="bold cyan")
    for conc in concurrency_levels:
        table3.add_column(str(conc), justify="right")

    for ctx in context_lengths:
        row = [format_context(ctx)]
        for conc in concurrency_levels:
            r = result_map.get((ctx, conc))
            if r and r.aggregate_tps < 0:
                row.append("skip")
            elif r and r.ttft_avg > 0:
                row.append(f"{r.ttft_avg:.2f}")
            else:
                row.append("-")
        table3.add_row(*row)

    console.print(table3)


def save_results(results: list, args, filepath: str, prefill_results: dict = None, engine: str = ""):
    concurrency_levels = [int(x) for x in args.concurrency.split(",")]
    context_lengths = [parse_token_value(x) for x in args.contexts.split(",")]

    # Build summary table (exclude skipped)
    summary = {}
    actual_results = [r for r in results if r.aggregate_tps >= 0]
    for r in actual_results:
        ctx_key = str(r.context_tokens)
        if ctx_key not in summary:
            summary[ctx_key] = {}
        summary[ctx_key][str(r.concurrency)] = r.aggregate_tps

    # Prefill summary
    prefill_summary = {}
    if prefill_results:
        for ctx, pr in sorted(prefill_results.items()):
            prefill_summary[str(ctx)] = {
                "ttft_seconds": round(pr["ttft"], 3),
                "tok_per_sec": round(pr["tok_per_sec"], 0),
            }

    output = {
        "metadata": {
            "version": VERSION,
            "engine": engine,
            "model": args.model,
            "server": args.host if args.host.startswith("http") else f"{args.host}:{args.port or 5000}",
            "timestamp": datetime.now().isoformat(),
            "duration_per_test": args.duration,
            "max_tokens": args.max_tokens,
            "max_total_tokens": args.max_total_tokens,
            "concurrency_levels": concurrency_levels,
            "context_lengths": context_lengths,
        },
        "prefill": prefill_summary,
        "results": [asdict(r) for r in actual_results],
        "summary_table": summary,
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Global skip event — set by keyboard listener when 's' is pressed
_skip_event = threading.Event()
_original_term_settings = None


def _keyboard_listener():
    """Background thread: listen for 's' key to skip current cell."""
    global _original_term_settings
    try:
        fd = sys.stdin.fileno()
        _original_term_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        while True:
            if select.select([sys.stdin], [], [], 0.2)[0]:
                ch = sys.stdin.read(1)
                if ch.lower() == "s":
                    _skip_event.set()
                elif ch.lower() == "q":
                    _restore_terminal()
                    os._exit(0)
    except Exception:
        pass  # not a terminal (piped input, etc.)


def _restore_terminal():
    """Restore terminal to original settings (undo cbreak mode)."""
    if _original_term_settings is not None:
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _original_term_settings)
        except Exception:
            pass


GITHUB_RAW_URL = "https://raw.githubusercontent.com/voipmonitor/llm-inference-bench/main/llm_decode_bench.py"


def parse_version(v: str) -> tuple:
    """Parse version string like '0.2.0' into tuple (0, 2, 0) for comparison."""
    return tuple(int(x) for x in v.strip().split("."))


def check_for_update(console: Console) -> bool:
    """Check GitHub for newer version. Returns True if user chose to upgrade and re-exec."""
    try:
        import urllib.request
        req = urllib.request.Request(GITHUB_RAW_URL, method="GET")
        req.add_header("User-Agent", f"llm-decode-bench/{VERSION}")
        with urllib.request.urlopen(req, timeout=5) as resp:
            # Read only first 2KB to find VERSION line
            head = resp.read(2048).decode("utf-8", errors="ignore")
        m = re.search(r'^VERSION\s*=\s*"([^"]+)"', head, re.MULTILINE)
        if not m:
            return False
        remote_version = m.group(1)
        if parse_version(remote_version) <= parse_version(VERSION):
            return False

        console.print(
            f"\n[bold yellow]New version available: v{remote_version} (current: v{VERSION})[/bold yellow]"
        )
        answer = console.input("[bold]Upgrade and restart? [Y/n]: [/bold]").strip().lower()
        if answer in ("", "y", "yes"):
            # Try git pull first (cleanest)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            git_dir = os.path.join(script_dir, ".git")
            if os.path.isdir(git_dir):
                console.print("[cyan]Upgrading via git pull...[/cyan]")
                script_name = os.path.basename(__file__)
                r1 = subprocess.run(
                    ["git", "fetch", "origin", "main"],
                    cwd=script_dir, capture_output=True, text=True,
                )
                r2 = subprocess.run(
                    ["git", "checkout", "origin/main", "--", script_name],
                    cwd=script_dir, capture_output=True, text=True,
                )
                if r1.returncode == 0 and r2.returncode == 0:
                    console.print(f"[green]Updated to v{remote_version}. Restarting...[/green]\n")
                    os.execv(sys.executable, [sys.executable] + sys.argv)
                else:
                    err = r1.stderr.strip() or r2.stderr.strip()
                    console.print(f"[red]git update failed: {err}[/red]")
                    return False
            else:
                # Fallback: download file directly
                console.print("[cyan]Downloading update...[/cyan]")
                script_path = os.path.abspath(__file__)
                with urllib.request.urlopen(GITHUB_RAW_URL, timeout=30) as resp:
                    new_content = resp.read()
                with open(script_path, "wb") as f:
                    f.write(new_content)
                console.print(f"[green]Updated to v{remote_version}. Restarting...[/green]\n")
                os.execv(sys.executable, [sys.executable] + sys.argv)
        else:
            console.print("[dim]Skipping update.[/dim]\n")
    except Exception:
        pass  # Network error, no git, etc. — silently continue
    return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Inference Benchmark with Rich TUI Dashboard (SGLang + vLLM)"
    )
    parser.add_argument(
        "--host", default="localhost",
        help="Server host or full URL (default: localhost). "
             "Accepts a full URL with scheme (e.g. https://ai.example.com) for HTTPS endpoints."
    )
    parser.add_argument("--port", type=int, default=None, help="Server port (default: 5000, or appended to URL when --host is a URL)")
    parser.add_argument(
        "--api-key", default="",
        help="API key for authenticated endpoints (sent as Authorization: Bearer header)"
    )
    parser.add_argument(
        "--concurrency", default="1,2,4,8,16,32,64,128",
        help="Comma-separated concurrency levels (default: 1,2,4,8,16,32,64,128)"
    )
    parser.add_argument(
        "--contexts", default="0,16k,32k,64k,128k",
        help="Comma-separated context lengths in tokens, supports k suffix (default: 0,16k,32k,64k,128k)"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="Duration per test cell in seconds (default: 30)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Max tokens to generate per request (default: 2048)"
    )
    parser.add_argument(
        "--output", default="benchmark_results.json",
        help="Output JSON file path (default: benchmark_results.json)"
    )
    parser.add_argument(
        "--model", default="Qwen3.5",
        help="Model name for API requests (default: Qwen3.5)"
    )
    parser.add_argument(
        "--max-total-tokens", type=int, default=0,
        help="Max total tokens budget (concurrency × (context + max_tokens)). "
             "Cells exceeding this are skipped. 0 = no limit (default: 0)"
    )
    parser.add_argument(
        "--kv-budget", type=int, default=0,
        help="KV cache budget in tokens. Overrides auto-detection. "
             "Cells where concurrency × (context + max_tokens) > budget are skipped. "
             "Use this for vLLM where auto-detection is unreliable. (default: 0 = auto-detect)"
    )
    parser.add_argument(
        "--skip-prefill", action="store_true",
        help="Skip the prefill benchmark phase (Phase 1) and go straight to decode tests"
    )
    return parser.parse_args()


_partial_results: list = []
_prefill_results: dict = {}


def main():
    global _partial_results, _prefill_results
    console = Console()
    check_for_update(console)
    args = parse_args()

    # Start keyboard listener (background daemon thread)
    kb_thread = threading.Thread(target=_keyboard_listener, daemon=True)
    kb_thread.start()

    concurrency_levels = [int(x) for x in args.concurrency.split(",")]
    context_lengths = [parse_token_value(x) for x in args.contexts.split(",")]
    decode_count = len(concurrency_levels) * len(context_lengths)

    console.print(Panel(
        f"[bold cyan]LLM Inference Benchmark[/bold cyan]\n"
        f"Model: {args.model} @ {args.host if args.host.startswith('http') else f'{args.host}:{args.port or 5000}'}\n"
        f"Decode concurrency: {concurrency_levels}\n"
        f"Decode contexts: {[format_context(c) for c in context_lengths]}\n"
        f"Duration: {args.duration}s per decode test | Max tokens: {args.max_tokens}\n"
        f"Phase 1: prefill (auto, up to 128k) | Phase 2: {decode_count} decode tests (cached)",
        title="Configuration",
        border_style="cyan",
    ))

    engine = ""
    try:
        results, prefill_results, engine = asyncio.run(run_benchmark(args))
        _prefill_results = prefill_results
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Saving partial results...[/yellow]")
        results = _partial_results
        prefill_results = _prefill_results

    if results or prefill_results:
        print_final_results(results, concurrency_levels, context_lengths, console, prefill_results)
        save_results(results, args, args.output, prefill_results, engine=engine)
        console.print(f"\n[green]Results saved to {args.output}[/green]")
    else:
        console.print("[red]No results collected.[/red]")


if __name__ == "__main__":
    try:
        main()
    finally:
        _restore_terminal()
