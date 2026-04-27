#!/usr/bin/env python3
"""
LLM Inference Benchmark with Rich TUI Dashboard.

Measures decode throughput across a matrix of concurrency levels and context lengths.
Auto-detects SGLang or vLLM engine and adapts metrics accordingly.

Usage:
    python3 llm_decode_bench.py
    python3 llm_decode_bench.py --port 5199 --concurrency 1,2,4 --contexts 0,16384
    python3 llm_decode_bench.py --port 5199 --kv-budget 692736
    python3 llm_decode_bench.py --port 5002 --dcp-size 8
    python3 llm_decode_bench.py --port 5001 --max-tokens 4096
    python3 llm_decode_bench.py --host https://openrouter.ai --api-key sk-or-... --model meta-llama/llama-3-70b
    python3 llm_decode_bench.py --skip-prefill --concurrency 1,2,4 --contexts 0
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import select
import shutil
import signal
import string
import subprocess
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass, field, asdict
from datetime import datetime
from statistics import mean, median, pstdev
from typing import Optional
from urllib.parse import urlparse

import httpx
from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERSION = "0.4.6"

CHARS_PER_TOKEN = 4
DEFAULT_CALIBRATION_CACHE = "/tmp/llm_decode_bench_token_calibration_cache.json"

PANEL_BOX = box.ROUNDED
HEADER_BOX = box.ROUNDED
TABLE_BOX = box.SIMPLE_HEAD
REPORT_BOX = box.ROUNDED
SUBTLE_BORDER = "#005f2f"
FRAME_BORDER = "#00af5f"
PHOSPHOR = "#00ff66"
PHOSPHOR_DIM = "#00cc55"
PHOSPHOR_SOFT = "#66ff99"
PHOSPHOR_WARN = "#ff5555"
REPORT_MUTED = "#8fd6a3"
TITLE_COLOR = "#008f4f"
TITLE_STYLE = f"bold {TITLE_COLOR}"
TEXT_PRIMARY = "#d7ffe6"
THEME_ERROR = "#ff5555"
CAPACITY_LIMIT_MARK = "∅"

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
ENGINE_OPENAI_PROXY = "openai_proxy"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RequestSample:
    ttft: float = 0.0
    time_to_second_token: float = 0.0
    latency: float = 0.0
    inter_token_latency_avg: float = 0.0
    chunk_inter_token_latency_avg: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    output_tps_per_user: float = 0.0
    e2e_output_tps_per_user: float = 0.0
    completed: bool = False


@dataclass
class StreamResult:
    ttft: float = 0.0
    total_tokens: int = 0
    total_time: float = 0.0
    tokens_per_sec: float = 0.0
    error: Optional[str] = None
    request_samples: list = field(default_factory=list)


@dataclass
class CellResult:
    concurrency: int = 0
    context_tokens: int = 0
    benchmark_mode: str = "duration"
    request_count_target: int = 0
    warmup_request_count: int = 0
    measurement_seconds: float = 0.0
    client_output_tokens: int = 0
    server_output_tokens: int = 0
    aggregate_source: str = ""
    aggregate_tps: float = 0.0
    per_request_avg_tps: float = 0.0
    ttft_avg: float = 0.0
    ttft_p50: float = 0.0
    ttft_p90: float = 0.0
    ttft_p99: float = 0.0
    time_to_second_token_avg: float = 0.0
    time_to_second_token_p50: float = 0.0
    time_to_second_token_p90: float = 0.0
    time_to_second_token_p99: float = 0.0
    request_latency_avg: float = 0.0
    request_latency_p50: float = 0.0
    request_latency_p90: float = 0.0
    request_latency_p99: float = 0.0
    inter_token_latency_avg: float = 0.0
    inter_token_latency_p50: float = 0.0
    inter_token_latency_p90: float = 0.0
    inter_token_latency_p99: float = 0.0
    output_tps_per_user_avg: float = 0.0
    output_tps_per_user_p50: float = 0.0
    output_tps_per_user_p90: float = 0.0
    output_tps_per_user_p99: float = 0.0
    e2e_output_tps_per_user_avg: float = 0.0
    e2e_output_tps_per_user_p50: float = 0.0
    e2e_output_tps_per_user_p90: float = 0.0
    e2e_output_tps_per_user_p99: float = 0.0
    chunk_inter_token_latency_avg: float = 0.0
    chunk_inter_token_latency_p50: float = 0.0
    chunk_inter_token_latency_p90: float = 0.0
    chunk_inter_token_latency_p99: float = 0.0
    input_seq_len_avg: float = 0.0
    output_seq_len_avg: float = 0.0
    output_seq_len_p50: float = 0.0
    output_seq_len_p90: float = 0.0
    output_seq_len_p99: float = 0.0
    request_count: int = 0
    completed_request_count: int = 0
    request_samples: list = field(default_factory=list)
    total_tokens: int = 0
    wall_time: float = 0.0
    num_completed: int = 0
    num_errors: int = 0
    server_gen_throughput: float = 0.0
    server_utilization: float = 0.0
    server_spec_accept_rate: float = 0.0
    server_spec_accept_length: float = 0.0
    # Queue / effective concurrency tracking
    avg_running_reqs: float = 0.0
    max_running_reqs: int = 0
    effective_concurrency: float = 0.0
    avg_queue_reqs: float = 0.0
    max_queue_reqs: int = 0
    queue_fraction: float = 0.0  # fraction of samples where queue > 0
    underfilled: bool = False
    warmup_timed_out: bool = False
    warmup_duration: float = 0.0
    ready_reason: str = ""
    timeout_reason: str = ""
    capacity_limited: bool = False
    hardware_summary: dict = field(default_factory=dict)


@dataclass
class GpuStats:
    index: int = 0
    temp_c: float = 0.0
    gpu_util_pct: float = 0.0
    mem_util_pct: float = 0.0
    mem_used_mb: float = 0.0
    mem_total_mb: float = 0.0
    power_w: float = 0.0
    power_limit_w: float = 0.0
    sm_clock_mhz: float = 0.0
    mem_clock_mhz: float = 0.0
    pcie_gen: float = 0.0
    pcie_width: float = 0.0
    pcie_rx_mb_s: float = 0.0
    pcie_tx_mb_s: float = 0.0


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
    benchmark_mode: str = "duration"
    request_count_target: int = 0
    cell_tokens: int = 0
    cell_live_tps: float = 0.0
    cell_tps_history: list = field(default_factory=list)
    cell_running: bool = False
    cell_warmup: bool = False  # True during prefill ramp-up before measurement
    cell_measurement_start: float = 0.0  # when actual measurement begins (after warmup)
    cell_request_samples: int = 0
    cell_completed_requests: int = 0
    cell_ttft_p50_ms: float = 0.0
    cell_itl_p50_ms: float = 0.0
    cell_user_tps_p50: float = 0.0
    cell_request_latency_p50_ms: float = 0.0
    cell_request_latency_p90_ms: float = 0.0
    # Server metrics
    metrics_available: bool = True
    metrics_warning: str = ""
    srv_gen_throughput: float = 0.0
    srv_running_reqs: int = 0
    srv_queue_reqs: int = 0
    srv_utilization: float = 0.0
    srv_spec_accept_rate: float = 0.0
    srv_spec_accept_length: float = 0.0
    # Results
    results: dict = field(default_factory=dict)  # (ctx, conc) -> aggregate_tps
    errors: dict = field(default_factory=dict)   # (ctx, conc) -> num_errors
    queue_info: dict = field(default_factory=dict)  # (ctx, conc) -> (avg_running, avg_queue, capacity_limited)
    client_info: dict = field(default_factory=dict)  # (ctx, conc) -> compact per-cell client metrics
    concurrency_levels: list = field(default_factory=list)
    context_lengths: list = field(default_factory=list)
    # Prefill results: ctx -> {ttft, tok_per_sec}
    prefill_results: dict = field(default_factory=dict)
    prefill_contexts: list = field(default_factory=list)
    prefill_phase: bool = False
    prefill_status: str = ""
    prefill_samples_done: int = 0
    prefill_last_tps: float = 0.0
    prefill_last_tokens: int = 0
    prefill_last_seconds: float = 0.0
    prefill_method: str = ""
    # Server limits
    kv_cache_budget: int = 0
    max_running_requests: int = 0
    skipped_cells: int = 0
    max_tokens: int = 0
    show_capacity_limited_values: bool = False
    # Timing
    cell_times: list = field(default_factory=list)
    # Hardware monitor
    hw_monitor_enabled: bool = True
    hw_available: bool = False
    hw_last_error: str = ""
    hw_last_update: float = 0.0
    cpu_util_pct: float = 0.0
    cpu_freq_mhz: float = 0.0
    hw_gpu_limit: int = 8
    gpu_stats: list[GpuStats] = field(default_factory=list)
    hw_history: list = field(default_factory=list)
    events: list = field(default_factory=list)


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


def calibration_template_id() -> str:
    payload = json.dumps(
        {
            "chars_per_token_default": CHARS_PER_TOKEN,
            "padding_sentences": PADDING_SENTENCES,
            "generation_prompt": GENERATION_PROMPT,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def load_calibration_cache(path: str) -> dict:
    if not path:
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_calibration_cache(path: str, data: dict) -> None:
    if not path:
        return
    try:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    except Exception:
        pass


def _to_float(value: str) -> float:
    value = value.strip()
    if not value or value.upper() in ("N/A", "[N/A]"):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _sample_gpu_query() -> list[GpuStats]:
    if shutil.which("nvidia-smi") is None:
        return []
    fields = [
        "index",
        "temperature.gpu",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "power.draw",
        "power.limit",
        "clocks.sm",
        "clocks.mem",
        "pcie.link.gen.current",
        "pcie.link.width.current",
    ]
    proc = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    if proc.returncode != 0:
        return []
    gpus: list[GpuStats] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(fields):
            continue
        try:
            idx = int(_to_float(parts[0]))
        except ValueError:
            continue
        gpus.append(
            GpuStats(
                index=idx,
                temp_c=_to_float(parts[1]),
                gpu_util_pct=_to_float(parts[2]),
                mem_util_pct=_to_float(parts[3]),
                mem_used_mb=_to_float(parts[4]),
                mem_total_mb=_to_float(parts[5]),
                power_w=_to_float(parts[6]),
                power_limit_w=_to_float(parts[7]),
                sm_clock_mhz=_to_float(parts[8]),
                mem_clock_mhz=_to_float(parts[9]),
                pcie_gen=_to_float(parts[10]),
                pcie_width=_to_float(parts[11]),
            )
        )
    return gpus


def _sample_gpu_pcie() -> dict[int, tuple[float, float]]:
    if shutil.which("nvidia-smi") is None:
        return {}
    proc = subprocess.run(
        ["nvidia-smi", "dmon", "-s", "t", "-c", "1"],
        check=False,
        capture_output=True,
        text=True,
        timeout=4,
    )
    if proc.returncode != 0:
        return {}
    pcie = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            pcie[int(parts[0])] = (_to_float(parts[1]), _to_float(parts[2]))
        except ValueError:
            continue
    return pcie


def _sample_cpu_stats() -> tuple[float, float]:
    try:
        import psutil  # optional runtime dependency; benchmark works without it.

        util = float(psutil.cpu_percent(interval=None))
        freq = psutil.cpu_freq()
        return util, float(freq.current if freq else 0.0)
    except Exception:
        return 0.0, 0.0


def add_event(state: TUIState, message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    state.events.append(f"{ts} {message}")
    if len(state.events) > 80:
        state.events = state.events[-80:]


def snapshot_partial_prefill(state: TUIState) -> None:
    """Keep Ctrl-C/q final reports in sync with already measured prefill rows."""
    global _prefill_results
    _prefill_results = dict(state.prefill_results)


def hardware_snapshot(
    gpus: list[GpuStats],
    cpu_util: float,
    cpu_freq: float,
    timestamp: float | None = None,
) -> dict:
    if not gpus:
        return {
            "timestamp": timestamp or time.monotonic(),
            "gpu_count": 0,
            "cpu_util_pct": round(cpu_util, 2),
            "cpu_freq_mhz": round(cpu_freq, 1),
        }
    gpu_utils = [g.gpu_util_pct for g in gpus]
    mem_utils = [g.mem_util_pct for g in gpus]
    temps = [g.temp_c for g in gpus]
    powers = [g.power_w for g in gpus]
    limits = [g.power_limit_w for g in gpus if g.power_limit_w > 0]
    vram_used = sum(g.mem_used_mb for g in gpus)
    vram_total = sum(g.mem_total_mb for g in gpus)
    pcie_rx = sum(g.pcie_rx_mb_s for g in gpus)
    pcie_tx = sum(g.pcie_tx_mb_s for g in gpus)
    return {
        "timestamp": timestamp or time.monotonic(),
        "gpu_count": len(gpus),
        "cpu_util_pct": round(cpu_util, 2),
        "cpu_freq_mhz": round(cpu_freq, 1),
        "gpu_util_avg_pct": round(mean(gpu_utils), 2),
        "gpu_util_max_pct": round(max(gpu_utils), 2),
        "mem_util_avg_pct": round(mean(mem_utils), 2),
        "mem_util_max_pct": round(max(mem_utils), 2),
        "temp_avg_c": round(mean(temps), 2),
        "temp_max_c": round(max(temps), 2),
        "power_total_w": round(sum(powers), 2),
        "power_max_w": round(max(powers), 2),
        "power_limit_total_w": round(sum(limits), 2),
        "vram_used_mb": round(vram_used, 1),
        "vram_total_mb": round(vram_total, 1),
        "vram_used_pct": round((vram_used / vram_total * 100) if vram_total > 0 else 0.0, 2),
        "pcie_rx_total_mb_s": round(pcie_rx, 2),
        "pcie_tx_total_mb_s": round(pcie_tx, 2),
    }


def summarize_hardware_history(samples: list[dict]) -> dict:
    samples = [s for s in samples if s.get("gpu_count", 0) > 0]
    if not samples:
        return {}

    def values(key: str) -> list[float]:
        return [float(s.get(key, 0.0)) for s in samples if s.get(key, 0.0) is not None]

    def avg(key: str) -> float:
        vals = values(key)
        return round(mean(vals), 2) if vals else 0.0

    def maxv(key: str) -> float:
        vals = values(key)
        return round(max(vals), 2) if vals else 0.0

    first_ts = float(samples[0].get("timestamp", 0.0))
    last_ts = float(samples[-1].get("timestamp", first_ts))
    return {
        "samples": len(samples),
        "duration_seconds": round(max(0.0, last_ts - first_ts), 3),
        "gpu_count": max(int(s.get("gpu_count", 0)) for s in samples),
        "cpu_util_avg_pct": avg("cpu_util_pct"),
        "gpu_util_avg_pct": avg("gpu_util_avg_pct"),
        "gpu_util_max_pct": maxv("gpu_util_max_pct"),
        "mem_util_avg_pct": avg("mem_util_avg_pct"),
        "mem_util_max_pct": maxv("mem_util_max_pct"),
        "temp_avg_c": avg("temp_avg_c"),
        "temp_max_c": maxv("temp_max_c"),
        "power_total_avg_w": avg("power_total_w"),
        "power_total_max_w": maxv("power_total_w"),
        "power_limit_total_w": maxv("power_limit_total_w"),
        "vram_used_avg_mb": avg("vram_used_mb"),
        "vram_used_max_mb": maxv("vram_used_mb"),
        "vram_total_mb": maxv("vram_total_mb"),
        "vram_used_avg_pct": avg("vram_used_pct"),
        "vram_used_max_pct": maxv("vram_used_pct"),
        "pcie_rx_avg_mb_s": avg("pcie_rx_total_mb_s"),
        "pcie_rx_max_mb_s": maxv("pcie_rx_total_mb_s"),
        "pcie_tx_avg_mb_s": avg("pcie_tx_total_mb_s"),
        "pcie_tx_max_mb_s": maxv("pcie_tx_total_mb_s"),
    }


def _diag_command(cmd: list[str], timeout: float = 5.0) -> dict:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {
            "cmd": cmd,
            "returncode": -1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def collect_startup_diagnostics(args, base_url: str) -> dict:
    env_prefixes = ("NCCL_", "VLLM_", "SGLANG_", "CUDA_", "OMP_")
    env = {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(env_prefixes)
    }
    diagnostics = {
        "version": VERSION,
        "server_url": base_url,
        "hostname": _diag_command(["hostname"], timeout=2.0).get("stdout", ""),
        "uname": _diag_command(["uname", "-a"], timeout=2.0).get("stdout", ""),
        "env": env,
        "args": {
            "concurrency": args.concurrency,
            "contexts": args.contexts,
            "max_tokens": args.max_tokens,
            "duration": args.duration,
            "request_count": getattr(args, "request_count", 0),
            "run_burst": getattr(args, "run_burst", False),
            "standalone_prefill": getattr(args, "standalone_prefill", False),
            "skip_prefill": getattr(args, "skip_prefill", False),
            "prefill_contexts": getattr(args, "prefill_contexts", ""),
            "prefill_metric": getattr(args, "prefill_metric", ""),
            "dcp_size": getattr(args, "dcp_size", 0),
            "kv_budget": getattr(args, "kv_budget", 0),
        },
    }
    if shutil.which("nvidia-smi"):
        diagnostics["nvidia_smi_query"] = _diag_command([
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,power.limit",
            "--format=csv,noheader,nounits",
        ])
        diagnostics["nvidia_smi_topo"] = _diag_command(["nvidia-smi", "topo", "-m"], timeout=8.0)
    else:
        diagnostics["nvidia_smi_error"] = "nvidia-smi not found"
    return diagnostics


def start_hardware_monitor(state: TUIState, interval: float) -> None:
    if interval <= 0:
        state.hw_monitor_enabled = False
        state.hw_last_error = "hardware monitor disabled"
        return
    if shutil.which("nvidia-smi") is None:
        state.hw_monitor_enabled = False
        state.hw_last_error = "nvidia-smi not found"
        return
    state.hw_monitor_enabled = True

    def loop() -> None:
        while True:
            try:
                gpus = _sample_gpu_query()
                pcie = _sample_gpu_pcie()
                for gpu in gpus:
                    rx, tx = pcie.get(gpu.index, (0.0, 0.0))
                    gpu.pcie_rx_mb_s = rx
                    gpu.pcie_tx_mb_s = tx
                cpu_util, cpu_freq = _sample_cpu_stats()
                state.cpu_util_pct = cpu_util
                state.cpu_freq_mhz = cpu_freq
                state.gpu_stats = gpus
                state.hw_available = bool(gpus)
                state.hw_last_error = "" if gpus else "no GPU samples"
                state.hw_last_update = time.monotonic()
                if gpus:
                    state.hw_history.append(
                        hardware_snapshot(gpus, cpu_util, cpu_freq, state.hw_last_update)
                    )
                    if len(state.hw_history) > 20000:
                        state.hw_history = state.hw_history[-10000:]
            except Exception as exc:
                state.hw_last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(interval)

    thread = threading.Thread(target=loop, daemon=True, name="llmbench-hw-monitor")
    thread.start()


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


def summarize_numbers(values: list[float]) -> dict:
    values = [float(v) for v in values if v is not None and v > 0]
    if not values:
        return {
            "avg": 0.0, "min": 0.0, "max": 0.0,
            "p50": 0.0, "p90": 0.0, "p99": 0.0, "std": 0.0,
        }
    return {
        "avg": mean(values),
        "min": min(values),
        "max": max(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def request_metric_values(samples: list, attr: str, completed_only: bool = False) -> list[float]:
    values = []
    for sample in samples:
        if completed_only and not sample.completed:
            continue
        value = getattr(sample, attr, 0.0)
        if value and value > 0:
            values.append(float(value))
    return values


def summarize_request_samples(samples: list[RequestSample]) -> dict:
    generated = [s for s in samples if s.output_tokens > 0]
    completed = [s for s in generated if s.completed]
    completed_or_generated = completed if completed else generated
    return {
        "request_count": len(generated),
        "completed_request_count": len(completed),
        "ttft": summarize_numbers(request_metric_values(generated, "ttft")),
        "time_to_second_token": summarize_numbers(
            request_metric_values(generated, "time_to_second_token")
        ),
        "request_latency": summarize_numbers(
            request_metric_values(completed, "latency")
        ),
        "inter_token_latency": summarize_numbers(
            request_metric_values(generated, "inter_token_latency_avg")
        ),
        "chunk_inter_token_latency": summarize_numbers(
            request_metric_values(generated, "chunk_inter_token_latency_avg")
        ),
        "output_tps_per_user": summarize_numbers(
            request_metric_values(generated, "output_tps_per_user")
        ),
        "e2e_output_tps_per_user": summarize_numbers(
            request_metric_values(completed, "e2e_output_tps_per_user")
        ),
        "input_seq_len": summarize_numbers(
            request_metric_values(completed_or_generated, "input_tokens")
        ),
        "output_seq_len": summarize_numbers(
            request_metric_values(completed_or_generated, "output_tokens")
        ),
    }


def update_state_request_stats(state: TUIState, samples: list[RequestSample]) -> None:
    summary = summarize_request_samples(samples)
    state.cell_request_samples = summary["request_count"]
    state.cell_completed_requests = summary["completed_request_count"]
    state.cell_ttft_p50_ms = summary["ttft"]["p50"] * 1000
    state.cell_itl_p50_ms = summary["inter_token_latency"]["p50"] * 1000
    state.cell_user_tps_p50 = summary["output_tps_per_user"]["p50"]
    state.cell_request_latency_p50_ms = summary["request_latency"]["p50"] * 1000
    state.cell_request_latency_p90_ms = summary["request_latency"]["p90"] * 1000


def compact_client_info_from_cell(cell: CellResult) -> dict:
    return {
        "ttft_ms": cell.ttft_p50 * 1000,
        "itl_ms": cell.inter_token_latency_p50 * 1000,
        "user_tps": cell.output_tps_per_user_p50,
        "latency_p50_ms": cell.request_latency_p50 * 1000,
        "latency_p90_ms": cell.request_latency_p90 * 1000,
        "request_count": cell.request_count,
        "completed_request_count": cell.completed_request_count,
    }


def format_context(ctx: int) -> str:
    if ctx == 0:
        return "0"
    elif ctx >= 1024:
        return f"{ctx // 1024}k"
    return str(ctx)


def render_title(label: str, suffix: str = "") -> Text:
    """Return a high-contrast title without background/reverse styling."""
    text = Text(label, style=TITLE_STYLE)
    if suffix:
        text.append(f" {suffix}", style=REPORT_MUTED)
    return text


def format_token_budget(tokens: int) -> str:
    if tokens >= 1024 * 1024:
        return f"{tokens / (1024 * 1024):.2f}M"
    if tokens >= 1024:
        return f"{tokens // 1024}k"
    return str(tokens)


def format_kv_missing(needed: int, budget: int) -> str:
    missing = max(0, needed - budget)
    if missing > 0:
        return f"KV+{format_token_budget(missing)}"
    return "KV"


def capacity_limit_cell(styled: bool = False) -> str:
    if styled:
        return f"[{PHOSPHOR_WARN}]{CAPACITY_LIMIT_MARK}[/{PHOSPHOR_WARN}]"
    return CAPACITY_LIMIT_MARK


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def format_prefill_eta(state: TUIState, elapsed: float) -> str:
    """Build a live ETA for long prefill phases from completed prefill samples."""
    target_tokens = max(0, int(getattr(state, "current_context", 0) or 0))
    if target_tokens <= 0:
        return f"[dim]eta: waiting for prefill completion, elapsed {format_time(elapsed)}[/dim]"

    candidates: list[tuple[int, int, int, float, str]] = []
    for raw_ctx, result in getattr(state, "prefill_results", {}).items():
        if not isinstance(result, dict) or result.get("skipped"):
            continue
        rate = float(result.get("tok_per_sec") or 0.0)
        if rate <= 0:
            continue
        try:
            ctx_label = int(raw_ctx)
            observed_tokens = int(result.get("tokens") or ctx_label)
        except (TypeError, ValueError):
            continue
        method = str(result.get("method") or "prefill")
        candidates.append((abs(ctx_label - target_tokens), ctx_label, observed_tokens, rate, method))

    source = ""
    rate = 0.0
    estimated_prompt_tokens = target_tokens
    if state.prefill_last_tps > 0:
        rate = float(state.prefill_last_tps)
        if state.prefill_last_tokens > 0:
            estimated_prompt_tokens = int(state.prefill_last_tokens)
        source = f"last sample @ {rate:,.0f} tok/s"
    elif candidates:
        _, ctx_label, observed_tokens, rate, method = min(candidates, key=lambda item: item[0])
        if ctx_label > 0 and observed_tokens > 0:
            estimated_prompt_tokens = max(1, int(target_tokens * (observed_tokens / ctx_label)))
        source = f"{format_context(ctx_label)} {method} @ {rate:,.0f} tok/s"

    if rate <= 0:
        return (
            f"[dim]eta: waiting for first completed prefill sample; "
            f"elapsed {format_time(elapsed)}[/dim]"
        )

    estimated_total = max(estimated_prompt_tokens / rate, 0.1)
    progress = min(max(elapsed / estimated_total, 0.0), 0.99)
    if elapsed > estimated_total * 1.15:
        return (
            f"[dim]eta: estimate {format_time(estimated_total)} exceeded; "
            f"still waiting ({source})[/dim]"
        )

    remaining = max(0.0, estimated_total - elapsed)
    bar = render_progress_bar(progress, width=18)
    return (
        f"[dim]eta:[/dim] {bar} {progress * 100:>3.0f}%  "
        f"left~{format_time(remaining)}  [dim]({source})[/dim]"
    )


def format_ms_value(seconds: float) -> str:
    if not seconds or seconds <= 0:
        return "—"
    ms = seconds * 1000
    if ms >= 1000:
        return f"{ms:,.0f}"
    if ms >= 100:
        return f"{ms:.0f}"
    if ms >= 10:
        return f"{ms:.1f}"
    return f"{ms:.2f}"


def format_rate_value(value: float) -> str:
    if not value or value <= 0:
        return "—"
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def render_progress_bar(pct: float, width: int = 30) -> str:
    pct = min(max(pct, 0.0), 1.0)
    filled = int(pct * width)
    empty = max(0, width - filled)
    if filled <= 0:
        return f"[{PHOSPHOR_DIM}]○[/{PHOSPHOR_DIM}][{SUBTLE_BORDER}]{'╌' * max(0, width - 1)}[/{SUBTLE_BORDER}]"
    if filled >= width:
        return f"[{PHOSPHOR}]{'━' * width}[/]"
    return (
        f"[{PHOSPHOR}]{'━' * filled}[/]"
        f"[{PHOSPHOR_SOFT}]●[/]"
        f"[{SUBTLE_BORDER}]{'╌' * max(0, empty - 1)}[/{SUBTLE_BORDER}]"
    )


def render_speed_trace(samples: list[float], width: int = 22) -> str:
    if not samples:
        return "[dim]trace: waiting for measured decode samples[/dim]"

    if len(samples) > width:
        bucketed = []
        for i in range(width):
            start = int(i * len(samples) / width)
            end = int((i + 1) * len(samples) / width)
            chunk = samples[start:max(end, start + 1)]
            bucketed.append(sum(chunk) / len(chunk))
        values = bucketed
    else:
        values = samples

    avg = sum(samples) / len(samples)
    spread = ((max(samples) - min(samples)) / avg * 100) if avg > 0 else 0.0
    if avg <= 0:
        return "[dim]trace: waiting for non-zero samples[/dim]"

    # Use a fixed deviation scale around the average. The previous min/max
    # normalization made tiny changes look like huge swings.
    levels = "▁▂▃▄▅▆▇█"
    half_range = max(avg * 0.20, 1.0)  # +/-20% fills the sparkline range.
    center = (len(levels) - 1) / 2

    if spread < 3.0:
        trace = "─" * len(values)
        status = "stable"
    else:
        chars = []
        for v in values:
            rel = (v - avg) / half_range
            idx = round(center + rel * center)
            idx = max(0, min(len(levels) - 1, idx))
            chars.append(levels[idx])
        trace = "".join(chars)
        status = "jitter"

    return (
        f"[dim]trace[/dim] [{PHOSPHOR_DIM}]{trace}[/{PHOSPHOR_DIM}]  "
        f"[dim]avg[/dim]={avg:.0f} [dim]{status}[/dim]={spread:.1f}%"
    )


def color_by_ratio(value: float, warn: float = 0.70, crit: float = 0.90) -> str:
    if value >= crit:
        return "#ff5555"
    if value >= warn:
        return "#ffb86c"
    return PHOSPHOR


def color_temp(temp_c: float) -> str:
    if temp_c >= 82:
        return "#ff5555"
    if temp_c >= 72:
        return "#ffb86c"
    return PHOSPHOR_DIM


def colorize(text: str, color: str) -> str:
    return f"[{color}]{text}[/{color}]"


def format_ghz(mhz: float) -> str:
    if mhz <= 0:
        return "—"
    return f"{mhz / 1000:.1f}G"


def format_gib_from_mb(mb: float) -> str:
    if mb <= 0:
        return "0G"
    return f"{mb / 1024:.1f}G"


def render_hardware_panel(state: TUIState, dense: bool = False) -> Panel:
    if not state.hw_available:
        msg = "Waiting for hardware samples..."
        if state.hw_last_error:
            msg = state.hw_last_error
        return Panel(
            f"[dim]{msg}[/dim]",
            title=render_title("HW" if dense else "HARDWARE"),
            title_align="left",
            box=PANEL_BOX,
            border_style=FRAME_BORDER,
            padding=(0, 0 if dense else 1),
        )

    total_rx = sum(g.pcie_rx_mb_s for g in state.gpu_stats)
    total_tx = sum(g.pcie_tx_mb_s for g in state.gpu_stats)
    cpu = f"CPU {state.cpu_util_pct:.0f}%"
    if state.cpu_freq_mhz > 0:
        cpu += f" {format_ghz(state.cpu_freq_mhz)}"
    age = time.monotonic() - state.hw_last_update if state.hw_last_update else 0

    table = Table(show_header=True, box=None, padding=(0, 0 if dense else 1), expand=True)
    table.add_column("G" if dense else "GPU", style=f"bold {PHOSPHOR_SOFT}", no_wrap=True)
    table.add_column("SM", justify="right", no_wrap=True)
    table.add_column("MC" if dense else "Mem", justify="right", no_wrap=True)
    table.add_column("VRAM", justify="right", no_wrap=True)
    table.add_column("W" if dense else "W now/lim", justify="right", no_wrap=True)
    table.add_column("T", justify="right", no_wrap=True)
    if not dense:
        table.add_column("Clk", justify="right", no_wrap=True)
    table.add_column("PCIe" if dense else "PCIe rx/tx", justify="right", no_wrap=True)

    visible_gpus = state.gpu_stats[:max(1, state.hw_gpu_limit)]
    for gpu in visible_gpus:
        power_ratio = gpu.power_w / gpu.power_limit_w if gpu.power_limit_w > 0 else 0.0
        vram_ratio = gpu.mem_used_mb / gpu.mem_total_mb if gpu.mem_total_mb > 0 else 0.0
        sm = colorize(f"{gpu.gpu_util_pct:.0f}%", PHOSPHOR if gpu.gpu_util_pct >= 60 else PHOSPHOR_DIM)
        mem = colorize(f"{gpu.mem_util_pct:.0f}%", PHOSPHOR if gpu.mem_util_pct >= 60 else PHOSPHOR_DIM)
        vram = colorize(
            f"{format_gib_from_mb(gpu.mem_used_mb)}/{format_gib_from_mb(gpu.mem_total_mb)}",
            color_by_ratio(vram_ratio, 0.85, 0.95),
        )
        power = (
            f"[{color_by_ratio(power_ratio, 0.65, 0.85)}]{gpu.power_w:.0f}[/]"
            f"[dim]/{gpu.power_limit_w:.0f}[/dim]"
        )
        temp = colorize(f"{gpu.temp_c:.0f}C", color_temp(gpu.temp_c))
        clk = f"{format_ghz(gpu.sm_clock_mhz)}/{format_ghz(gpu.mem_clock_mhz)}"
        pcie = f"{gpu.pcie_rx_mb_s:.0f}/{gpu.pcie_tx_mb_s:.0f}"
        row = [f"G{gpu.index}", sm, mem, vram, power, temp]
        if not dense:
            row.append(clk)
        row.append(pcie)
        table.add_row(*row)

    hidden = len(state.gpu_stats) - len(visible_gpus)
    hidden_note = f"  [dim]+{hidden} hidden[/dim]" if hidden > 0 else ""
    if dense:
        header = (
            f"[dim]{cpu}  PCIe[/dim] "
            f"[{PHOSPHOR_DIM}]{total_rx:.0f}/{total_tx:.0f} MB/s[/{PHOSPHOR_DIM}]  "
            f"[dim]age {age:.0f}s[/dim]{hidden_note}"
        )
    else:
        header = (
            f"[dim]{cpu}  PCIe rx/tx[/dim] "
            f"[{PHOSPHOR_DIM}]{total_rx:.0f}/{total_tx:.0f} MB/s[/{PHOSPHOR_DIM}]  "
            f"[dim]age {age:.0f}s[/dim]{hidden_note}\n"
            f"[dim]PCIe rx/tx are MB/s from nvidia-smi dmon; W color is current/limit ratio[/dim]"
        )
    return Panel(
        Group(header, table),
        title=render_title("HW" if dense else "HARDWARE"),
        title_align="left",
        box=PANEL_BOX,
        border_style=FRAME_BORDER,
        padding=(0, 0 if dense else 1),
    )


def render_compact_hardware_panel(state: TUIState, paired: bool = True) -> Panel:
    if not state.hw_available:
        msg = state.hw_last_error or "Waiting for hardware samples..."
        return Panel(
            f"[dim]{msg}[/dim]",
            title=render_title("HW"),
            title_align="left",
            box=PANEL_BOX,
            border_style=FRAME_BORDER,
            padding=(0, 1),
        )

    gpus = state.gpu_stats
    if not gpus:
        content = "[dim]No GPU samples yet[/dim]"
    else:
        total_rx = sum(g.pcie_rx_mb_s for g in gpus)
        total_tx = sum(g.pcie_tx_mb_s for g in gpus)
        avg_sm = mean(g.gpu_util_pct for g in gpus)
        avg_mem = mean(g.mem_util_pct for g in gpus)
        total_used = sum(g.mem_used_mb for g in gpus)
        total_mem = sum(g.mem_total_mb for g in gpus)
        total_w = sum(g.power_w for g in gpus)
        total_limit = sum(g.power_limit_w for g in gpus)
        max_temp = max(g.temp_c for g in gpus)
        power_ratio = total_w / total_limit if total_limit > 0 else 0.0
        vram_ratio = total_used / total_mem if total_mem > 0 else 0.0
        cpu = f"CPU {state.cpu_util_pct:.0f}%"
        if state.cpu_freq_mhz > 0:
            cpu += f" {format_ghz(state.cpu_freq_mhz)}"
        summary = (
            f"[dim]{cpu}[/dim]  "
            f"SM [{PHOSPHOR}]{avg_sm:.0f}%[/{PHOSPHOR}]  "
            f"Mem [{PHOSPHOR_DIM}]{avg_mem:.0f}%[/{PHOSPHOR_DIM}]  "
            f"VRAM [{color_by_ratio(vram_ratio, 0.85, 0.95)}]"
            f"{format_gib_from_mb(total_used)}/{format_gib_from_mb(total_mem)}[/]  "
            f"W [{color_by_ratio(power_ratio, 0.65, 0.85)}]{total_w:.0f}[/][dim]/{total_limit:.0f}[/dim]  "
            f"T [{color_temp(max_temp)}]{max_temp:.0f}C[/]  "
            f"PCIe [{PHOSPHOR_DIM}]{total_rx:.0f}/{total_tx:.0f} MB/s[/{PHOSPHOR_DIM}]"
        )
        visible = gpus[:max(1, state.hw_gpu_limit)]

        def gpu_line(gpu: GpuStats) -> str:
            vram_ratio_gpu = gpu.mem_used_mb / gpu.mem_total_mb if gpu.mem_total_mb > 0 else 0.0
            power_ratio_gpu = gpu.power_w / gpu.power_limit_w if gpu.power_limit_w > 0 else 0.0
            return (
                f"G{gpu.index} "
                f"[{PHOSPHOR}]{gpu.gpu_util_pct:.0f}[/{PHOSPHOR}]/"
                f"[{PHOSPHOR_DIM}]{gpu.mem_util_pct:.0f}[/{PHOSPHOR_DIM}] "
                f"v[{color_by_ratio(vram_ratio_gpu, 0.85, 0.95)}]{format_gib_from_mb(gpu.mem_used_mb)}[/] "
                f"w[{color_by_ratio(power_ratio_gpu, 0.65, 0.85)}]{gpu.power_w:.0f}[/] "
                f"t[{color_temp(gpu.temp_c)}]{gpu.temp_c:.0f}C[/]"
            )

        rows = []
        step = 2 if paired else 1
        for i in range(0, len(visible), step):
            left = gpu_line(visible[i])
            if paired:
                right = gpu_line(visible[i + 1]) if i + 1 < len(visible) else ""
                rows.append(f"{left}  [dim]|[/dim]  {right}" if right else left)
            else:
                rows.append(left)
        hidden = len(gpus) - len(visible)
        hidden_note = f"[dim]+{hidden} hidden[/dim]" if hidden > 0 else ""
        body = "\n".join(rows)
        content = Group(summary, body, hidden_note) if hidden_note else Group(summary, body)
    return Panel(
        content,
        title=render_title("HW"),
        title_align="left",
        box=PANEL_BOX,
        border_style=FRAME_BORDER,
        padding=(0, 1),
    )


def render_events_panel(state: TUIState, limit: int = 3) -> Panel:
    if not state.events:
        body = "[dim]No events yet[/dim]"
    else:
        body = "\n".join(f"[dim]{line}[/dim]" for line in state.events[-max(1, limit):])
    return Panel(
        body,
        title=render_title("EVENTS"),
        title_align="left",
        box=PANEL_BOX,
        border_style=SUBTLE_BORDER,
        padding=(0, 1),
    )


def render_live_stats_panel(state: TUIState) -> Panel:
    rows = []
    mode = "req-count" if state.benchmark_mode == "request-count" else "duration"
    rows.append(
        f"[dim]mode[/dim] {mode}  [dim]ctx[/dim] {format_context(state.current_context)}  "
        f"[dim]C[/dim] {state.current_concurrency}"
    )
    rows.append(
        f"[dim]live[/dim] [{PHOSPHOR}]{state.cell_live_tps:.1f}[/{PHOSPHOR}] tok/s  "
        f"[dim]samples[/dim] {state.cell_completed_requests}/{state.cell_request_samples}"
    )
    if state.cell_request_samples > 0:
        rows.append(
            f"TTFTp50 [{TEXT_PRIMARY}]{state.cell_ttft_p50_ms:.0f}ms[/{TEXT_PRIMARY}]  "
            f"ITLp50 [{TEXT_PRIMARY}]{state.cell_itl_p50_ms:.1f}ms[/{TEXT_PRIMARY}]"
        )
        latency = f"{state.cell_request_latency_p50_ms:.0f}/{state.cell_request_latency_p90_ms:.0f}ms"
        rows.append(
            f"lat p50/p90 [{TEXT_PRIMARY}]{latency}[/{TEXT_PRIMARY}]  "
            f"userp50 [{PHOSPHOR_DIM}]{state.cell_user_tps_p50:.1f}[/{PHOSPHOR_DIM}]"
        )
    else:
        rows.append("[dim]client latency: waiting for completed stream samples[/dim]")
    rows.append(
        f"[dim]server[/dim] run={state.srv_running_reqs} q={state.srv_queue_reqs} "
        f"kv={state.srv_utilization:.2%}"
    )
    if state.srv_spec_accept_rate > 0 or state.srv_spec_accept_length > 0:
        rows.append(
            f"[dim]spec[/dim] accept={state.srv_spec_accept_rate:.1%} "
            f"len={state.srv_spec_accept_length:.2f}"
        )
    if state.cell_tps_history:
        rows.append(render_speed_trace(state.cell_tps_history, width=16))
    return Panel(
        "\n".join(rows),
        title=render_title("STATS"),
        title_align="left",
        box=PANEL_BOX,
        border_style=SUBTLE_BORDER,
        padding=(0, 1),
    )


def render_compact_prefill_panel(state: TUIState, limit: int = 3) -> Panel:
    lines = []
    for ctx in state.prefill_contexts[:limit]:
        label = format_context(ctx)
        if ctx in state.prefill_results:
            pr = state.prefill_results[ctx]
            if pr.get("skipped"):
                value = "skip"
            else:
                value = f"{pr.get('tok_per_sec', 0):,.0f} tok/s"
        else:
            value = "..."
        lines.append(f"[{PHOSPHOR_SOFT}]{label:>4}[/{PHOSPHOR_SOFT}]  {value}")
    if not lines:
        lines.append("[dim]No prefill tests[/dim]")
    return Panel(
        "\n".join(lines),
        title=render_title("PREFILL"),
        title_align="left",
        box=PANEL_BOX,
        border_style=SUBTLE_BORDER,
        padding=(0, 1),
    )


def live_decode_panel_width(state: TUIState, prefill_visible: bool, detail_mode: str = "none") -> int:
    # Keep the decode matrix compact instead of stretching across the whole
    # dashboard. The cap preserves room for the vertical event log.
    mode, terminal_width, _ = live_layout_mode()
    col_width = live_decode_column_width(mode, detail_mode=detail_mode)
    desired = 20 + len(state.concurrency_levels) * (col_width + 1)
    min_width = 32
    prefill_width = 31 if prefill_visible and mode == "wide" else 0
    event_min_width = 18 if mode != "narrow" else 0
    available = terminal_width - prefill_width - event_min_width
    if available <= 0:
        return min_width
    return max(min_width, min(desired, available))


def live_layout_mode() -> tuple[str, int, int]:
    size = shutil.get_terminal_size((160, 40))
    width = size.columns
    if width >= 150:
        return "wide", width, size.lines
    if width >= 108:
        return "mid", width, size.lines
    return "narrow", width, size.lines


def live_decode_column_width(mode: str, detail_mode: str = "none") -> int:
    if detail_mode == "inline":
        return 17
    if detail_mode == "stacked":
        return 9
    if mode == "wide":
        return 9
    if mode == "mid":
        return 7
    return 5


def live_decode_detail_mode(
    state: TUIState,
    mode: str,
    term_width: int,
    term_height: int,
    middle_size: int,
    prefill_visible: bool,
) -> str:
    if mode == "narrow":
        return "none"
    # Do not key this only off "wide": a 140-column terminal can fit the
    # detailed decode matrix if prefill is stacked/compact instead of side-by-side.
    inline_col_width = live_decode_column_width(mode, detail_mode="inline")
    stacked_col_width = live_decode_column_width(mode, detail_mode="stacked")
    prefill_width = 31 if prefill_visible and mode == "wide" else 0
    event_min_width = 18
    available_width = term_width - prefill_width - event_min_width
    results_height = max(8, term_height - middle_size - 6)
    inline_needed_width = 20 + len(state.concurrency_levels) * (inline_col_width + 1)
    if available_width >= inline_needed_width and results_height >= len(state.context_lengths) + 7:
        return "inline"
    stacked_needed_width = 20 + len(state.concurrency_levels) * (stacked_col_width + 1)
    stacked_needed_height = len(state.context_lengths) * 2 + 9
    if available_width >= stacked_needed_width and results_height >= stacked_needed_height:
        return "stacked"
    return "none"


def compact_decode_cell(value: float, mode: str) -> str:
    if mode == "narrow":
        if abs(value) >= 1000:
            return f"{value / 1000:.2g}k"
        return f"{value:.0f}"
    if mode == "mid":
        if abs(value) >= 1000:
            return f"{value / 1000:.1f}k"
        return f"{value:.0f}"
    return f"{value:.1f}"


def compact_cell_ms(ms: float) -> str:
    if not ms or ms <= 0:
        return "—"
    if ms >= 10000:
        return f"{ms / 1000:.0f}k"
    if ms >= 1000:
        return f"{ms / 1000:.1g}k"
    return f"{ms:.0f}"


def compact_cell_rate(value: float) -> str:
    if not value or value <= 0:
        return "—"
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return f"{value:.0f}"


def compact_client_cell_detail(info: dict) -> str:
    if not info or info.get("request_count", 0) <= 0:
        return ""
    return (
        f"{compact_cell_ms(info.get('ttft_ms', 0))}/"
        f"{compact_cell_ms(info.get('itl_ms', 0))}"
    )


def narrow_decode_panel_size(state: TUIState) -> int:
    # Header/title/border overhead plus one row per context. Keep the live
    # decode matrix compact so spare vertical space goes to EVENTS instead of
    # empty panel body.
    return min(12, max(7, len(state.context_lengths) + 6))


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
    if not name:
        return 0.0
    for key, val in metrics.items():
        if key.startswith(name):
            if label_filter and label_filter not in key:
                continue
            return val
    return 0.0


def sum_metric(metrics: dict, name: str, label_filter: str = "") -> float:
    if not name:
        return 0.0
    total = 0.0
    for key, val in metrics.items():
        if not key.startswith(name):
            continue
        if label_filter and label_filter not in key:
            continue
        total += val
    return total


def has_metric(metrics: dict, name: str, label_filter: str = "") -> bool:
    if not name:
        return False
    for key in metrics:
        if not key.startswith(name):
            continue
        if label_filter and label_filter not in key:
            continue
        return True
    return False


def extract_label(metrics: dict, metric_name: str, label: str) -> str:
    """Extract a label value from a labeled Prometheus metric."""
    for key in metrics:
        if key.startswith(metric_name):
            m = re.search(rf'{label}="([^"]*)"', key)
            if m:
                return m.group(1)
    return ""


def _parse_metric_int(metrics: dict, metric_name: str, label: str) -> int:
    value = extract_label(metrics, metric_name, label)
    if not value:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


def extract_vllm_cp_size(metrics: dict) -> tuple[int, str]:
    """Extract CP multiplier if vLLM exports parallel config in metrics."""
    for metric_name in (
        "vllm:parallel_config_info",
        "vllm:vllm_config_info",
        "vllm:cache_config_info",
    ):
        cp_world = _parse_metric_int(metrics, metric_name, "cp_world_size")
        if cp_world > 0:
            return cp_world, f"{metric_name}.cp_world_size"

        pcp_size = _parse_metric_int(
            metrics, metric_name, "prefill_context_parallel_size"
        )
        dcp_size = _parse_metric_int(
            metrics, metric_name, "decode_context_parallel_size"
        )
        if pcp_size > 0 and dcp_size > 0:
            return (
                pcp_size * dcp_size,
                (
                    f"{metric_name}.prefill_context_parallel_size"
                    f"*decode_context_parallel_size"
                ),
            )

    candidates = (
        ("vllm:parallel_config_info", "dcp_world_size"),
        ("vllm:vllm_config_info", "dcp_world_size"),
        # Future-proof fallback if this gets added to cache_config_info.
        ("vllm:cache_config_info", "dcp_world_size"),
    )
    for metric_name, label in candidates:
        parsed = _parse_metric_int(metrics, metric_name, label)
        if parsed > 0:
            return parsed, f"{metric_name}.{label}"
    return 0, ""


def infer_local_vllm_dcp_size(base_url: str) -> int:
    """Infer local vLLM DCP size from the server process command line."""
    try:
        parsed = urlparse(base_url)
    except Exception:
        return 0
    host = parsed.hostname or ""
    port = parsed.port
    if host not in ("localhost", "127.0.0.1", "0.0.0.0", "::1") or not port:
        return 0

    try:
        proc = subprocess.run(
            ["ps", "-eo", "args="],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return 0

    port_re = re.compile(rf"--port(?:=|\s+){port}\b")
    dcp_re = re.compile(r"--decode-context-parallel-size(?:=|\s+)(\d+)\b")
    for cmd in proc.stdout.splitlines():
        if "vllm" not in cmd or not port_re.search(cmd):
            continue
        match = dcp_re.search(cmd)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return 0
    return 0


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
            "prompt_tokens_total": "sglang:prompt_tokens_total",
            "request_success_total": "sglang:num_requests_total",
            "prefill_time_count": "sglang:per_stage_req_latency_seconds_count",
            "prefill_time_sum": "sglang:per_stage_req_latency_seconds_sum",
        },
        ENGINE_VLLM: {
            "gen_throughput": "vllm:avg_generation_throughput_toks_per_s",
            "running_reqs": "vllm:num_requests_running",
            "queue_reqs": "vllm:num_requests_waiting",
            "utilization": "vllm:kv_cache_usage_perc",
            "spec_accept_rate": "",
            "spec_accept_length": "",
            "spec_drafts_total": "vllm:spec_decode_num_drafts_total",
            "spec_draft_tokens_total": "vllm:spec_decode_num_draft_tokens_total",
            "spec_accepted_tokens_total": "vllm:spec_decode_num_accepted_tokens_total",
            "gen_tokens_total": "vllm:generation_tokens_total",
            "prompt_tokens_total": "vllm:prompt_tokens_total",
            "request_success_total": "vllm:request_success_total",
            "prefill_time_count": "vllm:request_prefill_time_seconds_count",
            "prefill_time_sum": "vllm:request_prefill_time_seconds_sum",
        },
        ENGINE_OPENAI_PROXY: {},
    }
    return names.get(engine, {}).get(key, "")


def prefill_counter_snapshot(metrics: dict, engine: str) -> dict:
    """Return counters needed for exact server-side prefill measurement."""
    label_filter = 'stage="prefill_forward"' if engine == ENGINE_SGLANG else ""
    return {
        "prompt_tokens_total": sum_metric(metrics, metric_name(engine, "prompt_tokens_total")),
        "request_success_total": sum_metric(metrics, metric_name(engine, "request_success_total")),
        "prefill_count": sum_metric(metrics, metric_name(engine, "prefill_time_count"), label_filter),
        "prefill_sum": sum_metric(metrics, metric_name(engine, "prefill_time_sum"), label_filter),
    }


def counter_delta(after: dict, before: dict) -> dict:
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after.keys() | before.keys()}


async def wait_server_idle(
    client: httpx.AsyncClient,
    base_url: str,
    engine: str,
    stable_seconds: float = 1.0,
    timeout_seconds: float = 120.0,
    state: TUIState | None = None,
    live: object | None = None,
    status: str = "",
) -> dict:
    """Wait until server reports no running or queued requests, then return metrics."""
    if state is not None and not state.metrics_available:
        if status:
            state.prefill_status = f"{status} (metrics unavailable; using client timing)"
            if live is not None:
                live.update(build_display(state))
        return {}
    deadline = time.monotonic() + timeout_seconds
    stable_since = None
    last_metrics = {}
    while time.monotonic() < deadline:
        last_metrics = await scrape_metrics(client, base_url)
        running = extract_metric(last_metrics, metric_name(engine, "running_reqs"))
        waiting = extract_metric(last_metrics, metric_name(engine, "queue_reqs"))
        if state is not None:
            state.srv_running_reqs = int(running)
            state.srv_queue_reqs = int(waiting)
            state.srv_utilization = extract_metric(last_metrics, metric_name(engine, "utilization"))
            state.srv_gen_throughput = extract_metric(last_metrics, metric_name(engine, "gen_throughput"))
            state.srv_spec_accept_rate = extract_metric(last_metrics, metric_name(engine, "spec_accept_rate"))
            state.srv_spec_accept_length = extract_metric(last_metrics, metric_name(engine, "spec_accept_length"))
            if status:
                state.prefill_status = status
            if live is not None:
                live.update(build_display(state))
        if running == 0 and waiting == 0:
            if stable_since is None:
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= stable_seconds:
                return last_metrics
        else:
            stable_since = None
        await asyncio.sleep(0.2)
    return last_metrics


async def wait_prefill_task_with_live(
    request_task: asyncio.Task,
    client: httpx.AsyncClient,
    base_url: str,
    engine: str,
    state: TUIState,
    live: object,
    status: str,
):
    """Refresh the live dashboard while a long prefill request is in flight."""
    while not request_task.done():
        state.prefill_status = status
        if state.metrics_available:
            metrics = await scrape_metrics(client, base_url)
            state.srv_running_reqs = int(extract_metric(metrics, metric_name(engine, "running_reqs")))
            state.srv_queue_reqs = int(extract_metric(metrics, metric_name(engine, "queue_reqs")))
            state.srv_utilization = extract_metric(metrics, metric_name(engine, "utilization"))
            state.srv_gen_throughput = extract_metric(metrics, metric_name(engine, "gen_throughput"))
            state.srv_spec_accept_rate = extract_metric(metrics, metric_name(engine, "spec_accept_rate"))
            state.srv_spec_accept_length = extract_metric(metrics, metric_name(engine, "spec_accept_length"))
        live.update(build_display(state))
        await asyncio.sleep(0.5)
    return await request_task


class NullLive:
    """Drop-in replacement for Rich Live when --display-mode=plain is used."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, *_args, **_kwargs):
        return None


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
    shared_request_samples: list = None,
    shared_started_count: list = None,
    shared_completed_count: list = None,
    target_request_count: int = 0,
    shared_usage_token_count: list = None,
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
        if target_request_count > 0 and shared_started_count is not None:
            if shared_started_count[0] >= target_request_count:
                break
            # The asyncio event loop runs this synchronously until the next
            # await, so this is enough to hand out exact request slots.
            shared_started_count[0] += 1
        usage_tokens = None
        prompt_tokens = 0
        req_start = time.monotonic()
        req_first = None
        req_second = None
        req_prev_token = None
        req_gaps = []
        req_chunks = 0
        req_completed = False
        req_last_usage_tokens = 0
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
                        req_completed = True
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Check for usage in final chunk (stream_options.include_usage)
                    usage = data.get("usage")
                    if usage and "completion_tokens" in usage:
                        usage_tokens = int(usage["completion_tokens"])
                        if shared_usage_token_count is not None:
                            usage_delta = max(0, usage_tokens - req_last_usage_tokens)
                            if usage_delta > 0:
                                shared_usage_token_count[0] += usage_delta
                            req_last_usage_tokens = max(req_last_usage_tokens, usage_tokens)
                    if usage and "prompt_tokens" in usage:
                        prompt_tokens = usage["prompt_tokens"]

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
                        token_time = time.monotonic()
                        if t_first is None:
                            t_first = token_time
                        if req_first is None:
                            req_first = token_time
                        elif req_second is None:
                            req_second = token_time
                        if req_prev_token is not None:
                            req_gaps.append(token_time - req_prev_token)
                        req_prev_token = token_time
                        if not active_signaled and shared_active_streams is not None:
                            shared_active_streams[0] += 1
                            active_signaled = True
                        total_chunks += 1
                        req_chunks += 1
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

        req_output_tokens = usage_tokens if usage_tokens is not None else req_chunks
        if req_first is not None and req_output_tokens > 0:
            # ITL can be computed from the observed generated segment even when
            # a sustained-duration cell cancels the stream at the end of the
            # measurement window. Do not use cancel/HTTP close time as the last
            # token; only the timestamp of the last received content chunk is
            # valid. Full request latency remains completed-stream-only.
            req_latency = 0.0
            req_itl = 0.0
            req_output_tps = 0.0
            req_e2e_output_tps = 0.0
            if req_prev_token is not None and req_output_tokens >= 2 and req_prev_token > req_first:
                req_itl = (req_prev_token - req_first) / (req_output_tokens - 1)
                if req_itl > 0:
                    req_output_tps = 1.0 / req_itl
            if req_completed and req_prev_token is not None:
                req_latency = req_prev_token - req_start
                if req_latency > 0:
                    req_e2e_output_tps = req_output_tokens / req_latency
            req_chunk_itl = mean(req_gaps) if req_gaps else 0.0
            sample = RequestSample(
                ttft=req_first - req_start,
                time_to_second_token=(req_second - req_first) if req_second else 0.0,
                latency=req_latency,
                inter_token_latency_avg=req_itl,
                chunk_inter_token_latency_avg=req_chunk_itl,
                input_tokens=int(prompt_tokens or 0),
                output_tokens=int(req_output_tokens),
                output_tps_per_user=req_output_tps,
                e2e_output_tps_per_user=req_e2e_output_tps,
                completed=req_completed,
            )
            result.request_samples.append(sample)
            if shared_request_samples is not None:
                shared_request_samples.append(sample)
            if req_completed and shared_completed_count is not None:
                shared_completed_count[0] += 1

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
    ignore_eos: bool = True,
    request_count: int = 0,
    warmup_request_count: int = 0,
) -> CellResult:
    messages = build_messages(context_tokens, context_text)
    stream_options = {"include_usage": True}
    if request_count <= 0:
        # Duration-based Sustained Decode needs cumulative usage on every
        # streamed chunk so it can measure exactly inside a time window.
        # Finite request-count Burst / E2E only needs the final usage chunk;
        # requesting continuous usage there adds stream overhead and diverges
        # from AIPerf-style requests.
        stream_options["continuous_usage_stats"] = True

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "stream_options": stream_options,
    }
    if ignore_eos:
        payload["ignore_eos"] = True

    url = f"{base_url}/v1/chat/completions"
    cancel_event = asyncio.Event()
    shared_token_count = [0]
    shared_usage_token_count = [0]
    shared_active_streams = [0]  # how many streams have received first token
    shared_request_samples = []

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
    state.benchmark_mode = "request-count" if request_count > 0 else "duration"
    state.request_count_target = request_count
    state.cell_tokens = 0
    state.cell_live_tps = 0.0
    state.cell_tps_history = []
    state.cell_running = True
    state.cell_warmup = True
    state.cell_request_samples = 0
    state.cell_completed_requests = 0
    state.cell_ttft_p50_ms = 0.0
    state.cell_itl_p50_ms = 0.0
    state.cell_user_tps_p50 = 0.0
    state.cell_request_latency_p50_ms = 0.0
    state.cell_request_latency_p90_ms = 0.0
    hw_cell_start_idx = len(state.hw_history)
    hw_measurement_start_idx = hw_cell_start_idx
    add_event(state, f"cell start C={concurrency} ctx={format_context(context_tokens)}")

    # Scout request: ensure prefix cache is warm before launching full concurrency.
    # Send one request with max_tokens=1 to populate/refresh prefix cache,
    # then all C requests will get cache hits instead of competing for prefill.
    if context_tokens > 0:
        scout_payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": 1,
            "stream_options": {"include_usage": True},
        }
        if ignore_eos:
            scout_payload["ignore_eos"] = True
        should_record_prefill = (
            context_tokens in state.prefill_contexts
            and context_tokens not in state.prefill_results
        )
        old_prefill_phase = state.prefill_phase
        old_prefill_status = state.prefill_status
        if should_record_prefill:
            state.prefill_phase = True
            state.prefill_status = "decode scout prefill: populating prefix cache"
            state.prefill_method = "integrated"
            state.prefill_last_tps = 0.0
            state.prefill_last_tokens = 0
            state.prefill_last_seconds = 0.0
            add_event(state, f"integrated prefill start ctx={format_context(context_tokens)}")
            live.update(build_display(state))
        async def run_scout_request():
            scout_t0 = time.monotonic()
            scout_ttft = None
            scout_prompt_tokens = None
            scout_ok = False
            try:
                async with client.stream("POST", url, json=scout_payload,
                                         timeout=httpx.Timeout(600.0, connect=30.0)) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            scout_ok = True
                            break
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        usage = data.get("usage")
                        if usage and "prompt_tokens" in usage:
                            scout_prompt_tokens = usage["prompt_tokens"]
                        if scout_ttft is None and "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                                scout_ttft = time.monotonic() - scout_t0
                    if scout_prompt_tokens is not None or scout_ttft is not None:
                        scout_ok = True
                return scout_ttft, scout_prompt_tokens, scout_ok, None
            except Exception as exc:
                return scout_ttft, scout_prompt_tokens, scout_ok, exc

        scout_wall_start = time.monotonic()
        scout_task = asyncio.create_task(run_scout_request())
        scout_ttft, scout_prompt_tokens, scout_ok, scout_error = await wait_prefill_task_with_live(
            scout_task,
            client,
            base_url,
            engine,
            state,
            live,
            "decode scout prefill: waiting for first token",
        )
        if should_record_prefill and scout_error is not None:
            add_event(
                state,
                f"integrated prefill failed ctx={format_context(context_tokens)}: {type(scout_error).__name__}",
        )
        if should_record_prefill and scout_ok:
            if scout_ttft is None:
                scout_ttft = time.monotonic() - scout_wall_start
            prompt_tokens = int(scout_prompt_tokens or context_tokens)
            tok_per_sec = (prompt_tokens / scout_ttft) if scout_ttft > 0 else 0.0
            state.prefill_results[context_tokens] = {
                "method": "integrated_scout",
                "ttft": scout_ttft,
                "prefill_time": scout_ttft,
                "tok_per_sec": tok_per_sec,
                "prompt_tokens": prompt_tokens,
                "samples": 1,
                "server_tok_per_sec": 0.0,
                "server_prefill_time": 0.0,
                "server_prompt_tokens": 0,
                "server_samples": 0,
                "server_method": "",
                "server_invalid_reason": "",
            }
            snapshot_partial_prefill(state)
            state.prefill_samples_done = 1
            state.prefill_last_tps = tok_per_sec
            state.prefill_last_tokens = prompt_tokens
            state.prefill_last_seconds = scout_ttft
            state.prefill_status = "decode scout prefill complete"
            add_event(
                state,
                f"integrated prefill done ctx={format_context(context_tokens)} {tok_per_sec:,.0f} tok/s",
            )
            live.update(build_display(state))
        if should_record_prefill:
            state.prefill_phase = old_prefill_phase
            state.prefill_status = old_prefill_status
        live.update(build_display(state))

    metrics_interval = 1.0

    if request_count > 0:
        async def run_fixed_request_batch(target_count: int, record_samples: bool):
            batch_cancel = asyncio.Event()
            batch_tokens = [0]
            batch_active = [0]
            batch_samples = []
            batch_started = [0]
            batch_completed = [0]
            workers = max(1, min(concurrency, target_count))
            batch_tasks = [
                asyncio.create_task(
                    stream_one_request(
                        cell_client,
                        url,
                        payload,
                        i,
                        batch_cancel,
                        batch_tokens,
                        batch_active,
                        batch_samples,
                        batch_started,
                        batch_completed,
                        target_count,
                    )
                )
                for i in range(workers)
            ]
            while not all(t.done() for t in batch_tasks):
                if _skip_event.is_set():
                    _skip_event.clear()
                    batch_cancel.set()
                    break
                if record_samples:
                    update_state_request_stats(state, batch_samples)
                    state.cell_tokens = batch_tokens[0]
                    state._active_streams = batch_active[0]
                    elapsed = time.monotonic() - state.cell_measurement_start
                    if elapsed > 0.5 and batch_tokens[0] > 0:
                        state.cell_live_tps = batch_tokens[0] / elapsed
                    live.update(build_display(state))
                await asyncio.sleep(0.25)
            done, pending = await asyncio.wait(batch_tasks, timeout=30.0)
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.wait(pending, timeout=5.0)
            stream_results = []
            for t in batch_tasks:
                try:
                    stream_results.append(t.result())
                except (asyncio.CancelledError, Exception):
                    stream_results.append(StreamResult(error="cancelled"))
            samples = []
            for stream_result in stream_results:
                samples.extend(stream_result.request_samples)
            return stream_results, samples, batch_tokens[0], batch_started[0], batch_completed[0]

        # Request-count burst mode: optional warmup requests are
        # discarded, then exactly N profiling requests are sent and all are
        # allowed to complete. Aggregate throughput is completed output tokens
        # divided by the profiling wall time.
        if warmup_request_count > 0:
            state.cell_warmup = True
            state.cell_measurement_start = 0.0
            state.request_count_target = warmup_request_count
            live.update(build_display(state))
            await run_fixed_request_batch(warmup_request_count, record_samples=False)

        state.cell_warmup = False
        state.request_count_target = request_count
        state.cell_tokens = 0
        state.cell_live_tps = 0.0
        state.cell_tps_history = []
        state.cell_request_samples = 0
        state.cell_completed_requests = 0
        state.cell_ttft_p50_ms = 0.0
        state.cell_itl_p50_ms = 0.0
        state.cell_user_tps_p50 = 0.0
        state.cell_request_latency_p50_ms = 0.0
        state.cell_request_latency_p90_ms = 0.0

        start_metrics = await scrape_metrics(client, base_url) if state.metrics_available else {}
        measurement_gen_tokens_start = (
            extract_metric(start_metrics, metric_name(engine, "gen_tokens_total"))
            if engine == ENGINE_VLLM else None
        )
        measurement_start = time.monotonic()
        state.cell_measurement_start = measurement_start
        state.cell_start = measurement_start
        hw_measurement_start_idx = len(state.hw_history)
        add_event(state, f"measure start C={concurrency} ctx={format_context(context_tokens)} request-count={request_count}")
        gen_throughput_samples = []
        running_reqs_samples = []
        queue_reqs_samples = []
        last_metrics_time = 0.0
        prev_gen_tokens = measurement_gen_tokens_start
        prev_gen_time = measurement_start
        prev_spec_drafts = None
        prev_spec_draft_tokens = None
        prev_spec_accepted_tokens = None

        batch_cancel = asyncio.Event()
        batch_tokens = [0]
        batch_active = [0]
        batch_samples = []
        batch_started = [0]
        batch_completed = [0]
        workers = max(1, min(concurrency, request_count))
        tasks = [
            asyncio.create_task(
                stream_one_request(
                    cell_client,
                    url,
                    payload,
                    i,
                    batch_cancel,
                    batch_tokens,
                    batch_active,
                    batch_samples,
                    batch_started,
                    batch_completed,
                    request_count,
                )
            )
            for i in range(workers)
        ]

        while not all(t.done() for t in tasks):
            await asyncio.sleep(0.25)
            now = time.monotonic()
            state.cell_tokens = batch_tokens[0]
            state._active_streams = batch_active[0]
            update_state_request_stats(state, batch_samples)
            elapsed = now - measurement_start
            if elapsed > 0.5 and batch_tokens[0] > 0:
                state.cell_live_tps = batch_tokens[0] / elapsed

            if now - last_metrics_time > metrics_interval:
                metrics = await scrape_metrics(client, base_url) if state.metrics_available else {}
                if engine == ENGINE_SGLANG:
                    state.srv_gen_throughput = extract_metric(metrics, metric_name(engine, "gen_throughput"))
                else:
                    tp = extract_metric(metrics, metric_name(engine, "gen_throughput"))
                    if tp > 0:
                        state.srv_gen_throughput = tp
                    else:
                        gen_total = extract_metric(metrics, metric_name(engine, "gen_tokens_total"))
                        if gen_total > 0 and prev_gen_tokens is not None:
                            dt = now - prev_gen_time
                            if dt > 0.1:
                                state.srv_gen_throughput = (gen_total - prev_gen_tokens) / dt
                        prev_gen_tokens = gen_total
                        prev_gen_time = now

                if state.metrics_available:
                    state.srv_running_reqs = int(extract_metric(metrics, metric_name(engine, "running_reqs")))
                    state.srv_queue_reqs = int(extract_metric(metrics, metric_name(engine, "queue_reqs")))
                    state.srv_utilization = extract_metric(metrics, metric_name(engine, "utilization"))
                if engine == ENGINE_SGLANG:
                    state.srv_spec_accept_rate = extract_metric(metrics, metric_name(engine, "spec_accept_rate"))
                    state.srv_spec_accept_length = extract_metric(metrics, metric_name(engine, "spec_accept_length"))
                elif engine == ENGINE_VLLM:
                    drafts_total = extract_metric(metrics, metric_name(engine, "spec_drafts_total"))
                    draft_tokens_total = extract_metric(metrics, metric_name(engine, "spec_draft_tokens_total"))
                    accepted_tokens_total = extract_metric(metrics, metric_name(engine, "spec_accepted_tokens_total"))
                    if prev_spec_draft_tokens is not None:
                        dd = draft_tokens_total - prev_spec_draft_tokens
                        da = accepted_tokens_total - prev_spec_accepted_tokens
                        dn = drafts_total - prev_spec_drafts
                        if dd > 0:
                            state.srv_spec_accept_rate = max(0.0, min(1.0, da / dd))
                        if dn > 0:
                            state.srv_spec_accept_length = max(0.0, da / dn)
                    prev_spec_drafts = drafts_total
                    prev_spec_draft_tokens = draft_tokens_total
                    prev_spec_accepted_tokens = accepted_tokens_total

                if state.srv_gen_throughput > 0:
                    gen_throughput_samples.append(state.srv_gen_throughput)
                if state.metrics_available:
                    running_reqs_samples.append(state.srv_running_reqs)
                    queue_reqs_samples.append(state.srv_queue_reqs)
                if state.cell_live_tps > 0:
                    state.cell_tps_history.append(state.cell_live_tps)
                    if len(state.cell_tps_history) > 240:
                        state.cell_tps_history = state.cell_tps_history[-240:]
                last_metrics_time = now

            live.update(build_display(state))
            if _skip_event.is_set():
                _skip_event.clear()
                batch_cancel.set()
                await cell_client.aclose()
                return CellResult(
                    concurrency=concurrency,
                    context_tokens=context_tokens,
                    benchmark_mode="request-count",
                    aggregate_tps=-2,
                    hardware_summary=summarize_hardware_history(state.hw_history[hw_measurement_start_idx:]),
                )

        done, pending = await asyncio.wait(tasks, timeout=30.0)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.wait(pending, timeout=5.0)
        measurement_end = time.monotonic()

        stream_results = []
        for t in tasks:
            try:
                stream_results.append(t.result())
            except (asyncio.CancelledError, Exception):
                stream_results.append(StreamResult(error="cancelled"))
        request_samples = []
        for stream_result in stream_results:
            request_samples.extend(stream_result.request_samples)
        update_state_request_stats(state, request_samples)

        metrics = await scrape_metrics(client, base_url) if state.metrics_available else {}
        exact_server_tokens = 0
        if engine == ENGINE_VLLM and measurement_gen_tokens_start is not None:
            measurement_gen_tokens_end = extract_metric(metrics, metric_name(engine, "gen_tokens_total"))
            exact_server_tokens = max(
                0,
                int(round(measurement_gen_tokens_end - measurement_gen_tokens_start)),
            )
        measurement_seconds = max(0.0, measurement_end - measurement_start)
        completed_samples = [s for s in request_samples if s.completed]
        client_output_tokens = sum(s.output_tokens for s in completed_samples)
        aggregate_tps = (
            client_output_tokens / measurement_seconds
            if measurement_seconds > 0 and client_output_tokens > 0
            else 0.0
        )
        server_gen_throughput = (
            exact_server_tokens / measurement_seconds
            if measurement_seconds > 0 and exact_server_tokens > 0
            else (median(gen_throughput_samples) if gen_throughput_samples else 0.0)
        )

        request_summary = summarize_request_samples(request_samples)
        ttft_stats = request_summary["ttft"]
        ttst_stats = request_summary["time_to_second_token"]
        request_latency_stats = request_summary["request_latency"]
        itl_stats = request_summary["inter_token_latency"]
        chunk_itl_stats = request_summary["chunk_inter_token_latency"]
        user_tps_stats = request_summary["output_tps_per_user"]
        e2e_user_tps_stats = request_summary["e2e_output_tps_per_user"]
        input_seq_stats = request_summary["input_seq_len"]
        output_seq_stats = request_summary["output_seq_len"]

        avg_running = mean(running_reqs_samples) if running_reqs_samples else 0.0
        max_running = max(running_reqs_samples) if running_reqs_samples else 0
        avg_queue = mean(queue_reqs_samples) if queue_reqs_samples else 0.0
        max_queue = max(queue_reqs_samples) if queue_reqs_samples else 0
        queued_count = sum(1 for q in queue_reqs_samples if q > 0)
        queue_frac = queued_count / len(queue_reqs_samples) if queue_reqs_samples else 0.0
        capacity_limited = avg_queue > 0 or queue_frac > 0
        num_errors = sum(1 for r in stream_results if r.error)

        cell = CellResult(
            concurrency=concurrency,
            context_tokens=context_tokens,
            benchmark_mode="request-count",
            request_count_target=request_count,
            warmup_request_count=warmup_request_count,
            measurement_seconds=round(measurement_seconds, 6),
            client_output_tokens=client_output_tokens,
            server_output_tokens=exact_server_tokens,
            aggregate_source="openai_completed_usage",
            aggregate_tps=aggregate_tps,
            per_request_avg_tps=aggregate_tps / concurrency if concurrency > 0 else 0.0,
            ttft_avg=ttft_stats["avg"],
            ttft_p50=ttft_stats["p50"],
            ttft_p90=ttft_stats["p90"],
            ttft_p99=ttft_stats["p99"],
            time_to_second_token_avg=ttst_stats["avg"],
            time_to_second_token_p50=ttst_stats["p50"],
            time_to_second_token_p90=ttst_stats["p90"],
            time_to_second_token_p99=ttst_stats["p99"],
            request_latency_avg=request_latency_stats["avg"],
            request_latency_p50=request_latency_stats["p50"],
            request_latency_p90=request_latency_stats["p90"],
            request_latency_p99=request_latency_stats["p99"],
            inter_token_latency_avg=itl_stats["avg"],
            inter_token_latency_p50=itl_stats["p50"],
            inter_token_latency_p90=itl_stats["p90"],
            inter_token_latency_p99=itl_stats["p99"],
            output_tps_per_user_avg=user_tps_stats["avg"],
            output_tps_per_user_p50=user_tps_stats["p50"],
            output_tps_per_user_p90=user_tps_stats["p90"],
            output_tps_per_user_p99=user_tps_stats["p99"],
            e2e_output_tps_per_user_avg=e2e_user_tps_stats["avg"],
            e2e_output_tps_per_user_p50=e2e_user_tps_stats["p50"],
            e2e_output_tps_per_user_p90=e2e_user_tps_stats["p90"],
            e2e_output_tps_per_user_p99=e2e_user_tps_stats["p99"],
            chunk_inter_token_latency_avg=chunk_itl_stats["avg"],
            chunk_inter_token_latency_p50=chunk_itl_stats["p50"],
            chunk_inter_token_latency_p90=chunk_itl_stats["p90"],
            chunk_inter_token_latency_p99=chunk_itl_stats["p99"],
            input_seq_len_avg=input_seq_stats["avg"],
            output_seq_len_avg=output_seq_stats["avg"],
            output_seq_len_p50=output_seq_stats["p50"],
            output_seq_len_p90=output_seq_stats["p90"],
            output_seq_len_p99=output_seq_stats["p99"],
            request_count=request_summary["request_count"],
            completed_request_count=request_summary["completed_request_count"],
            request_samples=[asdict(s) for s in request_samples],
            total_tokens=client_output_tokens,
            wall_time=measurement_seconds,
            num_completed=len(completed_samples),
            num_errors=num_errors,
            server_gen_throughput=server_gen_throughput,
            server_utilization=extract_metric(metrics, metric_name(engine, "utilization")),
            server_spec_accept_rate=state.srv_spec_accept_rate,
            server_spec_accept_length=state.srv_spec_accept_length,
            avg_running_reqs=round(avg_running, 1),
            max_running_reqs=max_running,
            effective_concurrency=round(avg_running, 1),
            avg_queue_reqs=round(avg_queue, 1),
            max_queue_reqs=max_queue,
            queue_fraction=round(queue_frac, 3),
            underfilled=False,
            warmup_timed_out=False,
            warmup_duration=0.0,
            ready_reason=f"request_count={request_count}, warmup_request_count={warmup_request_count}",
            timeout_reason="",
            capacity_limited=capacity_limited,
            hardware_summary=summarize_hardware_history(state.hw_history[hw_measurement_start_idx:]),
        )

        state.cell_running = False
        state.results[(context_tokens, concurrency)] = cell.aggregate_tps
        state.errors[(context_tokens, concurrency)] = cell.num_errors
        state.queue_info[(context_tokens, concurrency)] = (
            cell.avg_running_reqs,
            cell.avg_queue_reqs,
            cell.capacity_limited,
        )
        state.client_info[(context_tokens, concurrency)] = compact_client_info_from_cell(cell)
        add_event(state, f"cell done C={concurrency} ctx={format_context(context_tokens)} {cell.aggregate_tps:.1f} tok/s")
        await cell_client.aclose()
        return cell

    # Launch all streams on fresh client (no stale keepalive connections)
    tasks = [
        asyncio.create_task(
            stream_one_request(cell_client, url, payload, i, cancel_event,
                               shared_token_count, shared_active_streams,
                               shared_request_samples,
                               shared_usage_token_count=shared_usage_token_count)
        )
        for i in range(concurrency)
    ]

    # Monitor loop — collect server gen_throughput samples for accurate measurement
    metrics_interval = 1.0
    min_warmup_seconds = 2.0     # minimum warmup (CUDA graph etc.)
    ready_stable_seconds = 3.0   # require sustained scheduler state
    max_warmup_seconds = 60.0    # give up waiting for full concurrency
    last_metrics_time = 0.0
    gen_throughput_samples = []
    # For vLLM: compute throughput rate from generation_tokens counter
    prev_gen_tokens = None
    prev_gen_time = None
    prev_spec_drafts = None
    prev_spec_draft_tokens = None
    prev_spec_accepted_tokens = None
    # Queue tracking: collect running/queue samples after warmup
    running_reqs_samples = []
    queue_reqs_samples = []
    # Dynamic warmup: wait until all requests are in decode (queue == 0)
    warmup_done = False
    warmup_stable_since = None  # time when queue first hit 0 after min_warmup
    warmup_timed_out = False
    warmup_duration = 0.0
    ready_reason = ""
    timeout_reason = ""
    measurement_start = None    # reset timer after warmup for full duration measurement
    measurement_end = None
    measurement_tokens_start = 0  # streamed content chunk count at measurement start
    measurement_usage_tokens_start = 0  # OpenAI usage completion_tokens at measurement start
    measurement_gen_tokens_start = None  # vLLM generation_tokens counter at measurement start
    measurement_gen_tokens_end = None

    while True:
        await asyncio.sleep(0.5)
        now = time.monotonic()
        elapsed = now - state.cell_start

        # Update token counts from client-side estimate (for TUI only)
        state.cell_tokens = shared_token_count[0]
        state._active_streams = shared_active_streams[0]

        # Scrape server metrics periodically
        if now - last_metrics_time > metrics_interval:
            metrics = await scrape_metrics(client, base_url) if state.metrics_available else {}

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

            if state.metrics_available:
                state.srv_running_reqs = int(extract_metric(metrics, metric_name(engine, "running_reqs")))
                state.srv_queue_reqs = int(extract_metric(metrics, metric_name(engine, "queue_reqs")))
                state.srv_utilization = extract_metric(metrics, metric_name(engine, "utilization"))
            if engine == ENGINE_SGLANG:
                state.srv_spec_accept_rate = extract_metric(metrics, metric_name(engine, "spec_accept_rate"))
                state.srv_spec_accept_length = extract_metric(metrics, metric_name(engine, "spec_accept_length"))
            elif engine == ENGINE_VLLM:
                drafts_total = extract_metric(metrics, metric_name(engine, "spec_drafts_total"))
                draft_tokens_total = extract_metric(metrics, metric_name(engine, "spec_draft_tokens_total"))
                accepted_tokens_total = extract_metric(metrics, metric_name(engine, "spec_accepted_tokens_total"))
                if prev_spec_draft_tokens is not None:
                    dd = draft_tokens_total - prev_spec_draft_tokens
                    da = accepted_tokens_total - prev_spec_accepted_tokens
                    dn = drafts_total - prev_spec_drafts
                    if dd > 0:
                        state.srv_spec_accept_rate = max(0.0, min(1.0, da / dd))
                    if dn > 0:
                        state.srv_spec_accept_length = max(0.0, da / dn)
                prev_spec_drafts = drafts_total
                prev_spec_draft_tokens = draft_tokens_total
                prev_spec_accepted_tokens = accepted_tokens_total

            # Dynamic warmup: wait for min_warmup AND queue==0 AND full
            # requested concurrency running. Do not use "every stream has ever
            # seen a token" as the primary readiness signal: in over-capacity
            # cells, streams can receive tokens in waves without ever running
            # concurrently.
            if not warmup_done:
                if elapsed >= min_warmup_seconds:
                    all_streams_active = shared_active_streams[0] >= concurrency
                    generating = state.srv_gen_throughput > 0 or shared_token_count[0] > 0
                    full_concurrency_running = state.srv_running_reqs >= concurrency
                    # With short max_tokens, requests can finish between 1s
                    # Prometheus scrapes, so running_reqs may never stay at C
                    # for ready_stable_seconds. In that mode, completed request
                    # flow is a better readiness signal; effective concurrency
                    # is still checked after measurement from scheduler samples.
                    short_request_flow = (
                        max_tokens <= 512
                        and len(shared_request_samples) >= concurrency
                        and state.srv_queue_reqs == 0
                        and generating
                    )
                    if engine not in (ENGINE_SGLANG, ENGINE_VLLM):
                        full_concurrency_running = all_streams_active
                    if not state.metrics_available:
                        full_concurrency_running = all_streams_active
                        server_ready = generating and full_concurrency_running
                    else:
                        server_ready = (
                            state.srv_queue_reqs == 0
                            and generating
                            and (full_concurrency_running or short_request_flow)
                        )
                    if server_ready:
                        if warmup_stable_since is None:
                            warmup_stable_since = now
                        elif now - warmup_stable_since >= ready_stable_seconds:
                            warmup_done = True
                            state.cell_warmup = False
                            warmup_duration = now - state.cell_start
                            if state.metrics_available:
                                ready_reason = (
                                    f"running_reqs={state.srv_running_reqs}/{concurrency}, "
                                    f"queue_reqs={state.srv_queue_reqs}, "
                                    f"stable={ready_stable_seconds:.1f}s"
                                )
                            else:
                                ready_reason = (
                                    f"metrics_unavailable, active_streams={shared_active_streams[0]}/{concurrency}, "
                                    f"stable={ready_stable_seconds:.1f}s"
                                )
                            measurement_start = now
                            measurement_end = None
                            state.cell_measurement_start = now
                            hw_measurement_start_idx = len(state.hw_history)
                            add_event(
                                state,
                                f"ready C={concurrency} ctx={format_context(context_tokens)} {ready_reason}",
                            )
                            measurement_tokens_start = shared_token_count[0]
                            measurement_usage_tokens_start = shared_usage_token_count[0]
                            if engine == ENGINE_VLLM:
                                measurement_gen_tokens_start = extract_metric(
                                    metrics, metric_name(engine, "gen_tokens_total")
                                )
                    else:
                        warmup_stable_since = None
                    # Give up after max_warmup — queue never drained (real capacity issue)
                    if elapsed >= max_warmup_seconds:
                        warmup_timed_out = True
                        warmup_done = True
                        state.cell_warmup = False
                        warmup_duration = now - state.cell_start
                        ready_reason = "warmup_timeout"
                        if not state.metrics_available:
                            timeout_reason = "metrics_unavailable"
                        elif state.srv_running_reqs < concurrency:
                            timeout_reason = f"running_reqs={state.srv_running_reqs}/{concurrency}"
                        elif state.srv_queue_reqs > 0:
                            timeout_reason = f"queue_reqs={state.srv_queue_reqs}"
                        elif not generating:
                            timeout_reason = "no_generation"
                        else:
                            timeout_reason = "not_stable"
                        measurement_start = now
                        measurement_end = None
                        state.cell_measurement_start = now
                        hw_measurement_start_idx = len(state.hw_history)
                        add_event(
                            state,
                            f"warmup timeout C={concurrency} ctx={format_context(context_tokens)} {timeout_reason}",
                        )
                        measurement_tokens_start = shared_token_count[0]
                        measurement_usage_tokens_start = shared_usage_token_count[0]
                        if engine == ENGINE_VLLM:
                            measurement_gen_tokens_start = extract_metric(
                                metrics, metric_name(engine, "gen_tokens_total")
                            )

            # Collect samples only after warmup is done
            if warmup_done:
                update_state_request_stats(state, shared_request_samples)
                if state.srv_gen_throughput > 0:
                    gen_throughput_samples.append(state.srv_gen_throughput)
                if state.metrics_available:
                    running_reqs_samples.append(state.srv_running_reqs)
                    queue_reqs_samples.append(state.srv_queue_reqs)
            last_metrics_time = now

        # Sustained Decode live display prefers OpenAI stream usage. Prometheus
        # remains visible as server validation and for scheduler state.
        if measurement_start:
            client_elapsed = now - measurement_start
            measurement_usage_tokens = shared_usage_token_count[0] - measurement_usage_tokens_start
            measurement_tokens = shared_token_count[0] - measurement_tokens_start
            if client_elapsed > 0.5 and measurement_usage_tokens > 0:
                state.cell_live_tps = measurement_usage_tokens / client_elapsed
            elif client_elapsed > 0.5 and measurement_tokens > 0:
                state.cell_live_tps = measurement_tokens / client_elapsed
            elif state.srv_gen_throughput > 0:
                state.cell_live_tps = state.srv_gen_throughput
            if warmup_done and state.cell_live_tps > 0:
                state.cell_tps_history.append(state.cell_live_tps)
                if len(state.cell_tps_history) > 240:
                    state.cell_tps_history = state.cell_tps_history[-240:]

        # Update TUI
        live.update(build_display(state))

        # Check duration (measured from after warmup completes)
        measure_elapsed = (now - measurement_start) if measurement_start else 0
        if measurement_start and measure_elapsed >= duration:
            if engine == ENGINE_VLLM and measurement_gen_tokens_start is not None:
                end_metrics = await scrape_metrics(client, base_url) if state.metrics_available else {}
                measurement_gen_tokens_end = extract_metric(
                    end_metrics, metric_name(engine, "gen_tokens_total")
                )
                measurement_end = time.monotonic()
            cancel_event.set()
            break

        # Check skip key
        if _skip_event.is_set():
            _skip_event.clear()
            cancel_event.set()
            add_event(state, f"cell skipped C={concurrency} ctx={format_context(context_tokens)}")
            return CellResult(
                concurrency=concurrency,
                context_tokens=context_tokens,
                aggregate_tps=-2,
                hardware_summary=summarize_hardware_history(state.hw_history[hw_measurement_start_idx:]),
            )

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
    request_samples = []
    for stream_result in stream_results:
        request_samples.extend(stream_result.request_samples)
    update_state_request_stats(state, request_samples)

    # Final metrics scrape
    metrics = await scrape_metrics(client, base_url) if state.metrics_available else {}
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

    # Use an exact vLLM generation_tokens_total delta over the measured window
    # as the primary decode metric. The previous implementation used the median
    # of 1s counter rates, which is robust for display but can bias short runs.
    exact_server_tokens = 0
    exact_server_throughput = 0.0
    if (
        engine == ENGINE_VLLM
        and measurement_gen_tokens_start is not None
        and measurement_gen_tokens_end is not None
        and measurement_end is not None
        and measurement_end > measurement_start
    ):
        exact_server_tokens = max(
            0,
            int(round(measurement_gen_tokens_end - measurement_gen_tokens_start)),
        )
        exact_server_throughput = exact_server_tokens / (measurement_end - measurement_start)

    server_gen_throughput = exact_server_throughput
    if server_gen_throughput == 0:
        server_gen_throughput = median(gen_throughput_samples) if gen_throughput_samples else 0.0

    measure_duration = (measurement_end - measurement_start) if (
        measurement_start and measurement_end and measurement_end > measurement_start
    ) else ((time.monotonic() - measurement_start) if measurement_start else wall_time)
    measurement_tokens = max(0, shared_token_count[0] - measurement_tokens_start)
    measurement_usage_tokens = max(0, shared_usage_token_count[0] - measurement_usage_tokens_start)
    aggregate_source = ""
    if measure_duration > 0 and measurement_usage_tokens > 0:
        avg_gen_throughput = measurement_usage_tokens / measure_duration
        aggregate_source = "openai_continuous_usage"
    elif measure_duration > 0 and measurement_tokens > 0:
        avg_gen_throughput = measurement_tokens / measure_duration
        aggregate_source = "openai_stream_chunks_fallback"
    else:
        avg_gen_throughput = server_gen_throughput
        aggregate_source = "prometheus_fallback" if server_gen_throughput > 0 else "none"

    # Client-side stats
    successful = [r for r in stream_results if r.error is None]
    total_tokens = (
        measurement_usage_tokens
        if measurement_usage_tokens > 0
        else (exact_server_tokens if exact_server_tokens > 0 else sum(r.total_tokens for r in stream_results))
    )

    # Derive per-request from aggregate for consistency
    per_req_tps = avg_gen_throughput / concurrency if concurrency > 0 else 0.0
    request_summary = summarize_request_samples(request_samples)
    ttft_stats = request_summary["ttft"]
    ttst_stats = request_summary["time_to_second_token"]
    request_latency_stats = request_summary["request_latency"]
    itl_stats = request_summary["inter_token_latency"]
    chunk_itl_stats = request_summary["chunk_inter_token_latency"]
    user_tps_stats = request_summary["output_tps_per_user"]
    e2e_user_tps_stats = request_summary["e2e_output_tps_per_user"]
    input_seq_stats = request_summary["input_seq_len"]
    output_seq_stats = request_summary["output_seq_len"]

    # Queue stats
    avg_running = mean(running_reqs_samples) if running_reqs_samples else 0.0
    max_running = max(running_reqs_samples) if running_reqs_samples else 0
    avg_queue = mean(queue_reqs_samples) if queue_reqs_samples else 0.0
    max_queue = max(queue_reqs_samples) if queue_reqs_samples else 0
    queued_count = sum(1 for q in queue_reqs_samples if q > 0)
    queue_frac = queued_count / len(queue_reqs_samples) if queue_reqs_samples else 0.0
    has_scheduler_samples = bool(running_reqs_samples) and (max_running > 0 or max_queue > 0)
    # A cell can be queue-free but still invalid for the requested concurrency:
    # vLLM may only admit the subset that fits in KV cache, then report queue=0
    # once the rejected/waiting work has drained. Track the effective concurrency
    # separately so headline tables do not mistake overload behavior for speed.
    underfilled = (
        concurrency > 1
        and has_scheduler_samples
        and avg_running < concurrency * 0.98
    )
    capacity_limited = (
        warmup_timed_out
        or (concurrency > 1 and has_scheduler_samples and max_running < concurrency)
        or (concurrency > 1 and has_scheduler_samples and avg_running < concurrency * 0.95)
        or avg_queue > 0
        or queue_frac > 0
    )

    cell = CellResult(
        concurrency=concurrency,
        context_tokens=context_tokens,
        benchmark_mode="duration",
        measurement_seconds=round(measure_duration, 6),
        client_output_tokens=measurement_usage_tokens if measurement_usage_tokens > 0 else measurement_tokens,
        server_output_tokens=exact_server_tokens,
        aggregate_source=aggregate_source,
        aggregate_tps=avg_gen_throughput,
        per_request_avg_tps=per_req_tps,
        ttft_avg=ttft_stats["avg"],
        ttft_p50=ttft_stats["p50"],
        ttft_p90=ttft_stats["p90"],
        ttft_p99=ttft_stats["p99"],
        time_to_second_token_avg=ttst_stats["avg"],
        time_to_second_token_p50=ttst_stats["p50"],
        time_to_second_token_p90=ttst_stats["p90"],
        time_to_second_token_p99=ttst_stats["p99"],
        request_latency_avg=request_latency_stats["avg"],
        request_latency_p50=request_latency_stats["p50"],
        request_latency_p90=request_latency_stats["p90"],
        request_latency_p99=request_latency_stats["p99"],
        inter_token_latency_avg=itl_stats["avg"],
        inter_token_latency_p50=itl_stats["p50"],
        inter_token_latency_p90=itl_stats["p90"],
        inter_token_latency_p99=itl_stats["p99"],
        output_tps_per_user_avg=user_tps_stats["avg"],
        output_tps_per_user_p50=user_tps_stats["p50"],
        output_tps_per_user_p90=user_tps_stats["p90"],
        output_tps_per_user_p99=user_tps_stats["p99"],
        e2e_output_tps_per_user_avg=e2e_user_tps_stats["avg"],
        e2e_output_tps_per_user_p50=e2e_user_tps_stats["p50"],
        e2e_output_tps_per_user_p90=e2e_user_tps_stats["p90"],
        e2e_output_tps_per_user_p99=e2e_user_tps_stats["p99"],
        chunk_inter_token_latency_avg=chunk_itl_stats["avg"],
        chunk_inter_token_latency_p50=chunk_itl_stats["p50"],
        chunk_inter_token_latency_p90=chunk_itl_stats["p90"],
        chunk_inter_token_latency_p99=chunk_itl_stats["p99"],
        input_seq_len_avg=input_seq_stats["avg"],
        output_seq_len_avg=output_seq_stats["avg"],
        output_seq_len_p50=output_seq_stats["p50"],
        output_seq_len_p90=output_seq_stats["p90"],
        output_seq_len_p99=output_seq_stats["p99"],
        request_count=request_summary["request_count"],
        completed_request_count=request_summary["completed_request_count"],
        request_samples=[asdict(s) for s in request_samples],
        total_tokens=total_tokens,
        wall_time=wall_time,
        num_completed=len(successful),
        num_errors=len(stream_results) - len(successful),
        server_gen_throughput=server_gen_throughput,
        server_utilization=extract_metric(metrics, metric_name(engine, "utilization")),
        server_spec_accept_rate=state.srv_spec_accept_rate,
        avg_running_reqs=round(avg_running, 1),
        max_running_reqs=max_running,
        effective_concurrency=round(avg_running, 1),
        avg_queue_reqs=round(avg_queue, 1),
        max_queue_reqs=max_queue,
        queue_fraction=round(queue_frac, 3),
        underfilled=underfilled,
        warmup_timed_out=warmup_timed_out,
        warmup_duration=round(warmup_duration, 3),
        ready_reason=ready_reason,
        timeout_reason=timeout_reason,
        capacity_limited=capacity_limited,
        hardware_summary=summarize_hardware_history(state.hw_history[hw_measurement_start_idx:]),
    )

    state.cell_running = False
    state.results[(context_tokens, concurrency)] = cell.aggregate_tps
    state.errors[(context_tokens, concurrency)] = cell.num_errors
    state.queue_info[(context_tokens, concurrency)] = (
        cell.avg_running_reqs,
        cell.avg_queue_reqs,
        cell.capacity_limited,
    )
    state.client_info[(context_tokens, concurrency)] = compact_client_info_from_cell(cell)
    add_event(state, f"cell done C={concurrency} ctx={format_context(context_tokens)} {cell.aggregate_tps:.1f} tok/s")

    await cell_client.aclose()
    return cell


# ---------------------------------------------------------------------------
# TUI rendering
# ---------------------------------------------------------------------------

def build_display(state: TUIState) -> Layout:
    mode, term_width, term_height = live_layout_mode()
    narrow = mode == "narrow"
    mid = mode == "mid"
    show_hw_panel = state.hw_monitor_enabled
    current_ratio, server_ratio, hardware_ratio = (5, 3, 7)
    if mid:
        # Mid-width terminals still have enough horizontal space for a real
        # GPU table if the hardware pane is given priority. The current/server
        # panes tolerate tighter widths better than 8 GPU telemetry rows do.
        current_ratio, server_ratio, hardware_ratio = (4, 2, 8)
    ratio_sum = current_ratio + server_ratio + (hardware_ratio if show_hw_panel else 0)
    hardware_est_width = max(0, int(term_width * hardware_ratio / max(1, ratio_sum))) if show_hw_panel else 0
    show_narrow_hw = show_hw_panel and narrow and term_height >= 30
    narrow_hw_size = 0
    narrow_top_size = 0
    if show_narrow_hw:
        narrow_hw_size = 10 if term_height >= 40 else (8 if term_height >= 34 else 6)
        narrow_top_size = 8 if term_height >= 34 else 7
    middle_size = 14 if not narrow else ((narrow_top_size + narrow_hw_size) if show_narrow_hw else 10)
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="middle", size=middle_size),
        Layout(name="results", ratio=1, minimum_size=10 if not narrow else 12),
        Layout(name="footer", size=3),
    )
    if narrow:
        if show_narrow_hw:
            layout["middle"].split_column(
                Layout(name="middle_top", size=narrow_top_size),
                Layout(name="hardware_metrics", size=narrow_hw_size),
            )
            layout["middle_top"].split_row(
                Layout(name="current_test", ratio=3),
                Layout(name="server_metrics", ratio=2),
            )
        else:
            layout["middle"].split_row(
                Layout(name="current_test", ratio=3),
                Layout(name="server_metrics", ratio=2),
            )
    elif show_hw_panel:
        layout["middle"].split_row(
            Layout(name="current_test", ratio=current_ratio),
            Layout(name="server_metrics", ratio=server_ratio),
            Layout(name="hardware_metrics", ratio=hardware_ratio),
        )
    else:
        layout["middle"].split_row(
            Layout(name="current_test", ratio=current_ratio),
            Layout(name="server_metrics", ratio=server_ratio),
        )

    # Header
    header_text = Text()
    engine_label = state.engine.upper() if state.engine else "Benchmark"
    header_text.append(f"{engine_label}", style=f"bold {PHOSPHOR_SOFT}")
    header_text.append(f" v{VERSION}", style="dim")
    if not narrow:
        header_text.append(f"  {state.model_name}", style=TEXT_PRIMARY)
        header_text.append(f" @ {state.server_url}", style="dim")
    cell_mode = (
        f"{state.request_count_target} req each"
        if state.benchmark_mode == "request-count" and state.request_count_target > 0
        else f"{state.cell_duration:.0f}s each"
    )
    header_text.append(f" | {state.total_tests} tests | {cell_mode}", style=FRAME_BORDER)
    if state.kv_cache_budget > 0 or state.max_running_requests > 0:
        header_text.append("  |  ", style=FRAME_BORDER)
        if state.kv_cache_budget > 0:
            header_text.append(f"KV: {state.kv_cache_budget:,}", style=PHOSPHOR)
        if state.max_running_requests > 0:
            if state.kv_cache_budget > 0:
                header_text.append("  ", style=FRAME_BORDER)
            header_text.append(f"MaxReqs: {state.max_running_requests}", style=PHOSPHOR)
        if state.skipped_cells > 0:
            header_text.append(f"  ({state.skipped_cells} skipped)", style=PHOSPHOR_WARN)
    if not state.metrics_available:
        header_text.append("  | metrics disabled", style=PHOSPHOR_WARN)
    layout["header"].update(
        Panel(
            header_text,
            box=HEADER_BOX,
            border_style=FRAME_BORDER,
            padding=(0, 1) if not narrow else (0, 0),
        )
    )

    # Current test panel
    if state.cell_running:
        elapsed = time.monotonic() - state.cell_start
        if state.prefill_phase:
            last_sample = "last sample: pending"
            if state.prefill_last_tps > 0:
                last_sample = f"last sample: {state.prefill_last_tps:,.0f} tok/s ({state.prefill_method})"
            status = state.prefill_status or "warming up"
            prefill_eta = format_prefill_eta(state, elapsed)
            cell_text = (
                f"[bold {PHOSPHOR_SOFT}]PREFILL[/bold {PHOSPHOR_SOFT}]  [dim]ctx=[/dim][{TEXT_PRIMARY}]{format_context(state.current_context)}[/{TEXT_PRIMARY}]\n"
                f"  Elapsed: {format_time(elapsed)}\n"
                f"  status: {status}\n"
                f"  {prefill_eta}\n"
                f"  [dim]{last_sample}[/dim]  samples={state.prefill_samples_done}\n"
                f"  [dim]server now:[/dim] running={state.srv_running_reqs} queue={state.srv_queue_reqs} "
                f"kv={state.srv_utilization:.2%}\n"
                f"  Test [bold]{state.completed_tests + 1}[/bold] of {state.total_tests}"
            )
        else:
            if state.cell_warmup:
                # During ramp-up: show elapsed warmup time and why we're waiting
                if state.srv_gen_throughput == 0 and state.cell_tokens == 0:
                    wait_reason = "waiting for token generation (JIT compile?)"
                elif not state.metrics_available:
                    active = getattr(state, "_active_streams", 0)
                    wait_reason = f"metrics disabled; waiting for streams ({active}/{state.current_concurrency})"
                elif state.srv_queue_reqs > 0:
                    wait_reason = "waiting for queue→0 (prefill ramp-up)"
                elif state.srv_running_reqs < state.current_concurrency:
                    wait_reason = f"waiting for running_reqs ({state.srv_running_reqs}/{state.current_concurrency})"
                elif hasattr(state, '_active_streams') and state._active_streams < state.current_concurrency:
                    wait_reason = f"waiting for all streams ({state._active_streams}/{state.current_concurrency})"
                else:
                    wait_reason = "stabilizing..."
                req_stats = "client stats: waiting for completed stream samples"
                if state.cell_request_samples > 0:
                    req_stats = (
                        f"client: req={state.cell_completed_requests}/{state.cell_request_samples} "
                        f"TTFTp50={state.cell_ttft_p50_ms:.0f}ms "
                        f"ITLp50={state.cell_itl_p50_ms:.1f}ms"
                    )
                cell_text = (
                    f"[bold {PHOSPHOR_SOFT}]DECODE[/bold {PHOSPHOR_SOFT}]  C={state.current_concurrency}, ctx={format_context(state.current_context)}"
                    f"  [{PHOSPHOR_WARN}]RAMP-UP[/{PHOSPHOR_WARN}]\n"
                    f"  {wait_reason} {elapsed:.0f}s\n"
                    f"  Server: [bold {PHOSPHOR}]{state.cell_live_tps:.1f}[/bold {PHOSPHOR}] tok/s  "
                    f"running={state.srv_running_reqs} queue={state.srv_queue_reqs}\n"
                    f"  [dim]{req_stats}[/dim]\n"
                    f"  Test [bold]{state.completed_tests + 1}[/bold] of {state.total_tests}"
                )
            else:
                # Measurement phase: progress bar from measurement_start or
                # completed requests in request-count mode.
                measure_elapsed = (time.monotonic() - state.cell_measurement_start) if state.cell_measurement_start > 0 else elapsed
                if state.benchmark_mode == "request-count" and state.request_count_target > 0:
                    pct = min(state.cell_completed_requests / state.request_count_target, 1.0)
                    progress_label = (
                        f"{state.cell_completed_requests}/{state.request_count_target} req "
                        f"({measure_elapsed:.0f}s)"
                    )
                else:
                    pct = min(measure_elapsed / state.cell_duration, 1.0) if state.cell_duration > 0 else 0
                    progress_label = f"{measure_elapsed:.0f}/{state.cell_duration:.0f}s"
                speed_label = "Client"
                bar = render_progress_bar(pct, 18 if narrow else 30)
                req_stats = "client stats: collecting"
                latency = ""
                if state.cell_request_samples > 0:
                    req_stats = (
                        f"req={state.cell_completed_requests}/{state.cell_request_samples} "
                        f"TTFTp50={state.cell_ttft_p50_ms:.0f}ms "
                        f"ITLp50={state.cell_itl_p50_ms:.1f}ms "
                        f"userp50={state.cell_user_tps_p50:.1f} tok/s"
                    )
                    if state.cell_request_latency_p50_ms > 0:
                        latency = (
                            f"\n  [dim]latency p50/p90="
                            f"{state.cell_request_latency_p50_ms:.0f}/"
                            f"{state.cell_request_latency_p90_ms:.0f}ms[/dim]"
                        )

                cell_text = (
                    f"[bold {PHOSPHOR_SOFT}]DECODE[/bold {PHOSPHOR_SOFT}]  C={state.current_concurrency}, ctx={format_context(state.current_context)}\n"
                    f"  {bar}  {progress_label}\n"
                    f"  {speed_label}: [bold {PHOSPHOR}]{state.cell_live_tps:.1f}[/bold {PHOSPHOR}] tok/s\n"
                    f"  [dim]{req_stats}[/dim]{latency}\n"
                    f"  {render_speed_trace(state.cell_tps_history)}\n"
                    f"  Test [bold]{state.completed_tests + 1}[/bold] of {state.total_tests}"
                )
    else:
        cell_text = "[dim]Waiting...[/dim]"
    layout["current_test"].update(
        Panel(
            cell_text,
            title=render_title("CURRENT"),
            title_align="left",
            box=PANEL_BOX,
            border_style=FRAME_BORDER,
            padding=(0, 1),
        )
    )

    # Server metrics panel
    srv_table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    srv_table.add_column("Metric", style="dim")
    srv_table.add_column("Value", style=f"bold {TEXT_PRIMARY}", justify="right")
    if not state.metrics_available:
        srv_table.add_row("metrics", f"[{PHOSPHOR_WARN}]disabled[/]")
        srv_table.add_row("source", "OpenAI stream")
        srv_table.add_row("scheduler", "[dim]unavailable[/dim]")
    else:
        srv_table.add_row("gen_throughput", f"[{PHOSPHOR}]{state.srv_gen_throughput:.1f} tok/s[/{PHOSPHOR}]")
        srv_table.add_row("running_reqs", str(state.srv_running_reqs))
        srv_table.add_row("queue_reqs", str(state.srv_queue_reqs))
        srv_table.add_row("utilization", f"[{PHOSPHOR_DIM}]{state.srv_utilization:.2%}[/{PHOSPHOR_DIM}]")
    if state.metrics_available and (state.srv_spec_accept_rate > 0 or state.srv_spec_accept_length > 0):
        srv_table.add_row("spec_accept_rate", f"[{PHOSPHOR_SOFT}]{state.srv_spec_accept_rate:.2%}[/{PHOSPHOR_SOFT}]")
        srv_table.add_row("spec_accept_len", f"[{PHOSPHOR_SOFT}]{state.srv_spec_accept_length:.1f}[/{PHOSPHOR_SOFT}]")
    layout["server_metrics"].update(
        Panel(
            srv_table,
            title=render_title("SERVER"),
            title_align="left",
            box=PANEL_BOX,
            border_style=FRAME_BORDER,
            padding=(0, 0) if narrow else (0, 1),
        )
    )

    if show_hw_panel and not narrow:
        if mode == "wide" or hardware_est_width >= 64:
            layout["hardware_metrics"].update(
                render_hardware_panel(state, dense=(mode == "mid" and hardware_est_width < 92))
            )
        else:
            layout["hardware_metrics"].update(render_compact_hardware_panel(state, paired=False))
    elif show_hw_panel and show_narrow_hw:
        layout["hardware_metrics"].update(render_compact_hardware_panel(state))

    # Results table
    prefill_visible = bool(state.prefill_contexts)
    detail_mode = live_decode_detail_mode(
        state,
        mode,
        term_width,
        term_height,
        middle_size,
        prefill_visible,
    )
    show_cell_details = detail_mode != "none"
    inline_cell_details = detail_mode == "inline"
    col_width = live_decode_column_width(mode, detail_mode=detail_mode)
    decode_suffix = "tok/s + TTFT/ITL" if show_cell_details else ("" if narrow else "tok/s")
    results_table = Table(
        title=render_title("DECODE tok/s" if narrow else "AGGREGATE DECODE", decode_suffix),
        title_justify="left",
        box=TABLE_BOX,
        border_style=FRAME_BORDER,
        header_style=f"bold {PHOSPHOR_DIM}",
        expand=False,
        pad_edge=False,
    )
    compact_header = narrow or show_cell_details
    results_table.add_column(
        "ctx\\C" if compact_header else "ctx \\ conc",
        style=f"bold {PHOSPHOR_SOFT}",
        min_width=5 if compact_header else 8,
        no_wrap=True,
    )
    for conc in state.concurrency_levels:
        results_table.add_column(str(conc), justify="right", min_width=col_width, max_width=col_width)

    def live_client_info() -> dict:
        return {
            "ttft_ms": state.cell_ttft_p50_ms,
            "itl_ms": state.cell_itl_p50_ms,
            "user_tps": state.cell_user_tps_p50,
            "latency_p50_ms": state.cell_request_latency_p50_ms,
            "latency_p90_ms": state.cell_request_latency_p90_ms,
            "request_count": state.cell_request_samples,
            "completed_request_count": state.cell_completed_requests,
        }

    def render_decode_matrix_cell(value: float, detail: str = "", suffix: str = "") -> str:
        primary = f"[bold {PHOSPHOR}]{compact_decode_cell(value, mode)}[/bold {PHOSPHOR}]{suffix}"
        if show_cell_details and detail:
            if inline_cell_details:
                return f"{primary} [dim]{detail}[/dim]"
            return f"{primary}\n[dim]{detail}[/dim]"
        return primary

    for ctx in state.context_lengths:
        row = [format_context(ctx)]
        for conc in state.concurrency_levels:
            key = (ctx, conc)
            if key in state.results:
                val = state.results[key]
                if val == -2:
                    row.append(f"[{PHOSPHOR_WARN}]{'skip' if narrow else 'skipped'}[/{PHOSPHOR_WARN}]")
                    continue
                if val < 0:
                    row.append(capacity_limit_cell(styled=True))
                    continue
                errs = state.errors.get(key, 0)
                style = f"bold {PHOSPHOR}"
                qi = state.queue_info.get(key)
                limited = bool(qi and qi[2])
                if limited and not state.show_capacity_limited_values:
                    cell = capacity_limit_cell()
                else:
                    cell = compact_decode_cell(val, mode)
                if errs > 0:
                    cell += f"[red]e[/red]" if narrow or show_cell_details else f" [red]({errs}e)[/red]"
                # Show running/requested when scheduler did not sustain the
                # requested concurrency. This catches vLLM KV-capacity cases
                # where queue drains to zero after only a subset is admitted.
                if qi:
                    avg_run, avg_q, limited = qi
                    if limited or avg_q > 0 or (avg_run > 0 and avg_run < conc * 0.98):
                        mark = "*" if limited else ""
                        if narrow or show_cell_details:
                            cell += f"[{PHOSPHOR_WARN}]*[/{PHOSPHOR_WARN}]"
                        else:
                            cell += f" [{PHOSPHOR_WARN}]({avg_run:.0f}/{conc}){mark}[/{PHOSPHOR_WARN}]"
                detail = compact_client_cell_detail(state.client_info.get(key, {}))
                if show_cell_details and detail:
                    if inline_cell_details:
                        row.append(f"[{style}]{cell}[/{style}] [dim]{detail}[/dim]")
                    else:
                        row.append(f"[{style}]{cell}[/{style}]\n[dim]{detail}[/dim]")
                else:
                    row.append(f"[{style}]{cell}[/{style}]")
            elif (
                state.cell_running
                and not state.prefill_phase
                and key == (state.current_context, state.current_concurrency)
                and state.cell_live_tps > 0
            ):
                detail = compact_client_cell_detail(live_client_info())
                row.append(render_decode_matrix_cell(state.cell_live_tps, detail))
            else:
                row.append("[dim]...[/dim]")
        results_table.add_row(*row)

    decode_body = (
        Group(
            results_table,
            Text(
                "Detail: TTFT ms / ITL ms. "
                "ITL uses observed generated tokens; latency needs completed streams.",
                style="dim",
            ),
        )
        if show_cell_details
        else results_table
    )

    # Prefill table (shown alongside decode results)
    if state.prefill_contexts:
        prefill_table = Table(
            title=render_title("PREFILL" if not narrow else "PF", "C=1" if not narrow else ""),
            title_justify="left",
            box=REPORT_BOX,
            border_style=SUBTLE_BORDER,
            header_style=f"bold {PHOSPHOR_DIM}",
            expand=False,
            pad_edge=False,
        )
        prefill_table.add_column("ctx", style=f"bold {PHOSPHOR_SOFT}")
        prefill_table.add_column("TTFT", justify="right")
        prefill_table.add_column("tok/s", justify="right")
        prefill_table.add_column("N", justify="right")
        for ctx in state.prefill_contexts:
            if ctx in state.prefill_results:
                pr = state.prefill_results[ctx]
                if pr.get("skipped"):
                    prefill_table.add_row(format_context(ctx), f"[{PHOSPHOR_WARN}]skip[/{PHOSPHOR_WARN}]", "", "")
                else:
                    prefill_table.add_row(
                        format_context(ctx),
                        f"{pr['ttft']:.2f}s",
                        f"[bold {PHOSPHOR}]{pr['tok_per_sec']:,.0f}[/bold {PHOSPHOR}]",
                        str(pr.get('samples', '')),
                    )
            else:
                prefill_table.add_row(format_context(ctx), "[dim]...[/dim]", "[dim]...[/dim]", "")

        decode_width = live_decode_panel_width(state, prefill_visible=True, detail_mode=detail_mode)
        results_layout = Layout()
        prefill_panel = Panel(
            prefill_table,
            box=PANEL_BOX,
            border_style=SUBTLE_BORDER,
            padding=(0, 1) if not narrow else (0, 0),
        )
        decode_panel = Panel(
            decode_body,
            box=PANEL_BOX,
            border_style=SUBTLE_BORDER,
            padding=(0, 1) if not narrow else (0, 0),
        )
        results_height = max(8, term_height - middle_size - 6)
        mid_side_events = mid and results_height >= 16 and term_width >= decode_width + 28
        side_event_limit = max(4, results_height - 7)
        full_event_limit = max(6, results_height - 2)
        show_side_stats = False
        stats_size = 8 if results_height >= 26 else 7
        if mode == "wide":
            side = Layout()
            if show_side_stats:
                side.split_column(
                    Layout(render_live_stats_panel(state), size=stats_size),
                    Layout(render_events_panel(state, limit=max(4, results_height - stats_size - 2)), ratio=1),
                )
            else:
                side.update(render_events_panel(state, limit=full_event_limit))
            results_layout.split_row(
                Layout(prefill_panel, size=31),
                Layout(decode_panel, size=decode_width),
                Layout(side, ratio=1, minimum_size=18),
            )
        elif mid_side_events:
            side = Layout()
            if show_side_stats:
                side.split_column(
                    Layout(render_compact_prefill_panel(state), size=5),
                    Layout(render_live_stats_panel(state), size=stats_size),
                    Layout(render_events_panel(state, limit=max(3, side_event_limit - stats_size)), ratio=1, minimum_size=4),
                )
            else:
                side.split_column(
                    Layout(render_compact_prefill_panel(state), size=5),
                    Layout(render_events_panel(state, limit=side_event_limit), ratio=1, minimum_size=6),
                )
            results_layout.split_row(
                Layout(decode_panel, size=decode_width),
                Layout(side, ratio=1, minimum_size=22),
            )
        elif mode == "mid":
            bottom_size = 10 if results_height >= 22 else (8 if results_height >= 18 else 5)
            bottom = Layout()
            bottom.split_row(
                Layout(render_compact_prefill_panel(state), size=31),
                Layout(render_events_panel(state, limit=max(4, bottom_size - 2)), ratio=1),
            )
            results_layout.split_column(
                Layout(decode_panel, ratio=1, minimum_size=9),
                Layout(bottom, size=bottom_size),
            )
        else:
            decode_rows = min(narrow_decode_panel_size(state), max(6, results_height - 4))
            remaining_rows = max(3, results_height - decode_rows)
            prefill_rows = 4 if remaining_rows >= 7 else 0
            event_rows = max(3, results_height - decode_rows - prefill_rows)
            if prefill_rows:
                results_layout.split_column(
                    Layout(decode_panel, size=decode_rows),
                    Layout(render_compact_prefill_panel(state, limit=3), size=prefill_rows),
                    Layout(render_events_panel(state, limit=max(1, event_rows - 2)), size=event_rows),
                )
            else:
                results_layout.split_column(
                    Layout(decode_panel, size=decode_rows),
                    Layout(render_events_panel(state, limit=max(1, event_rows - 2)), size=event_rows),
                )
        layout["results"].update(results_layout)
    else:
        decode_width = live_decode_panel_width(state, prefill_visible=False, detail_mode=detail_mode)
        results_layout = Layout()
        decode_panel = Panel(
            decode_body,
            box=PANEL_BOX,
            border_style=SUBTLE_BORDER,
            padding=(0, 1) if not narrow else (0, 0),
        )
        results_height = max(8, term_height - middle_size - 6)
        mid_side_events = mid and results_height >= 14 and term_width >= decode_width + 28
        full_event_limit = max(6, results_height - 2)
        show_side_stats = False
        stats_size = 8 if results_height >= 24 else 7
        if mode == "wide":
            side = Layout()
            if show_side_stats:
                side.split_column(
                    Layout(render_live_stats_panel(state), size=stats_size),
                    Layout(render_events_panel(state, limit=max(4, results_height - stats_size - 2)), ratio=1),
                )
            else:
                side.update(render_events_panel(state, limit=full_event_limit))
            results_layout.split_row(
                Layout(decode_panel, size=decode_width),
                Layout(side, ratio=1, minimum_size=18),
            )
        elif mid_side_events:
            side = Layout()
            if show_side_stats:
                side.split_column(
                    Layout(render_live_stats_panel(state), size=stats_size),
                    Layout(render_events_panel(state, limit=max(4, results_height - stats_size - 2)), ratio=1),
                )
            else:
                side.update(render_events_panel(state, limit=full_event_limit))
            results_layout.split_row(
                Layout(decode_panel, size=decode_width),
                Layout(side, ratio=1, minimum_size=22),
            )
        elif mode == "mid":
            bottom_size = 10 if results_height >= 22 else (8 if results_height >= 18 else 5)
            results_layout.split_column(
                Layout(decode_panel, ratio=3, minimum_size=7),
                Layout(render_events_panel(state, limit=max(5, bottom_size - 2)), size=bottom_size),
            )
        else:
            decode_rows = min(narrow_decode_panel_size(state), max(6, results_height - 4))
            event_rows = max(3, results_height - decode_rows)
            results_layout.split_column(
                Layout(decode_panel, size=decode_rows),
                Layout(render_events_panel(state, limit=max(1, event_rows - 2)), size=event_rows),
            )
        layout["results"].update(results_layout)

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

        bar = render_progress_bar(overall_pct, 24 if narrow else 50)
        if narrow or mid:
            footer_text = (
                f" {bar} {state.completed_tests}/{state.total_tests} "
                f"E:{format_time(elapsed_total)} ETA:{eta_str} "
                f"[dim]v{VERSION} s=skip q=finish[/dim]"
            )
        else:
            footer_text = (
                f"  {bar}  "
                f"{state.completed_tests}/{state.total_tests}  "
                f"Elapsed: {format_time(elapsed_total)}  "
                f"ETA: {eta_str}  "
            )
            footer_text += f"[dim]{CAPACITY_LIMIT_MARK}=KV limit  llm-decode-bench v{VERSION}  s=skip  q=finish[/dim]"
    else:
        footer_text = "Initializing..."
    layout["footer"].update(
        Panel(
            footer_text,
            box=PANEL_BOX,
            border_style=FRAME_BORDER,
            padding=(0, 1),
        )
    )

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
            parsed = urlparse(base_url)
            if parsed.port is None:
                base_url = f"{parsed.scheme}://{parsed.hostname}:{args.port}{parsed.path}"
    else:
        port = args.port if args.port is not None else 5000
        base_url = f"http://{args.host}:{port}"
    server_label = base_url.replace("http://", "").replace("https://", "")
    auth_headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    console = Console()
    startup_events: list[str] = []

    def remember_startup(message: str) -> None:
        startup_events.append(message)

    startup_diagnostics = collect_startup_diagnostics(args, base_url)
    setattr(args, "startup_diagnostics", startup_diagnostics)

    # --kv-budget overrides --max-total-tokens
    if args.kv_budget > 0:
        args.max_total_tokens = args.kv_budget
        msg = f"KV cache budget manual: {args.max_total_tokens:,} tokens"
        console.print(f"[cyan]KV cache budget (manual):[/cyan] {args.max_total_tokens:,} tokens")
        remember_startup(msg)

    # --- Step 1: Connect to server, detect engine, and read limits ---
    server_context_length = 0
    max_running = None
    engine = ENGINE_SGLANG
    metrics_available = True
    metrics_warning = ""
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
            return [], [], {}, engine

        # Detect engine: try SGLang's /get_server_info first, then vLLM's /version
        try:
            resp = await check_client.get(f"{base_url}/get_server_info", timeout=10.0)
            server_info = resp.json()
            if "max_total_num_tokens" in server_info:
                engine = ENGINE_SGLANG
                console.print(f"[green]Engine: SGLang {server_info.get('version', '?')}[/green]  Models: {model_ids}")
                remember_startup(f"engine SGLang {server_info.get('version', '?')} models={model_ids}")
                if args.max_total_tokens == 0:
                    kv_budget = server_info.get("max_total_num_tokens", 0)
                    if kv_budget:
                        args.max_total_tokens = int(kv_budget)
                        console.print(f"[cyan]KV cache budget:[/cyan] {args.max_total_tokens:,} tokens")
                        remember_startup(f"KV cache budget: {args.max_total_tokens:,} tokens")
                max_running = server_info.get("max_running_requests")
                if max_running:
                    max_running = int(max_running)
                server_context_length = server_info.get("context_length") or server_context_length
                # SGLang metrics are useful for scheduler/effective-concurrency
                # validation, but they are no longer required for the portable
                # OpenAI-stream headline metrics.
                try:
                    metrics_resp = await check_client.get(f"{base_url}/metrics", timeout=5.0)
                    if "sglang:" not in metrics_resp.text[:2000]:
                        metrics_available = False
                        metrics_warning = (
                            "SGLang /metrics is reachable but does not expose sglang:* metrics; "
                            "scheduler/effective-concurrency and Prometheus validation are disabled. "
                            "Restart with --enable-metrics for those diagnostics."
                        )
                        console.print(
                            "[bold yellow]WARNING: SGLang metrics are disabled.[/bold yellow]\n"
                            f"{metrics_warning}"
                        )
                        remember_startup(f"WARNING: {metrics_warning}")
                except httpx.HTTPError as exc:
                    metrics_available = False
                    metrics_warning = (
                        f"Cannot reach SGLang /metrics endpoint ({type(exc).__name__}); "
                        "scheduler/effective-concurrency and Prometheus validation are disabled. "
                        "Restart with --enable-metrics for those diagnostics."
                    )
                    console.print(
                        "[bold yellow]WARNING: SGLang metrics are disabled.[/bold yellow]\n"
                        f"{metrics_warning}"
                    )
                    remember_startup(f"WARNING: {metrics_warning}")
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
                    remember_startup(f"engine vLLM {version_info['version']} models={model_ids}")
                else:
                    raise ValueError("No version")
            except Exception:
                # Fallback: check /metrics prefix
                try:
                    resp = await check_client.get(f"{base_url}/metrics", timeout=5.0)
                    metrics_head = resp.text[:2000]
                    if resp.status_code < 400 and "vllm:" in metrics_head:
                        engine = ENGINE_VLLM
                        console.print(f"[green]Engine: vLLM (detected from metrics)[/green]  Models: {model_ids}")
                        remember_startup(f"engine vLLM from metrics models={model_ids}")
                    elif resp.status_code < 400 and "sglang:" in metrics_head:
                        engine = ENGINE_SGLANG
                        console.print(f"[green]Engine: SGLang (detected from metrics)[/green]  Models: {model_ids}")
                        remember_startup(f"engine SGLang from metrics models={model_ids}")
                    else:
                        engine = ENGINE_OPENAI_PROXY
                        metrics_available = False
                        reason = f"HTTP {resp.status_code}" if resp.status_code >= 400 else "no vllm:/sglang: metrics"
                        metrics_warning = (
                            f"Only OpenAI /v1 endpoints appear reachable ({reason} on /metrics); "
                            "engine-specific probes and Prometheus diagnostics are disabled. "
                            "Using OpenAI-stream-only measurement."
                        )
                        console.print(f"[yellow]Engine: OpenAI-compatible proxy[/yellow]  Models: {model_ids}")
                        console.print(f"[bold yellow]WARNING:[/bold yellow] {metrics_warning}")
                        remember_startup(f"engine OpenAI-compatible proxy models={model_ids}")
                        remember_startup(f"WARNING: {metrics_warning}")
                except Exception:
                    engine = ENGINE_OPENAI_PROXY
                    metrics_available = False
                    metrics_warning = (
                        "Only OpenAI /v1 endpoints appear reachable; engine-specific probes, "
                        "scheduler/effective-concurrency, and Prometheus validation are disabled. "
                        "Using OpenAI-stream-only measurement."
                    )
                    console.print(f"[yellow]Engine: OpenAI-compatible proxy[/yellow]  Models: {model_ids}")
                    console.print(f"[bold yellow]WARNING:[/bold yellow] {metrics_warning}")
                    remember_startup(f"engine OpenAI-compatible proxy models={model_ids}")
                    remember_startup(f"WARNING: {metrics_warning}")

        if engine == ENGINE_VLLM and metrics_available:
            try:
                metrics_resp = await check_client.get(f"{base_url}/metrics", timeout=5.0)
                if "vllm:" not in metrics_resp.text:
                    metrics_available = False
                    metrics_warning = (
                        "vLLM /metrics is reachable but does not expose vllm:* metrics; "
                        "scheduler/effective-concurrency, KV auto-detection, and Prometheus validation are disabled."
                    )
                    console.print(f"[bold yellow]WARNING:[/bold yellow] {metrics_warning}")
                    remember_startup(f"WARNING: {metrics_warning}")
            except httpx.HTTPError as exc:
                metrics_available = False
                metrics_warning = (
                    f"Cannot reach vLLM /metrics endpoint ({type(exc).__name__}); "
                    "scheduler/effective-concurrency, KV auto-detection, and Prometheus validation are disabled."
                )
                console.print(f"[bold yellow]WARNING:[/bold yellow] {metrics_warning}")
                remember_startup(f"WARNING: {metrics_warning}")

        # vLLM exposes cache_config_info with local num_gpu_blocks and block_size
        # in current builds. With DCP, vLLM multiplies this local budget by the
        # CP/DCP world size in the startup log. Current vLLM builds do not expose
        # the multiplier in Prometheus, so --dcp-size supplies it for remote runs.
        if engine == ENGINE_VLLM and metrics_available and args.max_total_tokens == 0:
            metrics = await scrape_metrics(check_client, base_url)
            block_size = extract_label(metrics, "vllm:cache_config_info", "block_size")
            num_gpu_blocks = extract_label(metrics, "vllm:cache_config_info", "num_gpu_blocks")
            try:
                if block_size and num_gpu_blocks:
                    local_kv_tokens = int(block_size) * int(num_gpu_blocks)
                    cp_size = int(args.dcp_size)
                    cp_source = "argument"
                    if cp_size <= 0:
                        cp_size, cp_source = extract_vllm_cp_size(metrics)
                    if cp_size <= 0:
                        cp_size = infer_local_vllm_dcp_size(base_url)
                        cp_source = "local process" if cp_size > 0 else ""
                    if cp_size > 0:
                        args.max_total_tokens = local_kv_tokens * cp_size
                    elif server_context_length and local_kv_tokens < server_context_length:
                        console.print(
                            "[yellow]vLLM Prometheus reports only local KV cache "
                            f"({local_kv_tokens:,} tokens), but DCP/CP multiplier "
                            "is not exported by this server. Pass --dcp-size N or "
                            "--kv-budget to enable exact KV capacity skips for remote "
                            "vLLM. Leaving KV budget unset instead of assuming DCP=1.[/yellow]"
                        )
                        remember_startup(
                            f"WARNING: vLLM Prometheus reports local KV only ({local_kv_tokens:,}); "
                            "pass --dcp-size or --kv-budget for exact capacity skips"
                        )
                        raise ValueError("vLLM DCP multiplier unknown")
                    else:
                        cp_size = 1
                        cp_source = "default"
                        args.max_total_tokens = local_kv_tokens
                    suffix = ""
                    if cp_size > 1:
                        suffix = f"; local {local_kv_tokens:,} × CP {cp_size}"
                    if cp_source and cp_source != "default":
                        suffix += f"; CP source: {cp_source}"
                    console.print(
                        "[cyan]KV cache budget (vLLM metrics):[/cyan] "
                        f"{args.max_total_tokens:,} tokens "
                        f"({num_gpu_blocks} blocks × {block_size}{suffix})"
                    )
                    remember_startup(
                        f"KV cache budget from vLLM metrics: {args.max_total_tokens:,} tokens "
                        f"({num_gpu_blocks} blocks x {block_size}{suffix})"
                    )
                else:
                    raise ValueError("cache_config_info missing block_size/num_gpu_blocks")
            except Exception:
                console.print(
                    "[yellow]vLLM: KV cache budget not available from metrics. "
                    "Use --kv-budget to skip over-capacity cells, or rely on queue detection.[/yellow]"
                )
                remember_startup(
                    "WARNING: vLLM KV cache budget not available from metrics; "
                    "use --kv-budget for exact capacity skips"
                )

        # Handle max_running_requests
        if max_running:
            over = [c for c in concurrency_levels if c > max_running]
            if over:
                concurrency_levels = [c for c in concurrency_levels if c <= max_running]
                console.print(f"[cyan]Max running requests:[/cyan] {max_running} (dropped C={','.join(str(c) for c in over)})")
                remember_startup(f"max running requests: {max_running}; dropped C={','.join(str(c) for c in over)}")
        if server_context_length:
            console.print(f"[cyan]Model context length:[/cyan] {server_context_length:,} tokens")
            remember_startup(f"model context length: {server_context_length:,} tokens")

    # --- Step 2: Build prefill context list ---
    # Default prefill is integrated into scout requests. Every non-zero decode
    # context already sends one scout request to populate the prefix cache; any
    # --prefill-contexts not present in the decode matrix get one scout-only
    # sample so small sanity points such as 8k remain visible.
    # The old repeated cold-prefill phase remains available for focused ingest
    # profiling with --standalone-prefill.
    PREFILL_CANDIDATES = [parse_token_value(x) for x in args.prefill_contexts.split(",") if x.strip()]
    max_prefill = min(131072, server_context_length - 64) if server_context_length > 0 else 131072
    decode_prefill_contexts = [c for c in context_lengths if c > 0 and c <= max_prefill]
    requested_standalone_prefill_contexts = [c for c in PREFILL_CANDIDATES if c <= max_prefill]
    standalone_prefill_contexts = requested_standalone_prefill_contexts if args.standalone_prefill else []
    prefill_scout_only_contexts = []
    if args.standalone_prefill and not standalone_prefill_contexts and max_prefill > 0:
        standalone_prefill_contexts = [max_prefill]
    if args.skip_prefill:
        prefill_contexts = []
        console.print("[cyan]Prefill tests:[/cyan] skipped")
        remember_startup("prefill tests: skipped")
    elif args.standalone_prefill:
        prefill_contexts = standalone_prefill_contexts
        console.print(
            "[cyan]Prefill tests:[/cyan] standalone cold profile "
            f"{[format_context(c) for c in prefill_contexts]}"
        )
        remember_startup(
            f"prefill tests: standalone cold profile {[format_context(c) for c in prefill_contexts]}"
        )
    else:
        prefill_contexts = sorted(set(decode_prefill_contexts + requested_standalone_prefill_contexts))
        prefill_scout_only_contexts = [c for c in prefill_contexts if c not in decode_prefill_contexts]
        extra = (
            f"; scout-only extras {[format_context(c) for c in prefill_scout_only_contexts]}"
            if prefill_scout_only_contexts else ""
        )
        console.print(
            "[cyan]Prefill tests:[/cyan] integrated from decode scout requests "
            f"{[format_context(c) for c in decode_prefill_contexts]}{extra}"
        )
        remember_startup(
            "prefill tests: integrated from decode scout requests "
            f"{[format_context(c) for c in decode_prefill_contexts]}{extra}"
        )

    # --- Step 3: Generate padding text calibrated to actual token counts ---
    # Default to one calibration request and extrapolate all contexts. Exact
    # /tokenize targeting is available, but it is slow on long context matrices
    # and SGLang deployments often do not expose the endpoint.
    all_ctx_sizes = sorted(set(prefill_contexts + [c for c in context_lengths if c > 0]))
    max_ctx = max(all_ctx_sizes) if all_ctx_sizes else 0
    context_cache = {}
    context_actual_tokens = {}
    run_id = ''.join(random.choices(string.ascii_lowercase, k=12))
    if max_ctx > 0:
        console.print(f"[bold]Calibrating padding text (run={run_id}, up to {format_context(max_ctx)})...[/bold]")
        remember_startup(f"calibrating padding text run={run_id} up_to={format_context(max_ctx)}")
        base_text = generate_padding_text(int(max_ctx * 3))

        def ensure_base_chars(min_chars: int) -> None:
            nonlocal base_text
            if len(base_text) < min_chars:
                base_text = generate_padding_text(max(int(min_chars / CHARS_PER_TOKEN) + 1024, max_ctx * 3))

        async def tokenize_context_text(client: httpx.AsyncClient, ctx: int, text: str) -> int:
            resp = await client.post(
                f"{base_url}/tokenize",
                json={"model": args.model, "messages": build_messages(ctx, text)},
                timeout=httpx.Timeout(120.0, connect=30.0),
            )
            resp.raise_for_status()
            data = resp.json()
            count = int(data.get("count", 0))
            if count <= 0:
                raise ValueError(f"/tokenize returned invalid count: {data}")
            return count

        async def build_exact_context_text(client: httpx.AsyncClient, ctx: int) -> tuple[str, int]:
            prefix = f"[BENCH_{run_id}_CTX_{ctx}] "

            def text_for(total_chars: int) -> str:
                ensure_base_chars(max(0, total_chars - len(prefix)))
                return (prefix + base_text)[:total_chars]

            low = 0
            high = max(1024, ctx * 8)
            best_chars = low
            best_count = await tokenize_context_text(client, ctx, text_for(low))
            max_high = max(high, ctx * 32)

            while True:
                count = await tokenize_context_text(client, ctx, text_for(high))
                if abs(count - ctx) < abs(best_count - ctx):
                    best_chars, best_count = high, count
                if count >= ctx:
                    break
                low = high
                high *= 2
                if high > max_high:
                    raise RuntimeError(
                        f"Could not reach {format_context(ctx)} via /tokenize "
                        f"(last {count:,} tokens at {low:,} chars)"
                    )

            while low + 1 < high:
                mid = (low + high) // 2
                count = await tokenize_context_text(client, ctx, text_for(mid))
                if abs(count - ctx) < abs(best_count - ctx):
                    best_chars, best_count = mid, count
                if count < ctx:
                    low = mid
                else:
                    high = mid

            for candidate in (low, high):
                count = await tokenize_context_text(client, ctx, text_for(candidate))
                if abs(count - ctx) < abs(best_count - ctx):
                    best_chars, best_count = candidate, count

            tolerance = max(64, int(ctx * 0.005))
            if abs(best_count - ctx) > tolerance:
                raise RuntimeError(
                    f"Could not build {format_context(ctx)} prompt accurately: "
                    f"got {best_count:,} tokens at {best_chars:,} chars"
                )
            return text_for(best_chars), best_count

        if args.token_targeting == "exact":
            async with httpx.AsyncClient(headers=auth_headers) as cal_client:
                try:
                    for ctx in all_ctx_sizes:
                        text, actual = await build_exact_context_text(cal_client, ctx)
                        context_cache[ctx] = text
                        context_actual_tokens[ctx] = actual
                        console.print(
                            f"  {format_context(ctx)}: {len(text):,} chars "
                            f"({actual:,} prompt tokens via /tokenize)"
                        )
                        remember_startup(
                            f"context {format_context(ctx)}: {len(text):,} chars "
                            f"({actual:,} prompt tokens via /tokenize)"
                        )
                    console.print("  Token targeting: /tokenize exact")
                    remember_startup("token targeting: /tokenize exact")
                except Exception as e:
                    console.print(
                        "[yellow]  /tokenize exact targeting failed; "
                        f"falling back to single-point estimate: {e}[/yellow]"
                    )
                    remember_startup(
                        f"WARNING: /tokenize exact targeting failed; falling back to estimate: {e}"
                    )
                    context_cache.clear()
                    context_actual_tokens.clear()
        else:
            console.print(
                f"  Token targeting: single-point estimate from {args.calibration_context} "
                "(use --token-targeting exact for /tokenize binary search)"
            )
            remember_startup(f"token targeting: estimate from {args.calibration_context}")

        if not context_cache:
            # Calibrate once, measure actual prompt_tokens, derive chars/token,
            # then extrapolate all requested context sizes.
            calibrated_cpt = CHARS_PER_TOKEN
            template_id = calibration_template_id()
            cache_key = f"{engine}:{args.model}:{template_id}"
            cache = {}
            cached = None
            if args.token_targeting == "estimate" and not args.no_calibration_cache:
                cache = load_calibration_cache(args.calibration_cache)
                cached = cache.get(cache_key)
                if cached and cached.get("chars_per_token", 0) > 0:
                    calibrated_cpt = float(cached["chars_per_token"])
                    console.print(
                        f"  Calibrated: {calibrated_cpt:.2f} chars/token "
                        f"(cached, source={cached.get('source_context', 'unknown')})"
                    )
                    remember_startup(
                        f"calibrated: {calibrated_cpt:.2f} chars/token "
                        f"(cached, source={cached.get('source_context', 'unknown')})"
                    )
                else:
                    cached = None

            if cached is None:
                cal_ctx = min(parse_token_value(args.calibration_context), max_ctx)
                if cal_ctx <= 0:
                    cal_ctx = max_ctx
                async with httpx.AsyncClient(headers=auth_headers) as cal_client:
                    prefix = f"[BENCH_{run_id}_CAL] "
                    for attempt in range(3):
                        target_chars = int(cal_ctx * calibrated_cpt)
                        cal_text = (prefix + base_text)[:target_chars]
                        msgs = build_messages(cal_ctx, cal_text)
                        payload = {
                            "model": args.model, "messages": msgs,
                            "stream": False, "max_tokens": 1,
                        }
                        try:
                            resp = await cal_client.post(
                                f"{base_url}/v1/chat/completions",
                                json=payload,
                                timeout=httpx.Timeout(120.0, connect=30.0),
                            )
                            resp.raise_for_status()
                            data = resp.json()
                            actual = data.get("usage", {}).get("prompt_tokens", 0)
                            if actual > 0:
                                calibrated_cpt = len(cal_text) / actual
                                if abs(actual - cal_ctx) / cal_ctx < 0.05:
                                    break
                        except Exception:
                            break
                console.print(
                    f"  Calibrated: {calibrated_cpt:.2f} chars/token "
                    f"(from {format_context(cal_ctx)} probe)"
                )
                remember_startup(
                    f"calibrated: {calibrated_cpt:.2f} chars/token "
                    f"(from {format_context(cal_ctx)} probe)"
                )
                if args.token_targeting == "estimate" and not args.no_calibration_cache:
                    cache[cache_key] = {
                        "chars_per_token": calibrated_cpt,
                        "engine": engine,
                        "model": args.model,
                        "source_context": format_context(cal_ctx),
                        "template_id": template_id,
                        "updated_at": datetime.now().isoformat(),
                    }
                    save_calibration_cache(args.calibration_cache, cache)

            # Generate text for each context size using calibrated ratio
            for ctx in all_ctx_sizes:
                prefix = f"[BENCH_{run_id}_CTX_{ctx}] "
                target_chars = int(ctx * calibrated_cpt)
                ensure_base_chars(max(0, target_chars - len(prefix)))
                text = (prefix + base_text)[:target_chars]
                context_cache[ctx] = text
                est_tokens = int(len(text) / calibrated_cpt)
                console.print(f"  {format_context(ctx)}: {len(text):,} chars (~{est_tokens:,} tokens)")
                remember_startup(f"context {format_context(ctx)}: {len(text):,} chars (~{est_tokens:,} tokens)")
    context_cache[0] = ""
    console.print("[green]Done.[/green]\n")
    remember_startup("startup preparation done")

    # --- Step 4: Initialize TUI state ---
    state = TUIState(
        engine=engine,
        model_name=args.model,
        server_url=server_label,
        total_tests=len(concurrency_levels) * len(context_lengths),
        concurrency_levels=concurrency_levels,
        context_lengths=context_lengths,
        overall_start=time.monotonic(),
        show_capacity_limited_values=args.show_capacity_limited_values,
        metrics_available=metrics_available,
        metrics_warning=metrics_warning,
    )
    args.metrics_available = metrics_available
    args.metrics_warning = metrics_warning
    add_event(state, f"benchmark start engine={engine or 'unknown'}")
    add_event(state, f"startup server={base_url} model={args.model}")
    add_event(
        state,
        f"startup decode concurrency={','.join(str(c) for c in concurrency_levels)} "
        f"contexts={','.join(format_context(c) for c in context_lengths)}",
    )
    for message in startup_events:
        add_event(state, f"startup {message}")
    if metrics_warning:
        add_event(state, f"startup WARNING: {metrics_warning}")
    if max_running:
        state.max_running_requests = int(max_running)
    state.max_tokens = args.max_tokens
    state.prefill_contexts = prefill_contexts
    state.benchmark_mode = "request-count" if args.request_count > 0 else "duration"
    state.request_count_target = args.request_count
    state.hw_gpu_limit = args.hw_gpu_limit
    hw_monitor_requested = (
        not args.no_hw_monitor
        and args.display_mode != "plain"
        and args.hw_monitor_interval > 0
    )
    if not hw_monitor_requested:
        state.hw_monitor_enabled = False
        state.hw_last_error = "hardware monitor disabled"
        add_event(state, "startup hardware monitor disabled")
    elif shutil.which("nvidia-smi") is None:
        state.hw_monitor_enabled = False
        state.hw_last_error = "nvidia-smi not found"
        console.print(
            "[bold yellow]WARNING:[/bold yellow] nvidia-smi not found; "
            "hardware panel disabled. This usually means the benchmark is not "
            "running on the GPU server/container."
        )
        add_event(
            state,
            "startup WARNING: nvidia-smi not found; hardware panel disabled "
            "(benchmark may be running off the GPU server/container)",
        )
    else:
        start_hardware_monitor(state, args.hw_monitor_interval)
        add_event(state, f"hardware monitor interval={args.hw_monitor_interval:g}s")

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

    # Decode-context prefill is part of each decode context scout request. Only
    # standalone samples and scout-only extra contexts add independent tests.
    state.prefill_contexts = prefill_contexts
    state.total_tests += len(standalone_prefill_contexts) if args.standalone_prefill and not args.skip_prefill else 0
    state.total_tests += len(prefill_scout_only_contexts) if not args.standalone_prefill and not args.skip_prefill else 0
    if args.run_burst and args.request_count == 0:
        state.total_tests += len(concurrency_levels) * len(context_lengths)

    # Run benchmark
    global _partial_results, _prefill_results
    all_results = []
    burst_results = []
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

    async def measure_prefill_scout_only(client, ctx: int, live) -> None:
        """Measure one scout-only prefill context not present in decode matrix."""
        add_event(state, f"prefill scout-only start ctx={format_context(ctx)}")
        state.prefill_phase = True
        state.current_concurrency = 1
        state.current_context = ctx
        state.cell_running = True
        state.cell_start = time.monotonic()
        state.cell_duration = 0
        state.prefill_status = "scout-only prefill sample"
        state.prefill_samples_done = 0
        state.prefill_last_tps = 0.0
        state.prefill_last_tokens = 0
        state.prefill_last_seconds = 0.0
        state.prefill_method = "scout"
        live.update(build_display(state))

        text = context_cache.get(ctx, "") or generate_padding_text(ctx)
        msgs = build_messages(ctx, text)
        request_task = asyncio.create_task(measure_ttft(client, msgs))
        ttft, usage_prompt_tokens = await wait_prefill_task_with_live(
            request_task,
            client,
            base_url,
            engine,
            state,
            live,
            "scout-only prefill: waiting for first token",
        )
        prompt_tokens = int(usage_prompt_tokens or ctx)
        tok_per_sec = (prompt_tokens / ttft) if ttft > 0 else 0.0
        state.prefill_results[ctx] = {
            "method": "scout_only",
            "ttft": ttft,
            "prefill_time": ttft,
            "tok_per_sec": tok_per_sec,
            "prompt_tokens": prompt_tokens,
            "samples": 1,
            "server_tok_per_sec": 0.0,
            "server_prefill_time": 0.0,
            "server_prompt_tokens": 0,
            "server_samples": 0,
            "server_method": "",
            "server_invalid_reason": "",
        }
        snapshot_partial_prefill(state)
        state.prefill_samples_done = 1
        state.prefill_last_tps = tok_per_sec
        state.prefill_last_tokens = prompt_tokens
        state.prefill_last_seconds = ttft
        state.prefill_status = "scout-only prefill complete"
        state.cell_running = False
        state.completed_tests += 1
        state.cell_times.append(time.monotonic() - state.cell_start)
        add_event(state, f"prefill scout-only done ctx={format_context(ctx)} {tok_per_sec:,.0f} tok/s")
        live.update(build_display(state))
        await asyncio.sleep(0.2)

    async def measure_prefill_once(client, ctx: int, messages: list) -> dict:
        """Measure one cold prefill request.

        Primary path uses prompt tokens divided by client-observed
        TTFT. When available, server-side Prometheus prefill counters are kept
        as validation only.
        """
        before_metrics = {}
        before = {}
        server_validation = {
            "server_valid": False,
            "server_prompt_tokens": 0,
            "server_prefill_time": 0.0,
            "server_tok_per_sec": 0.0,
            "server_prefill_count": 0.0,
            "server_request_success": 0.0,
            "server_invalid_reason": "",
        }
        label_filter = 'stage="prefill_forward"' if engine == ENGINE_SGLANG else ""
        collect_server_validation = state.metrics_available and args.prefill_metric in ("auto", "prometheus")
        if collect_server_validation:
            probe_metrics = await scrape_metrics(client, base_url)
            collect_server_validation = (
                has_metric(probe_metrics, metric_name(engine, "prompt_tokens_total"))
                and has_metric(probe_metrics, metric_name(engine, "prefill_time_sum"), label_filter)
            )

        if collect_server_validation:
            before_metrics = await wait_server_idle(
                client,
                base_url,
                engine,
                state=state,
                live=live,
                status="waiting for server idle before cold prefill sample",
            )
            before = prefill_counter_snapshot(before_metrics, engine)

        state.prefill_status = "request in-flight; waiting for first token"
        live.update(build_display(state))
        request_task = asyncio.create_task(measure_ttft(client, messages))
        ttft, usage_prompt_tokens = await wait_prefill_task_with_live(
            request_task,
            client,
            base_url,
            engine,
            state,
            live,
            "request in-flight; waiting for first token",
        )

        if collect_server_validation:
            after_metrics = await wait_server_idle(
                client,
                base_url,
                engine,
                state=state,
                live=live,
                status="waiting for server idle after prefill sample",
            )
            after = prefill_counter_snapshot(after_metrics, engine)
            d = counter_delta(after, before)
            prompt_tokens = int(round(d.get("prompt_tokens_total", 0.0)))
            prefill_seconds = d.get("prefill_sum", 0.0)
            prefill_count = d.get("prefill_count", 0.0)
            request_success = d.get("request_success_total", 0.0)
            valid = (
                prefill_count >= 0.5
                and request_success >= 0.5
                and prompt_tokens > 0
                and prefill_seconds > 0
            )
            if usage_prompt_tokens is not None and prompt_tokens > 0:
                # Treat a different prompt token count as contamination rather
                # than silently producing a bad prefill rate.
                valid = valid and abs(prompt_tokens - usage_prompt_tokens) <= 1
            server_validation.update({
                "server_valid": bool(valid),
                "server_prompt_tokens": prompt_tokens,
                "server_prefill_time": prefill_seconds,
                "server_tok_per_sec": (prompt_tokens / prefill_seconds) if valid else 0.0,
                "server_prefill_count": prefill_count,
                "server_request_success": request_success,
            })
            if not valid:
                server_validation["server_invalid_reason"] = (
                    f"prefill_count={prefill_count}, request_success={request_success}, "
                    f"prompt_tokens={prompt_tokens}, prefill_seconds={prefill_seconds:.6f}"
                )
            if valid:
                tps = prompt_tokens / prefill_seconds
                if args.prefill_metric == "prometheus":
                    state.prefill_samples_done += 1
                    state.prefill_last_tps = tps
                    state.prefill_last_tokens = prompt_tokens
                    state.prefill_last_seconds = prefill_seconds
                    state.prefill_method = "prometheus"
                    state.prefill_status = "server validation sample complete"
                    live.update(build_display(state))
                    return {
                        "method": "prometheus",
                        "ttft": ttft,
                        "prompt_tokens": prompt_tokens,
                        "prefill_time": prefill_seconds,
                        "tok_per_sec": tps,
                        "client_ttft": ttft,
                        "client_prompt_tokens": usage_prompt_tokens or ctx,
                        "client_tok_per_sec": ((usage_prompt_tokens or ctx) / ttft) if ttft > 0 else 0.0,
                        **server_validation,
                    }
            elif args.prefill_metric == "prometheus":
                return {
                    "method": "prometheus",
                    "ttft": ttft,
                    "prompt_tokens": prompt_tokens,
                    "prefill_time": 0,
                    "tok_per_sec": 0,
                    "invalid_reason": server_validation["server_invalid_reason"],
                    **server_validation,
                }

        if args.prefill_metric == "prometheus" and not collect_server_validation:
            return {
                "method": "prometheus",
                "ttft": ttft,
                "prompt_tokens": usage_prompt_tokens or ctx,
                "prefill_time": 0,
                "tok_per_sec": 0,
                "invalid_reason": "Prometheus prefill counters are not available",
                **server_validation,
            }

        client_prompt_tokens = int(usage_prompt_tokens or ctx)
        client_tok_per_sec = (client_prompt_tokens / ttft) if ttft > 0 else 0.0
        state.prefill_samples_done += 1
        state.prefill_last_tps = client_tok_per_sec
        state.prefill_last_tokens = client_prompt_tokens
        state.prefill_last_seconds = ttft
        state.prefill_method = "client"
        state.prefill_status = "client TTFT sample complete"
        live.update(build_display(state))
        return {
            "method": "client",
            "ttft": ttft,
            "prompt_tokens": client_prompt_tokens,
            "prefill_time": ttft,
            "tok_per_sec": client_tok_per_sec,
            "client_ttft": ttft,
            "client_prompt_tokens": client_prompt_tokens,
            "client_tok_per_sec": client_tok_per_sec,
            **server_validation,
        }

    async with httpx.AsyncClient(limits=limits, headers=auth_headers) as client:
        if args.display_mode == "plain":
            live_cm = NullLive()
        else:
            live_cm = Live(
                build_display(state),
                refresh_per_second=max(args.refresh_rate, 0.2),
                console=console,
                screen=args.display_mode == "screen",
            )
        with live_cm as live:

            # === Optional Phase 1: standalone prefill benchmark (C=1, max_tokens=1) ===
            if args.standalone_prefill and standalone_prefill_contexts and not args.skip_prefill:
                # Warmup: trigger CUDA graph compilation before measurements.
                state.prefill_phase = True
                add_event(state, "prefill warmup start")
                state.current_context = 0
                state.cell_running = True
                state.cell_start = time.monotonic()
                state.prefill_status = "warmup: short decode/JIT"
                state.prefill_samples_done = 0
                state.prefill_last_tps = 0.0
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
                warmup_ctx = standalone_prefill_contexts[0]
                state.current_context = warmup_ctx
                state.prefill_status = "warmup: first prefill/JIT"
                live.update(build_display(state))
                warmup_prefix = f"[WARMUP_{run_id}] "
                cached_warmup_text = context_cache.get(warmup_ctx, "") or generate_padding_text(warmup_ctx)
                cached_prefix = f"[BENCH_{run_id}_CTX_{warmup_ctx}] "
                if cached_warmup_text.startswith(cached_prefix):
                    warmup_text = warmup_prefix + cached_warmup_text[len(cached_prefix):]
                else:
                    warmup_text = warmup_prefix + cached_warmup_text
                warmup_msgs = build_messages(warmup_ctx, warmup_text)
                await measure_ttft(client, warmup_msgs)

                # Prefill is ISL / TTFT. No baseline subtraction:
                # TTFT is the client-observed end-to-end prefill estimate.
                baseline_ttft = 0.0
                state.cell_running = False

                # Measure each prefill context: repeat for a fixed duration to collect
                # many samples. Each request uses a unique prefix to avoid cache hits.
                PREFILL_DURATION = args.prefill_duration  # seconds per context size

                for ctx in standalone_prefill_contexts:
                    add_event(state, f"prefill start ctx={format_context(ctx)}")
                    state.current_concurrency = 1
                    state.current_context = ctx
                    state.cell_running = True
                    state.cell_start = time.monotonic()
                    state.cell_duration = PREFILL_DURATION
                    state.prefill_status = "starting cold prefill samples"
                    state.prefill_samples_done = 0
                    state.prefill_last_tps = 0.0
                    state.prefill_last_tokens = 0
                    state.prefill_last_seconds = 0.0
                    state.prefill_method = ""
                    live.update(build_display(state))

                    prefill_samples = []
                    orig_prefix_len = len(f"[BENCH_{run_id}_CTX_{ctx}] ")
                    r = 0
                    skipped = False
                    while True:
                        elapsed = time.monotonic() - state.cell_start
                        if elapsed >= PREFILL_DURATION and (len(prefill_samples) >= 2 or r >= 1):
                            break
                        if _skip_event.is_set():
                            _skip_event.clear()
                            skipped = True
                            break
                        # Unique prefix per iteration → no prefix cache hit
                        prefix = f"[BENCH_{run_id}_CTX_{ctx}_R{r}] "
                        variant_text = prefix + context_cache[ctx][orig_prefix_len:]
                        msgs = build_messages(ctx, variant_text)
                        sample = await measure_prefill_once(client, ctx, msgs)
                        if sample.get("tok_per_sec", 0) > 0:
                            prefill_samples.append(sample)
                        r += 1
                        live.update(build_display(state))

                    if skipped or not prefill_samples:
                        state.prefill_results[ctx] = {
                            "ttft": 0, "prefill_time": 0, "tok_per_sec": 0,
                            "baseline": baseline_ttft, "samples": 0,
                            "prompt_tokens": 0, "skipped": True,
                        }
                        snapshot_partial_prefill(state)
                    else:
                        sample_set = prefill_samples
                        method = sample_set[0].get("method", "client")
                        baseline = 0.0
                        raw_ttft = median(s["ttft"] for s in sample_set)
                        prefill_time = median(s["prefill_time"] for s in sample_set)
                        token_count = int(round(median(s["prompt_tokens"] for s in sample_set)))
                        tok_per_sec = median(s["tok_per_sec"] for s in sample_set)
                        server_samples = [s for s in sample_set if s.get("server_valid")]
                        server_tps = median(s["server_tok_per_sec"] for s in server_samples) if server_samples else 0.0
                        server_prefill_time = median(s["server_prefill_time"] for s in server_samples) if server_samples else 0.0
                        server_prompt_tokens = int(round(median(s["server_prompt_tokens"] for s in server_samples))) if server_samples else 0
                        invalid_reasons = [
                            s.get("server_invalid_reason", "")
                            for s in sample_set
                            if s.get("server_invalid_reason")
                        ]

                        state.prefill_results[ctx] = {
                            "ttft": raw_ttft,
                            "prefill_time": prefill_time,
                            "tok_per_sec": tok_per_sec,
                            "baseline": baseline,
                            "samples": len(sample_set),
                            "prompt_tokens": token_count,
                            "method": method,
                            "server_tok_per_sec": server_tps,
                            "server_prefill_time": server_prefill_time,
                            "server_prompt_tokens": server_prompt_tokens,
                            "server_samples": len(server_samples),
                            "server_method": "prometheus" if server_samples else "",
                            "server_invalid_reason": invalid_reasons[0] if invalid_reasons else "",
                        }
                        snapshot_partial_prefill(state)

                    state.cell_running = False
                    if skipped or not prefill_samples:
                        add_event(state, f"prefill skipped ctx={format_context(ctx)}")
                    else:
                        add_event(state, f"prefill done ctx={format_context(ctx)} {tok_per_sec:,.0f} tok/s")
                    state.completed_tests += 1
                    cell_time = time.monotonic() - state.cell_start
                    state.cell_times.append(cell_time)
                    live.update(build_display(state))
                    await asyncio.sleep(1.0)

            # === Warmup probe: ensure CUDA graphs / JIT compiled before decode ===
            # This runs whenever the standalone prefill phase did not already do
            # it. Sends a short request to trigger kernel compilation so the
            # first decode cell isn't contaminated.
            if not standalone_prefill_contexts or args.skip_prefill:
                state.prefill_phase = True
                add_event(state, "decode warmup start")
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

            if prefill_contexts and not args.skip_prefill and not args.standalone_prefill:
                # Keep default prefill rows in a consistent warm state. Without
                # this, the first reported 8k row can differ depending on
                # whether token calibration had to run a hidden 8k probe or was
                # loaded from the calibration cache.
                warmup_ctx = min(min(prefill_contexts), parse_token_value("8k"))
                add_event(state, f"prefill warmup start ctx={format_context(warmup_ctx)}")
                state.prefill_phase = True
                state.current_context = warmup_ctx
                state.current_concurrency = 1
                state.cell_running = True
                state.cell_start = time.monotonic()
                state.prefill_status = "warmup: prefill/JIT before measured rows"
                state.prefill_samples_done = 0
                state.prefill_last_tps = 0.0
                state.prefill_last_tokens = 0
                state.prefill_last_seconds = 0.0
                state.prefill_method = "warmup"
                live.update(build_display(state))

                cached_warmup_text = context_cache.get(warmup_ctx) or generate_padding_text(warmup_ctx)
                cached_prefix = f"[BENCH_{run_id}_CTX_{warmup_ctx}] "
                warmup_prefix = f"[WARMUP_{run_id}_PREFILL] "
                if cached_warmup_text.startswith(cached_prefix):
                    warmup_text = warmup_prefix + cached_warmup_text[len(cached_prefix):]
                else:
                    warmup_text = warmup_prefix + cached_warmup_text
                warmup_msgs = build_messages(warmup_ctx, warmup_text)
                warmup_task = asyncio.create_task(measure_ttft(client, warmup_msgs))
                await wait_prefill_task_with_live(
                    warmup_task,
                    client,
                    base_url,
                    engine,
                    state,
                    live,
                    "warmup: prefill/JIT before measured rows",
                )
                state.cell_running = False
                state.prefill_phase = False
                add_event(state, f"prefill warmup done ctx={format_context(warmup_ctx)}")
                live.update(build_display(state))

            if prefill_scout_only_contexts and not args.skip_prefill and not args.standalone_prefill:
                for ctx in prefill_scout_only_contexts:
                    if _skip_event.is_set():
                        _skip_event.clear()
                        break
                    await measure_prefill_scout_only(client, ctx, live)

            # === Phase 2: Decode benchmark (cached prefill, pure decode speed) ===
            # Cache warming per context is handled by scout request in run_one_cell.
            # Order: 1) first column (C=first, all ctx) — baseline
            #        2) first row (ctx=first, all C) — concurrency scaling
            #        3) rest row by row
            state.prefill_phase = False
            first_conc = concurrency_levels[0]
            first_ctx = context_lengths[0]

            # Optional hidden decode warmup. The short probe above is enough for
            # basic JIT compilation, but long-running speculative stacks can still
            # under-report the first measured decode cell. This runs the first
            # decode shape without recording it, then clears the transient TUI state.
            if args.decode_warmup_seconds > 0:
                await run_one_cell(
                    client=client,
                    base_url=base_url,
                    concurrency=first_conc,
                    context_tokens=first_ctx,
                    context_text=context_cache[first_ctx],
                    duration=args.decode_warmup_seconds,
                    max_tokens=args.max_tokens,
                    model=args.model,
                    state=state,
                    live=live,
                    engine=engine,
                    auth_headers=auth_headers,
                    ignore_eos=not args.respect_eos,
                    request_count=0,
                    warmup_request_count=0,
                )
                state.results.pop((first_ctx, first_conc), None)
                state.errors.pop((first_ctx, first_conc), None)
                state.queue_info.pop((first_ctx, first_conc), None)
                state.cell_running = False
                live.update(build_display(state))
                await asyncio.sleep(2.0)

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
                        needed = conc * (ctx + args.max_tokens)
                        missing = max(0, needed - args.max_total_tokens)
                        state.results[(ctx, conc)] = -1  # mark as skipped
                        cell = CellResult(
                            concurrency=conc,
                            context_tokens=ctx,
                            aggregate_tps=-1,
                            total_tokens=needed,
                            capacity_limited=True,
                            timeout_reason=format_token_budget(missing),
                        )
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
                            ignore_eos=not args.respect_eos,
                            request_count=args.request_count,
                            warmup_request_count=args.warmup_request_count,
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

            if args.run_burst and args.request_count == 0:
                console.print("[cyan]Phase 3: Burst / E2E decode request burst[/cyan]")
                state.benchmark_mode = "request-count"
                for ctx, conc in test_order:
                    if args.max_total_tokens > 0 and _should_skip(ctx, conc):
                        needed = conc * (ctx + args.max_tokens)
                        missing = max(0, needed - args.max_total_tokens)
                        state.results[(ctx, conc)] = -1
                        cell = CellResult(
                            concurrency=conc,
                            context_tokens=ctx,
                            benchmark_mode="burst-e2e",
                            aggregate_tps=-1,
                            total_tokens=needed,
                            capacity_limited=True,
                            timeout_reason=format_token_budget(missing),
                        )
                        burst_results.append(cell)
                        state.completed_tests += 1
                        live.update(build_display(state))
                        continue

                    measured_requests = (
                        args.burst_request_count
                        if args.burst_request_count > 0
                        else max(conc, conc * args.burst_requests_per_concurrency)
                    )
                    warmup_requests = (
                        args.burst_warmup_request_count
                        if args.burst_warmup_request_count > 0
                        else conc
                    )
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
                            ignore_eos=not args.respect_eos,
                            request_count=measured_requests,
                            warmup_request_count=warmup_requests,
                        )
                        result.benchmark_mode = "burst-e2e"
                        if result.aggregate_tps == -2:
                            state.results[(ctx, conc)] = -2
                        burst_results.append(result)
                    except Exception as e:
                        console.print(f"[red]Burst cell C={conc} ctx={format_context(ctx)} failed: {e}[/red]")
                        cell = CellResult(
                            concurrency=conc,
                            context_tokens=ctx,
                            benchmark_mode="burst-e2e",
                        )
                        burst_results.append(cell)
                        state.results[(ctx, conc)] = 0.0
                        state.errors[(ctx, conc)] = conc

                    cell_time = time.monotonic() - cell_start
                    state.cell_times.append(cell_time)
                    state.completed_tests += 1
                    live.update(build_display(state))
                    await asyncio.sleep(1.0)

    setattr(args, "event_log", list(state.events))
    return all_results, burst_results, state.prefill_results, engine


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------

def print_final_results(results: list, concurrency_levels: list, context_lengths: list,
                        console: Console, prefill_results: dict = None,
                        show_capacity_limited_values: bool = False,
                        burst_results: list = None):
    console.print("\n")
    console.print(f"[dim]llm-decode-bench v{VERSION}[/dim]")

    def needs_effective_note(r: CellResult, requested: int) -> bool:
        if getattr(r, "benchmark_mode", "") == "request-count":
            # Finite burst cells naturally drain near the end, so average
            # running_reqs is not an effective-concurrency signal there.
            return bool(r.capacity_limited or r.avg_queue_reqs > 0)
        return (
            r.capacity_limited
            or r.avg_queue_reqs > 0
            or (
                requested > 1
                and r.avg_running_reqs > 0
                and r.avg_running_reqs < requested * 0.98
            )
        )
    decode_modes = {getattr(r, "benchmark_mode", "duration") for r in results if r.aggregate_tps >= 0}
    request_count_mode = decode_modes == {"request-count"}
    sustained_mode = not request_count_mode

    # Prefill table
    if prefill_results:
        first_prefill = next(iter(prefill_results.values()), {})
        method = first_prefill.get("method", "client")
        if method == "prometheus":
            title = "Prefill Speed (C=1, Prometheus server prefill counter)"
        elif method in ("integrated_scout", "scout_only"):
            title = "Prefill Speed (scout requests, client ISL / TTFT)"
        else:
            title = "Prefill Speed (C=1, client ISL / TTFT)"
        pt = Table(
            title=render_title(title),
            title_justify="left",
            box=TABLE_BOX,
            border_style=FRAME_BORDER,
            header_style=f"bold {PHOSPHOR_DIM}",
        )
        pt.add_column("Context", style=f"bold {PHOSPHOR_SOFT}")
        pt.add_column("Tokens", justify="right")
        pt.add_column("TTFT (s)", justify="right")
        pt.add_column("Client tok/s", justify="right")
        pt.add_column("Server tok/s", justify="right")
        pt.add_column("N", justify="right")
        for ctx in sorted(prefill_results.keys()):
            pr = prefill_results[ctx]
            ptok = pr.get('prompt_tokens', ctx)
            server_tps = pr.get("server_tok_per_sec", 0)
            server_samples = pr.get("server_samples", 0)
            server_cell = (
                f"{server_tps:,.0f} ({server_samples})"
                if server_tps > 0 and server_samples > 0
                else "—"
            )
            pt.add_row(
                format_context(ctx),
                f"{ptok:,}" if ptok != ctx else f"~{ctx:,}",
                f"{pr['ttft']:.2f}",
                f"{pr['tok_per_sec']:,.0f}",
                server_cell,
                str(pr.get('samples', '?')),
            )
        console.print(pt)
        console.print(
            "[dim]Client tok/s = prompt_tokens / TTFT. "
            "Integrated scout rows come from the prefix-cache scout request that decode needs anyway. "
            "Server tok/s is optional Prometheus validation when the engine exports "
            "prefill counters and the exact counter delta is uncontaminated.[/dim]"
        )
        console.print()

    if results:
        if sustained_mode:
            console.print(Panel(
                "[bold]Sustained Decode[/bold]\n"
                "Steady-state decode throughput after the engine has admitted the requested concurrency and passed warmup. "
                "Use this as the main tuning/regression signal for kernels, NCCL, DCP, MTP, and scheduler changes.",
                title=render_title("Phase 2"),
                box=PANEL_BOX,
                border_style=FRAME_BORDER,
            ))
        else:
            console.print(Panel(
                "[bold]Burst / E2E Decode[/bold]\n"
                "Burst / E2E-only mode from --request-count. Aggregate throughput is completed output tokens divided by profiling wall time.",
                title=render_title("Decode"),
                box=PANEL_BOX,
                border_style=FRAME_BORDER,
            ))

    # Aggregate throughput table
    inline_client_stats = any(getattr(r, "request_count", 0) > 0 for r in results)
    inline_final_client_stats = (
        inline_client_stats
        and (14 + len(concurrency_levels) * 18) <= max(80, console.width)
    )
    table = Table(
        title=render_title("Aggregate tok/s", " + TTFT/ITL" if inline_client_stats else ""),
        title_justify="left",
        box=REPORT_BOX,
        border_style=SUBTLE_BORDER,
        header_style=f"bold {PHOSPHOR_DIM}",
    )
    table.add_column("ctx \\ conc", style=f"bold {PHOSPHOR_SOFT}")
    for conc in concurrency_levels:
        table.add_column(str(conc), justify="right")

    result_map = {(r.context_tokens, r.concurrency): r for r in results}
    any_limited = any(needs_effective_note(r, r.concurrency) for r in results if r.aggregate_tps >= 0)
    any_kv_limited = any(r.capacity_limited for r in results)

    for ctx in context_lengths:
        row = [format_context(ctx)]
        for conc in concurrency_levels:
            r = result_map.get((ctx, conc))
            if r and r.aggregate_tps < 0:
                if r.capacity_limited:
                    row.append(capacity_limit_cell())
                else:
                    row.append("skip")
            elif r:
                if r.capacity_limited and not show_capacity_limited_values:
                    val = capacity_limit_cell()
                else:
                    val = f"{r.aggregate_tps:.1f}"
                if r.num_errors > 0:
                    val += f" ({r.num_errors}e)"
                if needs_effective_note(r, conc):
                    mark = "*" if r.capacity_limited else ""
                    val += f" ({r.avg_running_reqs:.0f}/{conc}){mark}"
                detail = compact_client_cell_detail(compact_client_info_from_cell(r))
                if inline_client_stats and detail:
                    if inline_final_client_stats:
                        val += f" [dim]{detail}[/dim]"
                    else:
                        val += f"\n[dim]{detail}[/dim]"
                row.append(val)
            else:
                row.append("-")
        table.add_row(*row)

    console.print(table)
    if request_count_mode:
        console.print(
            "[dim]Burst / E2E-only mode: aggregate tok/s = completed output tokens / profiling wall time, "
            "using OpenAI stream usage. Prometheus throughput is kept in JSON as server_gen_throughput.[/dim]"
        )
    else:
        console.print(
            "[dim]Sustained Decode: aggregate tok/s uses OpenAI stream usage by default "
            "(continuous completion_tokens when the server supports it). Prometheus is kept as validation/scheduler data.[/dim]"
        )
        sources = sorted({
            r.aggregate_source
            for r in results
            if r.aggregate_tps >= 0 and getattr(r, "aggregate_source", "")
        })
        if sources:
            console.print(f"[dim]Aggregate source(s): {', '.join(sources)}[/dim]")
    if any_kv_limited:
        console.print(f"[dim]{CAPACITY_LIMIT_MARK} = skipped/hidden because the cell does not fit in KV cache; exact deficit is kept in JSON timeout_reason[/dim]")
    if any_limited:
        console.print("[dim](X/Y) = avg running / requested concurrency from Prometheus; * = capacity-limited or warmup timed out[/dim]")

    # Per-request avg tok/s table
    table2 = Table(
        title=render_title("Per-Request tok/s"),
        title_justify="left",
        box=REPORT_BOX,
        border_style=SUBTLE_BORDER,
        header_style=f"bold {PHOSPHOR_DIM}",
    )
    table2.add_column("ctx \\ conc", style=f"bold {PHOSPHOR_SOFT}")
    for conc in concurrency_levels:
        table2.add_column(str(conc), justify="right")

    for ctx in context_lengths:
        row = [format_context(ctx)]
        for conc in concurrency_levels:
            r = result_map.get((ctx, conc))
            if r and r.aggregate_tps < 0:
                if r.capacity_limited:
                    row.append(capacity_limit_cell())
                else:
                    row.append("skip")
            elif r and r.per_request_avg_tps > 0:
                if r.capacity_limited and not show_capacity_limited_values:
                    val = capacity_limit_cell()
                else:
                    val = f"{r.per_request_avg_tps:.1f}"
                if needs_effective_note(r, conc):
                    mark = "*" if r.capacity_limited else ""
                    val += f" ({r.avg_running_reqs:.0f}/{conc}){mark}"
                row.append(val)
            else:
                row.append("-")
        table2.add_row(*row)

    console.print(table2)

    # Client-visible latency matrices. Keep these in ctx x conc form so they
    # remain comparable to the throughput matrix. Full request-level samples are
    # preserved in JSON for detailed distribution analysis.
    if any(getattr(r, "request_count", 0) > 0 for r in results):
        def fmt_compact_ms(seconds: float) -> str:
            if not seconds or seconds <= 0:
                return "—"
            ms = seconds * 1000
            if ms >= 10000:
                return f"{ms / 1000:.1f}k"
            if ms >= 1000:
                return f"{ms / 1000:.2g}k"
            if ms >= 100:
                return f"{ms:.0f}"
            if ms >= 10:
                return f"{ms:.1f}"
            return f"{ms:.2f}"

        def fmt_compact_rate(value: float) -> str:
            if not value or value <= 0:
                return "—"
            if value >= 1000:
                return f"{value / 1000:.2g}k"
            if value >= 100:
                return f"{value:.0f}"
            if value >= 10:
                return f"{value:.1f}"
            return f"{value:.2f}"

        def client_cell(r: CellResult, kind: str) -> str:
            if r.aggregate_tps < 0:
                return capacity_limit_cell() if r.capacity_limited else "skip"
            if r.request_count <= 0:
                return "-"
            if kind == "latency":
                return (
                    f"{fmt_compact_ms(r.request_latency_p50)}/"
                    f"{fmt_compact_ms(r.request_latency_p90)}"
                )
            return "-"

        def print_client_matrix(title: str, kind: str) -> None:
            matrix = Table(
                title=render_title(title),
                title_justify="left",
                box=REPORT_BOX,
                border_style=SUBTLE_BORDER,
                header_style=f"bold {PHOSPHOR_DIM}",
            )
            matrix.add_column("ctx \\ conc", style=f"bold {PHOSPHOR_SOFT}", no_wrap=True)
            for conc in concurrency_levels:
                matrix.add_column(str(conc), justify="right", no_wrap=True)
            for ctx in context_lengths:
                row = [format_context(ctx)]
                for conc in concurrency_levels:
                    r = result_map.get((ctx, conc))
                    row.append(client_cell(r, kind) if r else "-")
                matrix.add_row(*row)
            console.print(matrix)

        print_client_matrix("Client request latency: p50 / p90 ms", "latency")
        console.print(
            "[dim]Aggregate cells show dim detail as TTFT ms / ITL ms for the same ctx/conc coordinate. "
            "ITL is computed from observed generated tokens, including streams stopped at the measurement boundary; "
            "a missing ITL means no stream produced at least two measured output tokens. "
            "Per-request tok/s and request latency are shown in separate per-cell matrices. "
            "Completion/sample counts and full request-level distributions remain in JSON under request_samples.[/dim]"
        )
        if request_count_mode:
            console.print(
                "[dim]Request-count mode: aggregate tok/s = completed output tokens / profiling wall time. "
                "Client ITL=(last_token_time-first_token_time)/(output_tokens-1), user tok/s=1/ITL.[/dim]"
            )
        else:
            console.print(
                "[dim]Sustained mode: client latency metrics explain request UX variance; aggregate tok/s remains the primary "
                "throughput signal. ITL=(last_token_time-first_token_time)/(output_tokens-1), user tok/s=1/ITL.[/dim]"
            )

    hardware_rows = [
        r for r in list(results) + list(burst_results or [])
        if r.aggregate_tps >= 0 and getattr(r, "hardware_summary", {})
    ]
    if hardware_rows:
        hw_table = Table(
            title=render_title("Hardware Summary"),
            title_justify="left",
            box=REPORT_BOX,
            border_style=SUBTLE_BORDER,
            header_style=f"bold {PHOSPHOR_DIM}",
        )
        hw_table.add_column("ctx", style=f"bold {PHOSPHOR_SOFT}", no_wrap=True)
        hw_table.add_column("C", justify="right", no_wrap=True)
        hw_table.add_column("mode", no_wrap=True)
        hw_table.add_column("GPU avg/max", justify="right", no_wrap=True)
        hw_table.add_column("Mem avg", justify="right", no_wrap=True)
        hw_table.add_column("W avg/max", justify="right", no_wrap=True)
        hw_table.add_column("T max", justify="right", no_wrap=True)
        hw_table.add_column("VRAM", justify="right", no_wrap=True)
        hw_table.add_column("PCIe rx/tx avg", justify="right", no_wrap=True)
        for r in hardware_rows:
            hw = r.hardware_summary
            hw_table.add_row(
                format_context(r.context_tokens),
                str(r.concurrency),
                "burst" if getattr(r, "benchmark_mode", "") in ("request-count", "burst-e2e") else "sustain",
                f"{hw.get('gpu_util_avg_pct', 0):.0f}/{hw.get('gpu_util_max_pct', 0):.0f}%",
                f"{hw.get('mem_util_avg_pct', 0):.0f}%",
                f"{hw.get('power_total_avg_w', 0):.0f}/{hw.get('power_total_max_w', 0):.0f}",
                f"{hw.get('temp_max_c', 0):.0f}C",
                f"{hw.get('vram_used_avg_pct', 0):.1f}%",
                f"{hw.get('pcie_rx_avg_mb_s', 0):.0f}/{hw.get('pcie_tx_avg_mb_s', 0):.0f}",
            )
        console.print(hw_table)
        console.print(
            "[dim]Hardware summary is sampled from nvidia-smi during the measured part of each cell. "
            "PCIe rx/tx is MB/s and is a coarse live diagnostic, not a per-kernel NCCL profiler.[/dim]"
        )

    if burst_results:
        console.print()
        console.print(Panel(
            "[bold]Burst / E2E Decode[/bold]\n"
            "Short client-facing burst probe run after sustained decode. It sends a fixed number of measured requests, "
            "waits for completion, and reports completed output tokens divided by wall time.",
            title=render_title("Phase 3"),
            box=PANEL_BOX,
            border_style=FRAME_BORDER,
        ))
        burst_map = {(r.context_tokens, r.concurrency): r for r in burst_results}
        burst_table = Table(
            title=render_title("Burst/E2E tok/s"),
            title_justify="left",
            box=REPORT_BOX,
            border_style=SUBTLE_BORDER,
            header_style=f"bold {PHOSPHOR_DIM}",
        )
        burst_table.add_column("ctx \\ conc", style=f"bold {PHOSPHOR_SOFT}")
        for conc in concurrency_levels:
            burst_table.add_column(str(conc), justify="right")
        for ctx in context_lengths:
            row = [format_context(ctx)]
            for conc in concurrency_levels:
                r = burst_map.get((ctx, conc))
                if r and r.aggregate_tps < 0:
                    row.append(capacity_limit_cell() if r.capacity_limited else "skip")
                elif r:
                    val = f"{r.aggregate_tps:.1f}"
                    if r.request_count_target:
                        val += f" ({r.completed_request_count}/{r.request_count_target})"
                    row.append(val)
                else:
                    row.append("-")
            burst_table.add_row(*row)
        console.print(burst_table)

        burst_client = Table(
            title=render_title("Burst/E2E p50"),
            title_justify="left",
            box=REPORT_BOX,
            border_style=SUBTLE_BORDER,
            header_style=f"bold {PHOSPHOR_DIM}",
        )
        burst_client.add_column("ctx", style=f"bold {PHOSPHOR_SOFT}", no_wrap=True)
        burst_client.add_column("C", justify="right", no_wrap=True)
        burst_client.add_column("Reqs", justify="right", no_wrap=True)
        burst_client.add_column("TTFT ms", justify="right", no_wrap=True)
        burst_client.add_column("ITL ms", justify="right", no_wrap=True)
        burst_client.add_column("Latency ms", justify="right", no_wrap=True)
        burst_client.add_column("tok/s/user", justify="right", no_wrap=True)
        for ctx in context_lengths:
            for conc in concurrency_levels:
                r = burst_map.get((ctx, conc))
                if not r or r.aggregate_tps < 0:
                    continue
                burst_client.add_row(
                    format_context(ctx),
                    str(conc),
                    f"{r.completed_request_count}/{r.request_count_target or r.request_count}",
                    format_ms_value(r.ttft_p50),
                    format_ms_value(r.inter_token_latency_p50),
                    format_ms_value(r.request_latency_p50),
                    format_rate_value(r.output_tps_per_user_p50),
                )
        console.print(burst_client)
        console.print(
            "[dim]Interpretation: sustained decode isolates steady engine throughput; Burst / E2E includes client-visible "
            "batch admission and completion behavior for a finite request burst. Compare them separately.[/dim]"
        )
    elif results and not request_count_mode:
        console.print()
        console.print(Panel(
            "[bold]Burst / E2E Decode[/bold]\n"
            "Not run. Re-run with --run-burst to append a finite client-facing request burst after Sustained Decode. "
            "This is intentionally disabled by default because it adds another full decode matrix.",
            title=render_title("Phase 3"),
            box=PANEL_BOX,
            border_style=SUBTLE_BORDER,
        ))

    if prefill_results or results:
        console.print()
        console.print(Panel(
            "Primary matrices repeated last so the important numbers are visible without scrolling back through diagnostics.",
            title=render_title("Primary Summary"),
            box=PANEL_BOX,
            border_style=FRAME_BORDER,
        ))

    if prefill_results:
        summary_prefill = Table(
            title=render_title("Prefill tok/s"),
            title_justify="left",
            box=REPORT_BOX,
            border_style=SUBTLE_BORDER,
            header_style=f"bold {PHOSPHOR_DIM}",
        )
        summary_prefill.add_column("ctx", style=f"bold {PHOSPHOR_SOFT}", no_wrap=True)
        summary_prefill.add_column("tokens", justify="right", no_wrap=True)
        summary_prefill.add_column("TTFT s", justify="right", no_wrap=True)
        summary_prefill.add_column("tok/s", justify="right", no_wrap=True)
        summary_prefill.add_column("N", justify="right", no_wrap=True)
        for ctx in sorted(prefill_results.keys()):
            pr = prefill_results[ctx]
            if pr.get("skipped"):
                summary_prefill.add_row(format_context(ctx), "skip", "—", "—", "0")
                continue
            summary_prefill.add_row(
                format_context(ctx),
                f"{pr.get('prompt_tokens', ctx):,}",
                f"{pr.get('ttft', 0):.2f}",
                f"{pr.get('tok_per_sec', 0):,.0f}",
                str(pr.get("samples", "?")),
            )
        console.print(summary_prefill)

    if results:
        summary_decode = Table(
            title=render_title("Aggregate decode tok/s"),
            title_justify="left",
            box=REPORT_BOX,
            border_style=SUBTLE_BORDER,
            header_style=f"bold {PHOSPHOR_DIM}",
        )
        summary_decode.add_column("ctx \\ conc", style=f"bold {PHOSPHOR_SOFT}", no_wrap=True)
        for conc in concurrency_levels:
            summary_decode.add_column(str(conc), justify="right", no_wrap=True)
        for ctx in context_lengths:
            row = [format_context(ctx)]
            for conc in concurrency_levels:
                r = result_map.get((ctx, conc))
                if r and r.aggregate_tps < 0:
                    row.append(capacity_limit_cell() if r.capacity_limited else "skip")
                elif r:
                    if r.capacity_limited and not show_capacity_limited_values:
                        val = capacity_limit_cell()
                    else:
                        val = f"{r.aggregate_tps:.1f}"
                    if r.num_errors > 0:
                        val += f" ({r.num_errors}e)"
                    if needs_effective_note(r, conc):
                        mark = "*" if r.capacity_limited else ""
                        val += f" ({r.avg_running_reqs:.0f}/{conc}){mark}"
                    row.append(val)
                else:
                    row.append("-")
            summary_decode.add_row(*row)
        console.print(summary_decode)


def save_results(results: list, args, filepath: str, prefill_results: dict = None,
                 engine: str = "", burst_results: list = None):
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
    burst_summary = {}
    actual_burst_results = [r for r in (burst_results or []) if r.aggregate_tps >= 0]
    for r in actual_burst_results:
        ctx_key = str(r.context_tokens)
        if ctx_key not in burst_summary:
            burst_summary[ctx_key] = {}
        burst_summary[ctx_key][str(r.concurrency)] = r.aggregate_tps

    # Prefill summary
    prefill_summary = {}
    if prefill_results:
        for ctx, pr in sorted(prefill_results.items()):
            prefill_summary[str(ctx)] = {
                "ttft_seconds": round(pr["ttft"], 3),
                "prefill_seconds": round(pr.get("prefill_time", pr["ttft"]), 3),
                "tok_per_sec": round(pr["tok_per_sec"], 0),
                "client_ttft_seconds": round(pr.get("ttft", 0), 3),
                "client_tok_per_sec": round(pr["tok_per_sec"], 0),
                "prompt_tokens": pr.get("prompt_tokens", ctx),
                "samples": pr.get("samples", 0),
                "method": pr.get("method", "client"),
                "server_validation": {
                    "method": pr.get("server_method", ""),
                    "tok_per_sec": round(pr.get("server_tok_per_sec", 0), 0),
                    "prefill_seconds": round(pr.get("server_prefill_time", 0), 3),
                    "prompt_tokens": pr.get("server_prompt_tokens", 0),
                    "samples": pr.get("server_samples", 0),
                    "invalid_reason": pr.get("server_invalid_reason", ""),
                },
            }

    output = {
        "metadata": {
            "version": VERSION,
            "engine": engine,
            "model": args.model,
            "server": args.host if args.host.startswith("http") else f"{args.host}:{args.port or 5000}",
            "timestamp": datetime.now().isoformat(),
            "decode_mode": "request-count" if getattr(args, "request_count", 0) > 0 else "duration",
            "primary_decode_layer": (
                "burst_e2e_decode"
                if getattr(args, "request_count", 0) > 0
                else "sustained_decode"
            ),
            "duration_per_test": args.duration,
            "request_count": getattr(args, "request_count", 0),
            "warmup_request_count": getattr(args, "warmup_request_count", 0),
            "run_burst": getattr(args, "run_burst", False),
            "prefill_mode": (
                "skipped" if getattr(args, "skip_prefill", False)
                else "standalone_cold" if getattr(args, "standalone_prefill", False)
                else "integrated_decode_scout"
            ),
            "standalone_prefill": getattr(args, "standalone_prefill", False),
            "skip_prefill": getattr(args, "skip_prefill", False),
            "burst_e2e_status": (
                "not_run_use_--run-burst"
                if not getattr(args, "run_burst", False) and getattr(args, "request_count", 0) == 0
                else "enabled"
            ),
            "burst_request_count": getattr(args, "burst_request_count", 0),
            "burst_warmup_request_count": getattr(args, "burst_warmup_request_count", 0),
            "burst_requests_per_concurrency": getattr(args, "burst_requests_per_concurrency", 5),
            "decode_warmup_seconds": getattr(args, "decode_warmup_seconds", 0),
            "show_capacity_limited_values": getattr(args, "show_capacity_limited_values", False),
            "max_tokens": args.max_tokens,
            "ignore_eos": not getattr(args, "respect_eos", False),
            "max_total_tokens": args.max_total_tokens,
            "dcp_size": getattr(args, "dcp_size", 1),
            "metrics_available": getattr(args, "metrics_available", True),
            "metrics_warning": getattr(args, "metrics_warning", ""),
            "concurrency_levels": concurrency_levels,
            "context_lengths": context_lengths,
            "startup_diagnostics_available": bool(getattr(args, "startup_diagnostics", {})),
        },
        "startup_diagnostics": getattr(args, "startup_diagnostics", {}),
        "event_log": getattr(args, "event_log", []),
        "prefill": prefill_summary,
        "results": [asdict(r) for r in actual_results],
        "summary_table": summary,
        "burst_results": [asdict(r) for r in actual_burst_results],
        "burst_summary_table": burst_summary,
        "methodology": {
            "prefill": {
                "name": "Prefill",
                "present": bool(prefill_summary),
                "mode": (
                    "skipped" if getattr(args, "skip_prefill", False)
                    else "standalone_cold" if getattr(args, "standalone_prefill", False)
                    else "integrated_decode_scout"
                ),
                "formula": "prompt_tokens / TTFT",
                "notes": (
                    "Default mode records the required decode scout request for each "
                    "non-zero decode context, so normal runs do not pay for a separate "
                    "prefill phase. Standalone mode repeats cold-prefill samples. "
                    "Prometheus prefill counters, when available and uncontaminated, "
                    "are stored as validation."
                ),
            },
            "sustained_decode": {
                "name": "Sustained Decode",
                "present": getattr(args, "request_count", 0) == 0 and bool(actual_results),
                "formula": (
                    "OpenAI stream usage completion_tokens per measured window; "
                    "client chunk fallback only when continuous usage is unavailable"
                ),
                "notes": (
                    "Duration-based steady-state cell after warmup. This is the "
                    "main tuning/regression signal for kernels, NCCL, DCP, MTP, and scheduling. "
                    "Prometheus metrics are stored as validation and scheduler state, not the default headline."
                ),
            },
            "burst_e2e_decode": {
                "name": "Burst / E2E Decode",
                "present": bool(actual_burst_results) or getattr(args, "request_count", 0) > 0,
                "status": (
                    "not run; use --run-burst"
                    if not actual_burst_results and getattr(args, "request_count", 0) == 0
                    else "present"
                ),
                "formula": "sum(completion_tokens) / profiling_wall_time",
                "notes": (
                    "Finite client-facing request burst using OpenAI stream usage. "
                    "It includes request admission, scheduling, prefill/cache behavior, and completion."
                ),
            },
        },
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Global events set by keyboard listener.
_skip_event = threading.Event()
_quit_event = threading.Event()
_original_term_settings = None


def _keyboard_listener():
    """Background thread: listen for 's' skip and 'q' graceful stop."""
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
                elif ch.lower() == "q" or ch == "\x03":
                    if _quit_event.is_set():
                        continue
                    _quit_event.set()
                    _restore_terminal()
                    os.kill(os.getpid(), signal.SIGINT)
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
        "--request-count", type=int, default=0,
        help="Burst / E2E-only decode mode: send exactly this many measured "
             "requests per cell and wait for all to finish. 0 keeps the default "
             "duration-based steady-state mode. (default: 0)"
    )
    parser.add_argument(
        "--warmup-request-count", type=int, default=0,
        help="When --request-count is set, send and discard this many warmup "
             "requests before each measured cell. (default: 0)"
    )
    parser.add_argument(
        "--run-burst", action="store_true",
        help="After the default sustained decode matrix, run an additional short "
             "Burst / E2E request-count matrix. Ignored when --request-count is set."
    )
    parser.add_argument(
        "--burst-request-count", type=int, default=0,
        help="Measured requests per Burst / E2E cell. 0 = auto, using "
             "concurrency × --burst-requests-per-concurrency. (default: 0)"
    )
    parser.add_argument(
        "--burst-warmup-request-count", type=int, default=0,
        help="Warmup requests per Burst / E2E cell. 0 = auto, using concurrency. "
             "(default: 0)"
    )
    parser.add_argument(
        "--burst-requests-per-concurrency", type=int, default=5,
        help="Auto Burst / E2E measured request multiplier when --burst-request-count=0. "
             "For C=10 and default 5, the burst sends 50 measured requests. (default: 5)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Max tokens to generate per request (default: 2048)"
    )
    parser.add_argument(
        "--decode-warmup-seconds", type=float, default=0.0,
        help="Hidden decode warmup duration before measured decode cells. "
             "Use 20 for short exact decode matrices on speculative stacks."
    )
    parser.add_argument(
        "--prefill-duration", type=float, default=10.0,
        help="Duration per standalone prefill context in seconds. Only used with "
             "--standalone-prefill. (default: 10)"
    )
    parser.add_argument(
        "--prefill-contexts", default="8k,64k,128k",
        help="Comma-separated prefill context lengths, supports k suffix. In default "
             "mode these are scout-only extras when not already present in --contexts; "
             "with --standalone-prefill they define the standalone cold profile. "
             "(default: 8k,64k,128k)"
    )
    parser.add_argument(
        "--prefill-metric", default="client",
        help="Headline prefill timing source: client, auto, prometheus, or ttft. "
             "client uses prompt_tokens / TTFT with no server dependency. auto uses "
             "the same client headline and also collects Prometheus validation when "
             "available. prometheus forces server counters as the headline. ttft is "
             "a legacy alias for client. (default: client)"
    )
    parser.add_argument(
        "--token-targeting", choices=("estimate", "exact"), default="estimate",
        help="How to build context prompts. estimate uses one calibration request "
             "and extrapolates all contexts; exact binary-searches each context "
             "through /tokenize and is much slower. (default: estimate)"
    )
    parser.add_argument(
        "--calibration-context", default="8k",
        help="Single-point token calibration context for --token-targeting=estimate "
             "(default: 8k)"
    )
    parser.add_argument(
        "--calibration-cache", default=DEFAULT_CALIBRATION_CACHE,
        help=f"Path to chars/token calibration cache (default: {DEFAULT_CALIBRATION_CACHE})"
    )
    parser.add_argument(
        "--no-calibration-cache", action="store_true",
        help="Disable cached chars/token calibration and force a fresh calibration request."
    )
    parser.add_argument(
        "--display-mode", choices=("screen", "live", "plain"), default="screen",
        help="Progress display mode. screen uses Rich alternate screen to avoid "
             "inline flicker; live is the old inline live table; plain disables "
             "live updates. (default: screen)"
    )
    parser.add_argument(
        "--refresh-rate", type=float, default=1.0,
        help="Rich live refresh rate in Hz for screen/live display modes (default: 1)"
    )
    parser.add_argument(
        "--hw-monitor-interval", type=float, default=2.0,
        help="Hardware monitor sampling interval in seconds for GPU/CPU live panel. "
             "Set 0 or use --no-hw-monitor to disable. (default: 2)"
    )
    parser.add_argument(
        "--no-hw-monitor", action="store_true",
        help="Disable the live hardware panel sampler."
    )
    parser.add_argument(
        "--hw-gpu-limit", type=int, default=8,
        help="Maximum GPUs to show in the live hardware panel. All sampled GPUs "
             "still contribute to aggregate PCIe rx/tx. (default: 8)"
    )
    parser.add_argument(
        "--respect-eos", action="store_true",
        help="Respect model EOS while measuring decode. Default is ignore_eos=true "
             "so decode cells are not contaminated by repeated short completions."
    )
    parser.add_argument(
        "--show-capacity-limited-values", action="store_true",
        help="Show raw throughput for capacity-limited cells in tables. By default "
             f"such cells are printed as {CAPACITY_LIMIT_MARK} and the raw value is kept in JSON."
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
        "--dcp-size", type=int, default=int(os.environ.get("LLM_BENCH_DCP_SIZE", "0")),
        help="vLLM CP/DCP multiplier for auto-detected KV cache budget. "
             "vLLM Prometheus exposes local GPU KV blocks; startup logs multiply "
             "them by prefill_context_parallel_size × decode_context_parallel_size. "
             "Set this to the effective CP multiplier, or use LLM_BENCH_DCP_SIZE. "
             "0 = auto-detect from local vLLM process when possible. (default: 0)"
    )
    parser.add_argument(
        "--skip-prefill", action="store_true",
        help="Skip prefill reporting entirely and go straight to decode tests"
    )
    parser.add_argument(
        "--standalone-prefill", action="store_true",
        help="Run the old standalone cold prefill profile before decode. By default, "
             "prefill is measured from the decode scout requests that are needed anyway."
    )
    args = parser.parse_args()
    if args.request_count < 0:
        parser.error("--request-count must be >= 0")
    if args.warmup_request_count < 0:
        parser.error("--warmup-request-count must be >= 0")
    if args.warmup_request_count > 0 and args.request_count == 0:
        parser.error("--warmup-request-count requires --request-count")
    if args.burst_request_count < 0:
        parser.error("--burst-request-count must be >= 0")
    if args.burst_warmup_request_count < 0:
        parser.error("--burst-warmup-request-count must be >= 0")
    if args.burst_requests_per_concurrency < 1:
        parser.error("--burst-requests-per-concurrency must be >= 1")
    if args.hw_gpu_limit < 1:
        parser.error("--hw-gpu-limit must be >= 1")
    prefill_metric = args.prefill_metric.lower()
    if prefill_metric in ("ttft", "aiperf"):
        prefill_metric = "client"
    if prefill_metric not in ("client", "auto", "prometheus"):
        parser.error("--prefill-metric must be one of: client, auto, prometheus, ttft")
    args.prefill_metric = prefill_metric
    return args


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
    decode_mode = (
        f"Request-count: {args.request_count} measured"
        + (f" + {args.warmup_request_count} warmup" if args.warmup_request_count else "")
        + " requests per cell"
        if args.request_count > 0
        else f"Duration: {args.duration}s per decode test"
    )
    phase3 = (
        " | Phase 3: Burst/E2E"
        if args.run_burst and args.request_count == 0
        else ""
    )
    if args.skip_prefill:
        prefill_label = "Prefill: skipped"
    elif args.standalone_prefill:
        prefill_label = f"Prefill: standalone cold profile ({args.prefill_metric})"
    else:
        prefill_label = "Prefill: integrated decode scouts"

    console.print(Panel(
        f"[bold {PHOSPHOR}]LLM Inference Benchmark[/bold {PHOSPHOR}]\n"
        f"Model: {args.model} @ {args.host if args.host.startswith('http') else f'{args.host}:{args.port or 5000}'}\n"
        f"Decode concurrency: {concurrency_levels}\n"
        f"Decode contexts: {[format_context(c) for c in context_lengths]}\n"
        f"{decode_mode} | Max tokens: {args.max_tokens}\n"
        f"{prefill_label} | Sustained decode: {decode_count} cells{phase3}",
        title=render_title("Configuration"),
        box=PANEL_BOX,
        border_style=FRAME_BORDER,
    ))

    engine = ""
    burst_results = []
    try:
        results, burst_results, prefill_results, engine = asyncio.run(run_benchmark(args))
        _prefill_results = prefill_results
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Saving partial results...[/yellow]")
        results = _partial_results
        prefill_results = _prefill_results

    if results or burst_results or prefill_results:
        print_final_results(
            results,
            concurrency_levels,
            context_lengths,
            console,
            prefill_results,
            show_capacity_limited_values=args.show_capacity_limited_values,
            burst_results=burst_results,
        )
        save_results(results, args, args.output, prefill_results, engine=engine, burst_results=burst_results)
        console.print(f"\n[green]Results saved to {args.output}[/green]")
    else:
        console.print("[red]No results collected.[/red]")


if __name__ == "__main__":
    try:
        main()
    finally:
        _restore_terminal()
