"""Persist benchmark performance and precision summaries."""

from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_device_result_name(device_type):
    """Return the device label used by the benchmark result directories."""
    normalized = str(device_type).split(":", 1)[0].lower()
    return {"cuda": "GPU", "npu": "NPU", "cpu": "CPU"}.get(
        normalized, normalized.upper()
    )


def resolve_results_dir(device_name, configured_dir=""):
    """Resolve the result directory, defaulting to a sibling of the repository."""
    if configured_dir:
        return Path(configured_dir).expanduser().resolve()
    return PROJECT_ROOT.parent / f"save_results_{device_name}"


def summarize_latencies(latency_seconds, batch_size, model_name):
    """Summarize non-negative per-step timings using millisecond units."""
    values = np.asarray(latency_seconds, dtype=np.float64)
    if values.size == 0:
        raise ValueError("at least one latency sample is required")
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("latency samples must be a finite, non-negative sequence")

    latency_ms = values * 1000.0
    return {
        "Batch_size": int(batch_size),
        "model_name": str(model_name),
        "AVG Latency": float(np.mean(latency_ms)),
        "P999 Latency": float(np.percentile(latency_ms, 99.9, method="lower")),
        "P99 Latency": float(np.percentile(latency_ms, 99, method="lower")),
        "P95 Latency": float(np.percentile(latency_ms, 95, method="lower")),
        "P90 Latency": float(np.percentile(latency_ms, 90, method="lower")),
    }


def _append_line(path, line):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as result_file:
        result_file.write(line + "\n")


def save_benchmark_results(
    latency_seconds,
    batch_size,
    model_name,
    device_type,
    inductor_enabled,
    precision_results,
    results_dir="",
):
    """Save DLRM-compatible performance and precision result artifacts."""
    device_name = get_device_result_name(device_type)
    normalized_model_name = str(model_name).upper()
    inductor_flag = "inductor" if inductor_enabled else "eager"
    output_dir = resolve_results_dir(device_name, results_dir)

    performance = summarize_latencies(
        latency_seconds, batch_size, normalized_model_name
    )
    model_detail = f"{device_name}_{normalized_model_name}_{inductor_flag}"
    performance_line = f"performance: {model_detail}_{performance}"
    performance_path = output_dir / "performance_result.txt"
    _append_line(performance_path, performance_line)

    precision = dict(precision_results)
    precision["ms/step"] = performance["AVG Latency"]
    precision_line = f"{model_detail}_{precision}"
    precision_path = output_dir / "precision_result.txt"
    _append_line(precision_path, precision_line)

    model_dir = output_dir / normalized_model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    step = precision.get("step", 0)
    artifact_path = (
        model_dir
        / f"predictions_{normalized_model_name}_{inductor_flag}_{step}.pt"
    )
    torch.save(precision, artifact_path)

    return {
        "performance": performance,
        "precision": precision,
        "performance_path": performance_path,
        "precision_path": precision_path,
        "artifact_path": artifact_path,
    }
