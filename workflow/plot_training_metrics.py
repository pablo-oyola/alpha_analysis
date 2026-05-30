#!/usr/bin/env python3
"""Plot train/validation MSE and MAE metrics from Transolver training metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_RE = re.compile(
    r"epoch=(?P<epoch>\d+)\s+"
    r"train_mse=(?P<train_mse>[-+0-9.eE]+)\s+"
    r"train_mae=(?P<train_mae>[-+0-9.eE]+)\s+"
    r"val_mse=(?P<val_mse>[-+0-9.eE]+)\s+"
    r"val_mae=(?P<val_mae>[-+0-9.eE]+)"
)


METRIC_NAMES = ("train_mse", "val_mse", "train_mae", "val_mae")


def _empty_metrics() -> dict[str, list[float]]:
    return {
        "epoch": [],
        "train_mse": [],
        "train_mae": [],
        "val_mse": [],
        "val_mae": [],
    }


def parse_jsonl_metrics(metrics_path: Path) -> dict[str, list[float]]:
    metrics = _empty_metrics()

    with metrics_path.open() as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {metrics_path}: {exc}"
                ) from exc

            if "epoch" not in row:
                raise ValueError(f"Missing 'epoch' on line {line_number} of {metrics_path}")
            metrics["epoch"].append(float(row["epoch"]))
            for name in METRIC_NAMES:
                value = row.get(name)
                metrics[name].append(float(value) if value is not None else float("nan"))

    if not metrics["epoch"]:
        raise ValueError(f"No epoch metric rows found in {metrics_path}")

    return metrics


def parse_log_metrics(log_path: Path) -> dict[str, list[float]]:
    metrics = _empty_metrics()

    for line in log_path.read_text().splitlines():
        match = METRIC_RE.search(line)
        if not match:
            continue
        metrics["epoch"].append(float(match.group("epoch")))
        for name in ("train_mse", "train_mae", "val_mse", "val_mae"):
            metrics[name].append(float(match.group(name)))

    if not metrics["epoch"]:
        raise ValueError(f"No epoch metric rows found in {log_path}")

    return metrics


def parse_metrics(metrics_path: Path) -> dict[str, list[float]]:
    if metrics_path.suffix.lower() == ".jsonl":
        return parse_jsonl_metrics(metrics_path)
    return parse_log_metrics(metrics_path)


def plot_metrics(metrics: dict[str, list[float]], output_path: Path, log_scale: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    epochs = metrics["epoch"]

    axes[0].plot(epochs, metrics["train_mse"], marker="o", label="Train MSE")
    if any(not math.isnan(value) for value in metrics["val_mse"]):
        axes[0].plot(epochs, metrics["val_mse"], marker="o", label="Validation MSE")
    axes[0].set_title("MSE vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")

    axes[1].plot(epochs, metrics["train_mae"], marker="o", label="Train MAE")
    if any(not math.isnan(value) for value in metrics["val_mae"]):
        axes[1].plot(epochs, metrics["val_mae"], marker="o", label="Validation MAE")
    axes[1].set_title("MAE vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")

    for axis in axes:
        if log_scale:
            axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()

    fig.suptitle("Transolver Training Metrics")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def default_metrics_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    run_candidates = sorted(
        (repo_root / "runs").glob("transolver_alpha/*/metrics.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if run_candidates:
        return run_candidates[0]

    logs_dir = Path(__file__).resolve().parent / "logs"
    candidates = sorted(
        logs_dir.glob("train_transolver_*.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No metrics.jsonl files found in {repo_root / 'runs'} and no "
            f"train_transolver logs found in {logs_dir}"
        )
    return candidates[0]


def default_output_path(metrics_path: Path) -> Path:
    if metrics_path.name == "metrics.jsonl":
        return metrics_path.with_name(f"{metrics_path.parent.name}_metrics.png")
    return metrics_path.with_name(f"{metrics_path.stem}_metrics.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot train/validation MSE and MAE by epoch from metrics.jsonl or a log."
    )
    parser.add_argument(
        "metrics_path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to metrics.jsonl or a training log. Defaults to the newest run metrics.jsonl.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "PNG output path. Defaults to '<input_stem>_metrics.png' next to the input, "
            "or '<run_id>_metrics.png' for a run metrics.jsonl."
        ),
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use a linear y-axis instead of the default log scale.",
    )
    args = parser.parse_args()

    metrics_path = (args.metrics_path or default_metrics_path()).expanduser().resolve()
    output_path = args.output
    if output_path is None:
        output_path = default_output_path(metrics_path)
    output_path = output_path.expanduser().resolve()

    metrics = parse_metrics(metrics_path)
    plot_metrics(metrics, output_path, log_scale=not args.linear)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
