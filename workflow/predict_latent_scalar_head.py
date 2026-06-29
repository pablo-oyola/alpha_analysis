#!/usr/bin/env python3
"""Run trained latent scalar-head inference for validation and new token records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from alpha_analysis.ai.train_latent_generator import (
    ManifestRecord,
    NormalizationStats,
    TokenScalarHead,
    _load_latent,
    _read_manifest,
    _unnormalize_target,
)


def _stats_from_state(state: dict[str, Tensor]) -> NormalizationStats:
    return NormalizationStats(
        condition_mean=state["condition_mean"],
        condition_std=state["condition_std"],
        latent_mean=state["latent_mean"],
        latent_std=state["latent_std"],
        target_mean=state["target_mean"],
        target_std=state["target_std"],
    )


def _load_scalar_head(checkpoint_path: Path, device: torch.device) -> tuple[TokenScalarHead, NormalizationStats, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    config_args = config["args"]
    model = TokenScalarHead(
        token_dim=int(config["latent_dim"]),
        num_tokens=int(config["num_tokens"]),
        width=int(config_args["width"]),
        depth=int(config_args["scalar_depth"]),
        heads=int(config_args["heads"]),
        dropout=float(config_args["dropout"]),
        mlp_ratio=int(config_args["mlp_ratio"]),
    ).to(device)
    model.load_state_dict(checkpoint["scalar_head_state_dict"])
    model.eval()
    return model, _stats_from_state(checkpoint["stats"]), config


def _predict_records(
    records: Sequence[ManifestRecord],
    *,
    scalar_head: TokenScalarHead,
    stats: NormalizationStats,
    token_kind: str,
    layers: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, record in enumerate(records):
            latent = _load_latent(record.token_path, token_kind, layers)
            latent = (latent - stats.latent_mean.cpu()) / stats.latent_std.cpu()
            prediction = scalar_head(latent.unsqueeze(0).to(device))
            prediction = _unnormalize_target(prediction, stats).squeeze().detach().cpu()
            rows.append(
                {
                    "dataset": record.split,
                    "index": index,
                    "split_index": record.split_index,
                    "dataset_index": record.dataset_index,
                    "ground_truth_target": record.target,
                    "prediction_target": float(prediction.item()),
                    "transolver_prediction": record.prediction,
                    "folder": record.folder,
                    "token_path": str(record.token_path),
                }
            )
    return rows


def _metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"mse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    targets = [float(row["ground_truth_target"]) for row in rows]
    predictions = [float(row["prediction_target"]) for row in rows]
    mse = sum((prediction - target) ** 2 for target, prediction in zip(targets, predictions)) / len(rows)
    mae = sum(abs(prediction - target) for target, prediction in zip(targets, predictions)) / len(rows)
    mean_target = sum(targets) / len(targets)
    ss_tot = sum((target - mean_target) ** 2 for target in targets)
    ss_res = sum((prediction - target) ** 2 for target, prediction in zip(targets, predictions))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return {"mse": mse, "mae": mae, "r2": r2}


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "index",
        "split_index",
        "dataset_index",
        "ground_truth_target",
        "prediction_target",
        "transolver_prediction",
        "folder",
        "token_path",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(output_path: Path, val_rows: Sequence[dict[str, Any]], new_rows: Sequence[dict[str, Any]]) -> None:
    val_targets = [float(row["ground_truth_target"]) for row in val_rows]
    val_predictions = [float(row["prediction_target"]) for row in val_rows]
    new_targets = [float(row["ground_truth_target"]) for row in new_rows]
    new_predictions = [float(row["prediction_target"]) for row in new_rows]
    all_targets = val_targets + new_targets
    all_predictions = val_predictions + new_predictions

    lo = min(min(all_targets), min(all_predictions))
    hi = max(max(all_targets), max(all_predictions))
    padding = 0.05 * max(hi - lo, 1.0e-12)
    lo -= padding
    hi += padding

    val_metrics = _metrics(val_rows)
    new_metrics = _metrics(new_rows)
    all_metrics = _metrics([*val_rows, *new_rows])

    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    ax.scatter(
        val_targets,
        val_predictions,
        s=34,
        alpha=0.78,
        color="#ff7f0e",
        label=f"Validation ({len(val_rows)})",
    )
    ax.scatter(
        new_targets,
        new_predictions,
        s=34,
        alpha=0.78,
        color="#d62728",
        label=f"New ({len(new_rows)})",
    )
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--", label="Ideal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Ground truth fraction_lost")
    ax.set_ylabel("Scalar-head predicted fraction_lost")
    ax.set_title("Scalar Head Predicted vs Ground Truth")
    ax.grid(True, alpha=0.3)
    ax.legend(
        title=(
            f"All R^2={all_metrics['r2']:.4g}\n"
            f"Val MAE={val_metrics['mae']:.4g}\n"
            f"New MAE={new_metrics['mae']:.4g}"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token-dir",
        type=Path,
        default=Path("runs/transolver_alpha/53562942/best_slice_tokens"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/transolver_alpha/53562942/best_slice_tokens/"
            "latent_generator_54945515/scalar_head_best.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/transolver_alpha/53562942/best_slice_tokens/"
            "latent_generator_54945515/scalar_head_predicted_vs_ground_truth_with_new.png"
        ),
    )
    parser.add_argument(
        "--validation-output-csv",
        type=Path,
        default=Path(
            "runs/transolver_alpha/53562942/best_slice_tokens/"
            "latent_generator_54945515/scalar_head_predictions_val.csv"
        ),
    )
    parser.add_argument(
        "--new-output-csv",
        type=Path,
        default=Path(
            "runs/transolver_alpha/53562942/best_slice_tokens/"
            "latent_generator_54945515/scalar_head_predictions_new.csv"
        ),
    )
    parser.add_argument(
        "--combined-output-csv",
        type=Path,
        default=Path(
            "runs/transolver_alpha/53562942/best_slice_tokens/"
            "latent_generator_54945515/scalar_head_predictions_val_new.csv"
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    token_dir = args.token_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    device = torch.device(args.device)
    scalar_head, stats, config = _load_scalar_head(checkpoint_path, device)
    token_kind = str(config["token_kind"])
    layers = str(config["layers"])

    records = _read_manifest(token_dir)
    val_records = [record for record in records if record.split == "val"]
    new_records = [record for record in records if record.split == "new"]
    if not val_records:
        raise ValueError("No validation records found in manifest.")
    if not new_records:
        raise ValueError("No new records found in manifest.")

    val_rows = _predict_records(
        val_records,
        scalar_head=scalar_head,
        stats=stats,
        token_kind=token_kind,
        layers=layers,
        device=device,
    )
    new_rows = _predict_records(
        new_records,
        scalar_head=scalar_head,
        stats=stats,
        token_kind=token_kind,
        layers=layers,
        device=device,
    )
    _write_csv(args.validation_output_csv, val_rows)
    _write_csv(args.new_output_csv, new_rows)
    _write_csv(args.combined_output_csv, [*val_rows, *new_rows])
    _plot(args.output, val_rows, new_rows)

    print(f"Wrote {args.validation_output_csv.resolve()}")
    print(f"Wrote {args.new_output_csv.resolve()}")
    print(f"Wrote {args.combined_output_csv.resolve()}")
    print(f"Wrote {args.output.resolve()}")
    print(f"Validation metrics: {_metrics(val_rows)}")
    print(f"New metrics: {_metrics(new_rows)}")
    print(f"Combined metrics: {_metrics([*val_rows, *new_rows])}")


if __name__ == "__main__":
    main()
