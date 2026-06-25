#!/usr/bin/env python3
"""Predict fraction_lost for post-training ASCOT samples and make a parity plot."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist

from alpha_analysis.ai.dataloader import Ascot5Dataset
from alpha_analysis.ai.train_transolver import (
    _discover_sample_folders,
    _ensure_distributed,
    _predict_sample_scalar,
    _reduce_target,
    _split_indices,
    patch_transolver_attention_for_cuda,
    sample_to_transolver_tensors,
)
from models.Transolver_plus import Model as TransolverPlusModel


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    return json.loads(config_path.read_text())


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _resolve_results_root(saved_root: str, override: Path | None) -> Path:
    candidates = []
    if override is not None:
        candidates.append(override.expanduser())
    candidates.append(Path(saved_root).expanduser())

    saved_text = str(Path(saved_root))
    if saved_text.startswith("/global/cfs/cdirs/"):
        candidates.append(Path(saved_text.replace("/global/cfs/cdirs/", "/global/cfs/projectdirs/", 1)))
    if saved_text.startswith("/global/cfs/projectdirs/"):
        candidates.append(Path(saved_text.replace("/global/cfs/projectdirs/", "/global/cfs/cdirs/", 1)))

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find results root. Tried: " + ", ".join(str(item) for item in candidates)
    )


def _model_value_to_target_units(value: float, saved_args: dict[str, Any]) -> float:
    if saved_args.get("no_log_target", False):
        return value
    return max((10.0**value) - float(saved_args["target_eps"]), 0.0)


def _sample_key(folder: Path) -> str:
    match = re.search(r"_(\d+)$", folder.name)
    if match is None:
        raise ValueError(f"Cannot extract sample key from folder name: {folder}")
    return match.group(1)


def _load_target_keys(database_path: Path, target_key: str) -> set[str]:
    with database_path.expanduser().open() as file:
        database = json.load(file)
    return {
        str(sample_key)
        for sample_key, sample_data in database.items()
        if isinstance(sample_data, dict) and target_key in sample_data
    }


def _validation_folders(csv_path: Path) -> list[str]:
    with csv_path.open(newline="") as file:
        return [Path(row["folder"]).name for row in csv.DictReader(file)]


def _infer_training_sample_count(
    folders: Sequence[Path],
    validation_csv: Path,
    *,
    train_fraction: float,
    seed: int,
) -> int:
    validation_names = _validation_folders(validation_csv)
    if not validation_names:
        raise ValueError(f"No validation rows found in {validation_csv}")

    best_count = 0
    best_matches = -1
    for count in range(len(validation_names), len(folders) + 1):
        _, val_indices = _split_indices(count, train_fraction, seed)
        names = [folders[index].name for index in val_indices]
        matches = sum(a == b for a, b in zip(names, validation_names))
        if matches > best_matches:
            best_count = count
            best_matches = matches
        if names == validation_names:
            return count

    raise ValueError(
        "Could not exactly infer training sample count from "
        f"{validation_csv}; best partial match was {best_matches} rows at count={best_count}."
    )


def _target_scalar(sample: dict[str, Any], saved_args: dict[str, Any]) -> torch.Tensor:
    return _reduce_target(
        sample["target"],
        saved_args["target_reduction"],
        log10_target=not saved_args["no_log_target"],
        eps=saved_args["target_eps"],
    )


def _predict_one(
    model: torch.nn.Module,
    sample: dict[str, Any],
    saved_args: dict[str, Any],
    *,
    max_nodes: int | None,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[float, float, str]:
    x, pos, target = sample_to_transolver_tensors(
        sample,
        max_nodes=max_nodes,
        target_reduction=saved_args["target_reduction"],
        log10_target=not saved_args["no_log_target"],
        target_eps=saved_args["target_eps"],
        profile_log1p=not saved_args["no_profile_log1p"],
        generator=generator,
    )
    mask = torch.ones((1, x.shape[0]), dtype=torch.bool, device=device)
    with torch.no_grad():
        prediction = _predict_sample_scalar(
            model,
            x.unsqueeze(0).to(device),
            pos.unsqueeze(0).to(device),
            mask,
        )
    return (
        float(target.detach().cpu().item()),
        float(prediction.squeeze(0).detach().cpu().item()),
        sample["folder"],
    )


def _read_validation_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as file:
        return list(csv.DictReader(file))


def _plot_combined(
    output_path: Path,
    validation_rows: Sequence[dict[str, str]],
    new_rows: Sequence[dict[str, float | str]],
    *,
    target_name: str,
) -> None:
    val_targets = [float(row[f"ground_truth_{target_name}"]) for row in validation_rows]
    val_predictions = [float(row[f"prediction_{target_name}"]) for row in validation_rows]
    new_targets = [float(row[f"ground_truth_{target_name}"]) for row in new_rows]
    new_predictions = [float(row[f"prediction_{target_name}"]) for row in new_rows]

    all_targets = val_targets + new_targets
    all_predictions = val_predictions + new_predictions
    mean_target = sum(all_targets) / len(all_targets)
    ss_res = sum((target - prediction) ** 2 for target, prediction in zip(all_targets, all_predictions))
    ss_tot = sum((target - mean_target) ** 2 for target in all_targets)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    mae = sum(abs(prediction - target) for target, prediction in zip(all_targets, all_predictions)) / len(
        all_targets
    )
    mse = ss_res / len(all_targets)

    lo = min(min(all_targets), min(all_predictions))
    hi = max(max(all_targets), max(all_predictions))
    padding = 0.05 * max(hi - lo, 1.0e-12)
    lo -= padding
    hi += padding

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].scatter(
        val_targets,
        val_predictions,
        s=26,
        alpha=0.75,
        color="#1f77b4",
        label=f"Validation ({len(val_targets)})",
    )
    axes[0].scatter(
        new_targets,
        new_predictions,
        s=30,
        alpha=0.8,
        color="#d62728",
        label=f"New ASCOT ({len(new_targets)})",
    )
    axes[0].plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--", label="Ideal")
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)
    axes[0].set_xlabel(f"Ground truth {target_name}")
    axes[0].set_ylabel(f"Predicted {target_name}")
    axes[0].set_title(f"Predicted vs Ground Truth {target_name}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title=f"All R^2 = {r2:.4g}")

    val_axis = list(range(len(val_targets)))
    new_axis = list(range(len(val_targets), len(val_targets) + len(new_targets)))
    axes[1].plot(val_axis, val_targets, marker="o", linewidth=1.5, label="Validation Ground Truth")
    axes[1].plot(val_axis, val_predictions, marker="o", linewidth=1.5, label="Validation Prediction")
    axes[1].plot(
        new_axis,
        new_targets,
        marker="o",
        linewidth=1.2,
        color="#d62728",
        label="New ASCOT Ground Truth",
    )
    axes[1].plot(
        new_axis,
        new_predictions,
        marker="o",
        linewidth=1.2,
        color="#9467bd",
        label="New ASCOT Prediction",
    )
    axes[1].axvline(len(val_targets) - 0.5, color="black", linewidth=1, alpha=0.4)
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel(target_name)
    axes[1].set_title("Validation and New ASCOT Examples")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        f"best.pt: validation + new {target_name} predictions, MSE={mse:.4g}, MAE={mae:.4g}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/transolver_alpha/53562942"),
        help="Training run directory containing config.json and checkpoints.",
    )
    parser.add_argument("--checkpoint", default="best.pt", help="Checkpoint filename or path.")
    parser.add_argument("--results-root", type=Path, help="Override results root from config.json.")
    parser.add_argument(
        "--validation-csv",
        type=Path,
        default=Path("fraction_lost_validation.csv"),
        help="Existing validation CSV used as the fixed baseline.",
    )
    parser.add_argument(
        "--training-sample-count",
        type=int,
        help="Number of sorted samples available when the model was trained.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fraction_lost_validation_with_new.png"),
        help="Output PNG for validation and new ASCOT comparison.",
    )
    parser.add_argument(
        "--new-output-csv",
        type=Path,
        default=Path("fraction_lost_new_predictions.csv"),
        help="Output CSV containing only new ASCOT predictions.",
    )
    parser.add_argument(
        "--combined-output-csv",
        type=Path,
        default=Path("fraction_lost_validation_with_new.csv"),
        help="Output CSV containing validation and new ASCOT predictions.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        help="Nodes sampled per sample for scalar predictions. Defaults to the training value.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    config = _load_run_config(run_dir)
    saved_args = config["args"]
    target_name = str(saved_args.get("target_database_key") or "fraction_lost")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path
    checkpoint = _load_checkpoint(checkpoint_path)

    results_root = _resolve_results_root(saved_args["results_root"], args.results_root)
    target_database_path = saved_args.get("target_database")
    if target_database_path is None:
        raise ValueError("This workflow expects the run to use a scalar target database.")
    target_database_path = Path(target_database_path).expanduser()
    target_keys = _load_target_keys(target_database_path, target_name)

    folders = _discover_sample_folders(
        results_root,
        saved_args["analysis_filename"],
        saved_args["equilibrium_filename"],
        saved_args["bfield_filename"],
    )
    folders = [folder for folder in folders if _sample_key(folder) in target_keys]
    training_count = args.training_sample_count
    if training_count is None:
        training_count = _infer_training_sample_count(
            folders,
            args.validation_csv,
            train_fraction=saved_args["train_fraction"],
            seed=saved_args["seed"],
        )
    new_folders = folders[training_count:]
    if not new_folders:
        raise ValueError("No post-training folders with target database entries were found.")

    dataset = Ascot5Dataset(
        new_folders,
        analysis_filename=saved_args["analysis_filename"],
        equilibrium_filename=saved_args["equilibrium_filename"],
        bfield_filename=saved_args["bfield_filename"],
        include_bfield=True,
        strict=True,
        target_database_path=target_database_path,
        target_database_key=target_name,
    )

    device = torch.device(args.device)
    _ensure_distributed(device)
    patch_transolver_attention_for_cuda()
    model_config = checkpoint.get("model_config", config["model_config"])
    model = TransolverPlusModel(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    torch.manual_seed(saved_args["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(saved_args["seed"])
    generator = torch.Generator().manual_seed(saved_args["seed"])
    max_nodes = args.max_nodes if args.max_nodes is not None else saved_args["max_nodes"]
    new_rows: list[dict[str, float | str]] = []
    for index in range(len(dataset)):
        model_target, model_prediction, folder = _predict_one(
            model,
            dataset[index],
            saved_args,
            max_nodes=max_nodes,
            generator=generator,
            device=device,
        )
        target = _model_value_to_target_units(model_target, saved_args)
        prediction = _model_value_to_target_units(model_prediction, saved_args)
        new_rows.append(
            {
                "dataset": "new",
                "index": index,
                f"ground_truth_{target_name}": target,
                f"prediction_{target_name}": prediction,
                "model_space_ground_truth": model_target,
                "model_space_prediction": model_prediction,
                "folder": folder,
            }
        )
        print(
            f"{index + 1}/{len(dataset)} {Path(folder).name} "
            f"truth={target:.6g} prediction={prediction:.6g}",
            flush=True,
        )

    fieldnames = [
        "dataset",
        "index",
        f"ground_truth_{target_name}",
        f"prediction_{target_name}",
        "model_space_ground_truth",
        "model_space_prediction",
        "folder",
    ]
    args.new_output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.new_output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)

    validation_rows = _read_validation_rows(args.validation_csv)
    combined_rows: list[dict[str, float | str]] = []
    for row in validation_rows:
        combined_rows.append(
            {
                "dataset": "validation",
                "index": row["validation_index"],
                f"ground_truth_{target_name}": row[f"ground_truth_{target_name}"],
                f"prediction_{target_name}": row[f"prediction_{target_name}"],
                "model_space_ground_truth": row["model_space_ground_truth"],
                "model_space_prediction": row["model_space_prediction"],
                "folder": row["folder"],
            }
        )
    combined_rows.extend(new_rows)
    args.combined_output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.combined_output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)

    _plot_combined(
        args.output,
        validation_rows,
        new_rows,
        target_name=target_name,
    )
    print(f"Wrote {args.new_output_csv.resolve()}")
    print(f"Wrote {args.combined_output_csv.resolve()}")
    print(f"Wrote {args.output.resolve()}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
