#!/usr/bin/env python3
"""Plot Transolver validation ground truth versus checkpoint predictions."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

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


AXIS_NAMES = ("rho", "theta", "phi")


def _default_run_dir() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    candidates = sorted(
        (repo_root / "runs" / "transolver_alpha").glob("*/config.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].parent
    return Path("runs/transolver_alpha/53147341")


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

    saved = Path(saved_root)
    saved_text = str(saved)
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


def _parse_index_list(value: str | None, available: Sequence[int]) -> list[int]:
    if not value:
        return list(available)

    selected: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if ":" in token:
            pieces = [int(piece) if piece else None for piece in token.split(":")]
            if len(pieces) > 3:
                raise ValueError(f"Invalid index range: {token}")
            selected.extend(available[slice(*pieces)])
        else:
            index = int(token)
            selected.append(available[index])
    return selected


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


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
    return _to_float(target), _to_float(prediction.squeeze(0)), sample["folder"]


def _model_value_to_target_units(value: float, saved_args: dict[str, Any]) -> float:
    if saved_args.get("no_log_target", False):
        return value
    return max((10.0**value) - float(saved_args["target_eps"]), 0.0)


def _target_name(saved_args: dict[str, Any]) -> str:
    return str(saved_args.get("target_database_key") or "reduced target")


def _predict_node_grid(
    model: torch.nn.Module,
    sample: dict[str, Any],
    saved_args: dict[str, Any],
    *,
    device: torch.device,
) -> torch.Tensor:
    x, pos, _ = sample_to_transolver_tensors(
        sample,
        max_nodes=None,
        target_reduction=saved_args["target_reduction"],
        log10_target=not saved_args["no_log_target"],
        target_eps=saved_args["target_eps"],
        profile_log1p=not saved_args["no_profile_log1p"],
        generator=torch.Generator().manual_seed(0),
    )
    grid_shape = tuple(int(size) for size in sample["bfield"]["br"].shape)
    if x.shape[0] != math.prod(grid_shape):
        raise ValueError(f"Cannot reshape {x.shape[0]} nodes to grid shape {grid_shape}")
    with torch.no_grad():
        node_values = model((x.unsqueeze(0).to(device), pos.unsqueeze(0).to(device), None))
    return node_values.squeeze(0).squeeze(-1).detach().cpu().reshape(grid_shape)


def _take_slice(array: torch.Tensor, axis: int, index: int) -> torch.Tensor:
    index = max(0, min(index, array.shape[axis] - 1))
    slices = [slice(None)] * array.ndim
    slices[axis] = index
    return array[tuple(slices)]


def _plot_scalar_comparison(
    output_path: Path,
    validation_offsets: Sequence[int],
    targets: Sequence[float],
    predictions: Sequence[float],
    model_targets: Sequence[float],
    model_predictions: Sequence[float],
    folders: Sequence[str],
    *,
    checkpoint_name: str,
    target_name: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    mean_target = sum(targets) / len(targets)
    ss_res = sum((target - prediction) ** 2 for target, prediction in zip(targets, predictions))
    ss_tot = sum((target - mean_target) ** 2 for target in targets)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    lo = min(min(targets), min(predictions))
    hi = max(max(targets), max(predictions))
    padding = 0.05 * max(hi - lo, 1.0e-12)
    lo -= padding
    hi += padding

    axes[0].scatter(targets, predictions, s=28, alpha=0.8, label=f"R^2 = {r2:.4g}")
    axes[0].plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--", label="Ideal")
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)
    axes[0].set_xlabel(f"Ground truth {target_name}")
    axes[0].set_ylabel(f"Predicted {target_name}")
    axes[0].set_title(f"Predicted vs Ground Truth {target_name}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    sample_axis = list(range(len(targets)))
    axes[1].plot(sample_axis, targets, marker="o", label="Ground Truth")
    axes[1].plot(sample_axis, predictions, marker="o", label="Prediction")
    axes[1].set_xlabel("Selected Validation Sample")
    axes[1].set_ylabel(target_name)
    axes[1].set_title("Selected Validation Examples")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    mae = sum(abs(pred - target) for pred, target in zip(predictions, targets)) / len(targets)
    mse = ss_res / len(targets)
    fig.suptitle(
        f"{checkpoint_name}: validation {target_name} predictions, MSE={mse:.4g}, MAE={mae:.4g}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w") as file:
        file.write(
            "validation_index,"
            f"ground_truth_{target_name},prediction_{target_name},"
            "model_space_ground_truth,model_space_prediction,folder\n"
        )
        for offset, target, prediction, model_target, model_prediction, folder in zip(
            validation_offsets,
            targets,
            predictions,
            model_targets,
            model_predictions,
            folders,
        ):
            file.write(
                f"{offset},{target:.10g},{prediction:.10g},"
                f"{model_target:.10g},{model_prediction:.10g},{folder}\n"
            )


def _plot_spatial_slices(
    output_path: Path,
    model: torch.nn.Module,
    dataset: Ascot5Dataset,
    val_indices: Sequence[int],
    saved_args: dict[str, Any],
    *,
    example_indices: Sequence[int],
    axis: int,
    slice_index: int | None,
    device: torch.device,
) -> None:
    if not example_indices:
        return

    rows = len(example_indices)
    fig, axes = plt.subplots(rows, 2, figsize=(9, 4 * rows), constrained_layout=True)
    if rows == 1:
        axes = axes[None, :]

    for row, validation_offset in enumerate(example_indices):
        dataset_index = val_indices[validation_offset]
        sample = dataset[dataset_index]
        prediction_grid = _predict_node_grid(model, sample, saved_args, device=device)
        target = sample["target"].detach().cpu()
        grid_shape = prediction_grid.shape
        index = slice_index if slice_index is not None else grid_shape[axis] // 2
        prediction_slice = _take_slice(prediction_grid, axis, index)

        axes[row, 0].imshow(prediction_slice, origin="lower", aspect="auto")
        axes[row, 0].set_title(
            f"Val {validation_offset}: node prediction, {AXIS_NAMES[axis]} index {index}"
        )
        axes[row, 0].set_xlabel("Grid axis")
        axes[row, 0].set_ylabel("Grid axis")

        if tuple(target.shape) == tuple(grid_shape):
            target_slice = _take_slice(target, axis, index)
            image = axes[row, 1].imshow(target_slice, origin="lower", aspect="auto")
            axes[row, 1].set_title("Ground-truth target slice")
            fig.colorbar(image, ax=axes[row, 1], shrink=0.8)
        else:
            target_value = _target_scalar(sample, saved_args)
            axes[row, 1].axis("off")
            axes[row, 1].text(
                0.5,
                0.55,
                f"Scalar ground truth: {_to_float(target_value):.6g}\n"
                f"Raw target shape: {tuple(target.shape)}\n"
                "This run was trained on the reduced scalar target.",
                ha="center",
                va="center",
                fontsize=12,
            )
        image = axes[row, 0].images[0]
        fig.colorbar(image, ax=axes[row, 0], shrink=0.8)

    fig.suptitle("Validation Spatial Slices")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Training run directory containing config.json and checkpoints.",
    )
    parser.add_argument("--checkpoint", default="best.pt", help="Checkpoint filename or path.")
    parser.add_argument("--results-root", type=Path, help="Override results root from config.json.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG for scalar validation comparison.",
    )
    parser.add_argument(
        "--indices",
        help=(
            "Validation-relative indices to plot, e.g. '0,1,5' or '0:20:2'. "
            "Defaults to all validation samples."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit the number of validation samples when --indices is not provided.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        help="Nodes sampled per sample for scalar predictions. Defaults to the training value.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--slice-output",
        type=Path,
        help="Optional PNG for spatial node-output slices from selected examples.",
    )
    parser.add_argument(
        "--slice-examples",
        default="0",
        help="Validation-relative example indices for --slice-output, e.g. '0,4,8'.",
    )
    parser.add_argument("--slice-axis", choices=AXIS_NAMES, default="phi")
    parser.add_argument("--slice-index", type=int, help="Index along --slice-axis. Defaults to middle.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = (args.run_dir or _default_run_dir()).expanduser().resolve()
    config = _load_run_config(run_dir)
    saved_args = config["args"]
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path
    checkpoint = _load_checkpoint(checkpoint_path)

    results_root = _resolve_results_root(saved_args["results_root"], args.results_root)
    target_database_path = saved_args.get("target_database")
    if target_database_path is not None:
        target_database_path = Path(target_database_path).expanduser()
    folders = _discover_sample_folders(
        results_root,
        saved_args["analysis_filename"],
        saved_args["equilibrium_filename"],
        saved_args["bfield_filename"],
    )
    if saved_args["max_samples"] is not None:
        folders = folders[: saved_args["max_samples"]]

    _, val_indices = _split_indices(len(folders), saved_args["train_fraction"], saved_args["seed"])
    if not val_indices:
        raise ValueError("This run has no validation split.")

    available_offsets = list(range(len(val_indices)))
    selected_offsets = _parse_index_list(args.indices, available_offsets)
    if args.indices is None and args.max_samples is not None:
        selected_offsets = selected_offsets[: args.max_samples]
    selected_dataset_indices = [val_indices[offset] for offset in selected_offsets]

    dataset = Ascot5Dataset(
        folders,
        analysis_filename=saved_args["analysis_filename"],
        equilibrium_filename=saved_args["equilibrium_filename"],
        bfield_filename=saved_args["bfield_filename"],
        include_bfield=True,
        strict=True,
        target_database_path=target_database_path,
        target_database_key=saved_args.get("target_database_key", "fraction_lost"),
    )

    device = torch.device(args.device)
    _ensure_distributed(device)
    patch_transolver_attention_for_cuda()
    model_config = checkpoint.get("model_config", config["model_config"])
    model = TransolverPlusModel(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    generator = torch.Generator().manual_seed(saved_args["seed"])
    max_nodes = args.max_nodes if args.max_nodes is not None else saved_args["max_nodes"]
    targets, predictions, model_targets, model_predictions, folders_out = [], [], [], [], []
    for dataset_index in selected_dataset_indices:
        model_target, model_prediction, folder = _predict_one(
            model,
            dataset[dataset_index],
            saved_args,
            max_nodes=max_nodes,
            generator=generator,
            device=device,
        )
        model_targets.append(model_target)
        model_predictions.append(model_prediction)
        targets.append(_model_value_to_target_units(model_target, saved_args))
        predictions.append(_model_value_to_target_units(model_prediction, saved_args))
        folders_out.append(folder)

    output_path = args.output
    if output_path is None:
        output_path = run_dir / f"{checkpoint_path.stem}_validation_predictions.png"
    _plot_scalar_comparison(
        output_path.expanduser().resolve(),
        selected_offsets,
        targets,
        predictions,
        model_targets,
        model_predictions,
        folders_out,
        checkpoint_name=checkpoint_path.name,
        target_name=_target_name(saved_args),
    )
    print(f"Wrote {output_path.expanduser().resolve()}")

    if args.slice_output is not None:
        slice_offsets = _parse_index_list(args.slice_examples, available_offsets)
        axis = AXIS_NAMES.index(args.slice_axis)
        _plot_spatial_slices(
            args.slice_output.expanduser().resolve(),
            model,
            dataset,
            val_indices,
            saved_args,
            example_indices=slice_offsets,
            axis=axis,
            slice_index=args.slice_index,
            device=device,
        )
        print(f"Wrote {args.slice_output.expanduser().resolve()}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
