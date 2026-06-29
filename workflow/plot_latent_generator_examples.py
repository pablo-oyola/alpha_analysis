"""Plot generated vs ground-truth Transolver latents for a trained generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from alpha_analysis.ai.train_latent_generator import (
    ConditionalLatentGenerator,
    NormalizationStats,
    TokenScalarHead,
    _load_latent,
    _read_manifest,
    _records_for_split,
    _select_latent,
    _split_train_records,
    _stack_conditions,
    _unnormalize_target,
)


def _stats_from_state_dict(state: dict[str, torch.Tensor]) -> NormalizationStats:
    return NormalizationStats(
        condition_mean=state["condition_mean"],
        condition_std=state["condition_std"],
        latent_mean=state["latent_mean"],
        latent_std=state["latent_std"],
        target_mean=state["target_mean"],
        target_std=state["target_std"],
    )


def _load_models(
    checkpoint_path: Path,
    *,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[TokenScalarHead, ConditionalLatentGenerator, NormalizationStats]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = config["args"]
    scalar_head = TokenScalarHead(
        token_dim=int(config["latent_dim"]),
        num_tokens=int(config["num_tokens"]),
        width=int(args["width"]),
        depth=int(args["scalar_depth"]),
        heads=int(args["heads"]),
        dropout=float(args["dropout"]),
        mlp_ratio=int(args["mlp_ratio"]),
    )
    generator = ConditionalLatentGenerator(
        condition_dim=int(config["condition_dim"]),
        latent_dim=int(config["latent_dim"]),
        num_tokens=int(config["num_tokens"]),
        noise_dim=int(args["noise_dim"]),
        width=int(args["width"]),
        condition_depth=int(args["condition_depth"]),
        transformer_depth=int(args["generator_depth"]),
        heads=int(args["heads"]),
        dropout=float(args["dropout"]),
        mlp_ratio=int(args["mlp_ratio"]),
    )
    scalar_head.load_state_dict(checkpoint["scalar_head_state_dict"])
    generator.load_state_dict(checkpoint["generator_state_dict"])
    stats = _stats_from_state_dict(checkpoint["stats"])
    scalar_head.to(device).eval()
    generator.to(device).eval()
    return scalar_head, generator, stats


def _load_validation_tensors(
    config: dict[str, Any],
    *,
    stats: NormalizationStats,
) -> tuple[list[Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    token_dir = Path(config["token_dir"])
    args = config["args"]
    records = _read_manifest(token_dir)
    try:
        val_records = _records_for_split(records, "val")
    except ValueError:
        _, val_records = _split_train_records(
            records,
            fraction=float(args["fallback_val_fraction"]),
            seed=int(args["seed"]),
        )

    condition_cache = _stack_conditions(
        val_records,
        include_profile_stats=not bool(args["no_profile_condition"]),
        include_bfield_stats=not bool(args["no_bfield_condition"]),
        profile_log1p=not bool(args["no_profile_log1p"]),
    )
    conditions = []
    latents = []
    targets = []
    for record in val_records:
        condition = condition_cache[record.folder]
        condition = (condition - stats.condition_mean) / stats.condition_std
        latent = _load_latent(
            record.token_path,
            config["token_kind"],
            config["layers"],
        )
        latent = (latent - stats.latent_mean) / stats.latent_std
        conditions.append(condition)
        latents.append(latent)
        targets.append(float(getattr(record, config["scalar_target"])))
    return (
        val_records,
        torch.stack(conditions, dim=0),
        torch.stack(latents, dim=0),
        torch.tensor(targets, dtype=torch.float32).reshape(-1, 1),
    )


def _generate_mean_latents(
    generator: ConditionalLatentGenerator,
    conditions: torch.Tensor,
    *,
    noise_samples: int,
    noise_std: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator.eval()
    conditions = conditions.to(device)
    generated_sum: torch.Tensor | None = None
    generated_sq_sum: torch.Tensor | None = None
    with torch.no_grad():
        for sample_index in range(noise_samples):
            noise = noise_std * torch.randn(
                conditions.shape[0],
                generator.noise_dim,
                device=device,
            )
            generated = generator(conditions, noise)
            if generated_sum is None:
                generated_sum = torch.zeros_like(generated)
                generated_sq_sum = torch.zeros_like(generated)
            generated_sum += generated
            generated_sq_sum += generated.square()
            if (sample_index + 1) % 8 == 0 or sample_index + 1 == noise_samples:
                print(
                    f"generated {sample_index + 1}/{noise_samples} noise draws",
                    flush=True,
                )
    if generated_sum is None or generated_sq_sum is None:
        raise ValueError("noise_samples must be positive")
    mean = generated_sum / noise_samples
    variance = generated_sq_sum / noise_samples - mean.square()
    return mean.cpu(), variance.clamp_min(0.0).sqrt().cpu()


def _evaluate_examples(
    scalar_head: TokenScalarHead,
    stats: NormalizationStats,
    records: list[Any],
    latents: torch.Tensor,
    targets: torch.Tensor,
    generated_mean: torch.Tensor,
    generated_std: torch.Tensor,
    *,
    noise_samples: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[int]]:
    diff = generated_mean - latents
    mse = diff.square().mean(dim=(1, 2))
    mae = diff.abs().mean(dim=(1, 2))
    cosine = F.cosine_similarity(
        generated_mean.flatten(1),
        latents.flatten(1),
        dim=1,
    )
    with torch.no_grad():
        teacher_norm = scalar_head(latents.to(device)).cpu()
        generated_norm = scalar_head(generated_mean.to(device)).cpu()
    teacher_real = _unnormalize_target(teacher_norm, stats).reshape(-1)
    generated_real = _unnormalize_target(generated_norm, stats).reshape(-1)

    low_index = int(torch.argmin(mse).item())
    high_index = int(torch.argmax(mse).item())
    selected = [low_index, high_index]
    labels = ["Low-loss validation case", "High-loss validation case"]
    summaries = []
    for label, index in zip(labels, selected, strict=True):
        record = records[index]
        summaries.append(
            {
                "label": label,
                "split": record.split,
                "split_index": record.split_index,
                "dataset_index": record.dataset_index,
                "folder": record.folder,
                "target": float(targets[index].item()),
                "teacher_scalar_head_prediction": float(teacher_real[index].item()),
                "generated_scalar_head_prediction": float(generated_real[index].item()),
                "generated_minus_target": float(
                    generated_real[index].item() - targets[index].item()
                ),
                "latent_mse_normalized": float(mse[index].item()),
                "latent_mae_normalized": float(mae[index].item()),
                "latent_cosine_normalized": float(cosine[index].item()),
                "generated_latent_mean_std_normalized": float(
                    generated_std[index].mean().item()
                ),
                "noise_samples": noise_samples,
            }
        )
    return summaries, selected


def _imshow_with_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    values: np.ndarray,
    *,
    title: str,
    vlim: float,
) -> None:
    image = ax.imshow(
        values,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vlim,
        vmax=vlim,
        origin="lower",
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("latent channel")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.045)


def _plot_examples(
    output_path: Path,
    *,
    config: dict[str, Any],
    summaries: list[dict[str, Any]],
    selected_indices: list[int],
    latents: torch.Tensor,
    generated_mean: torch.Tensor,
) -> None:
    num_tokens = int(config["num_tokens"])
    latent_dim = int(config["latent_dim"])
    layer_text = "last-layer" if config["layers"] == "last" else "all-layer"
    title = (
        "Conditional Generator Latents vs Ground Truth Transolver Latents\n"
        f"validation examples, normalized {layer_text} {config['token_kind']} "
        f"[{num_tokens} tokens x {latent_dim} channels]"
    )

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.35), constrained_layout=True)
    fig.suptitle(title, fontsize=18)

    latent_values = torch.cat(
        [latents[selected_indices], generated_mean[selected_indices]], dim=0
    )
    latent_limit = max(3.5, float(torch.quantile(latent_values.abs(), 0.995).item()))
    diff_values = generated_mean[selected_indices] - latents[selected_indices]
    diff_limit = max(1.0, float(torch.quantile(diff_values.abs(), 0.995).item()))

    for row, (summary, index) in enumerate(zip(summaries, selected_indices, strict=True)):
        ground_truth = latents[index].numpy()
        generated = generated_mean[index].numpy()
        diff = generated - ground_truth
        sample_name = Path(summary["folder"]).name

        _imshow_with_colorbar(
            fig,
            axes[row, 0],
            ground_truth,
            title="Ground-truth latent",
            vlim=latent_limit,
        )
        axes[row, 0].set_ylabel("latent token")
        axes[row, 0].text(
            0.02,
            0.98,
            "\n".join(
                [
                    summary["label"],
                    f"sample: {sample_name}",
                    f"target: {summary['target']:.4f}",
                    f"scalar(real): {summary['teacher_scalar_head_prediction']:.4f}",
                    f"scalar(gen): {summary['generated_scalar_head_prediction']:.4f}",
                ]
            ),
            transform=axes[row, 0].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

        _imshow_with_colorbar(
            fig,
            axes[row, 1],
            generated,
            title=f"Generated latent\nmean of {summary['noise_samples']} noise draws",
            vlim=latent_limit,
        )
        axes[row, 1].set_ylabel("latent token")

        _imshow_with_colorbar(
            fig,
            axes[row, 2],
            diff,
            title="Generated - ground truth",
            vlim=diff_limit,
        )
        axes[row, 2].set_ylabel("latent token")
        axes[row, 2].text(
            0.02,
            0.98,
            "\n".join(
                [
                    f"latent MSE: {summary['latent_mse_normalized']:.4f}",
                    f"latent MAE: {summary['latent_mae_normalized']:.4f}",
                    f"cosine: {summary['latent_cosine_normalized']:.4f}",
                    f"mean sample std: {summary['generated_latent_mean_std_normalized']:.4f}",
                ]
            ),
            transform=axes[row, 2].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="latent_generator_* run directory")
    parser.add_argument("--checkpoint", default="generator_best.pt")
    parser.add_argument("--noise-samples", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-name",
        default="generated_vs_ground_truth_latents_val_examples.png",
    )
    parser.add_argument(
        "--json-name",
        default="generated_vs_ground_truth_latents_val_examples.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    config = json.loads((run_dir / "config.json").read_text())
    device = torch.device(args.device)
    torch.manual_seed(int(config["args"]["seed"]))

    scalar_head, generator, stats = _load_models(
        run_dir / args.checkpoint,
        config=config,
        device=device,
    )
    records, conditions, latents, targets = _load_validation_tensors(config, stats=stats)
    generated_mean, generated_std = _generate_mean_latents(
        generator,
        conditions,
        noise_samples=args.noise_samples,
        noise_std=float(config["args"]["noise_std"]),
        device=device,
    )
    summaries, selected_indices = _evaluate_examples(
        scalar_head,
        stats,
        records,
        latents,
        targets,
        generated_mean,
        generated_std,
        noise_samples=args.noise_samples,
        device=device,
    )
    _plot_examples(
        run_dir / args.output_name,
        config=config,
        summaries=summaries,
        selected_indices=selected_indices,
        latents=latents,
        generated_mean=generated_mean,
    )
    (run_dir / args.json_name).write_text(json.dumps(summaries, indent=2))
    print(f"Wrote {run_dir / args.output_name}")
    print(f"Wrote {run_dir / args.json_name}")


if __name__ == "__main__":
    main()
