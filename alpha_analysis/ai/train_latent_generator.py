"""Train a scalar readout and conditional generator for exported Transolver latents."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


TokenKind = Literal["slice_tokens", "out_slice_tokens"]
LayerSelection = Literal["last", "all"]


@dataclass(frozen=True)
class ManifestRecord:
    split: str
    split_index: int
    dataset_index: int
    folder: str
    token_path: Path
    target: float
    prediction: float


@dataclass
class NormalizationStats:
    condition_mean: Tensor
    condition_std: Tensor
    latent_mean: Tensor
    latent_std: Tensor
    target_mean: Tensor
    target_std: Tensor

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "condition_mean": self.condition_mean,
            "condition_std": self.condition_std,
            "latent_mean": self.latent_mean,
            "latent_std": self.latent_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }


def _read_manifest(token_dir: Path) -> list[ManifestRecord]:
    manifest_path = token_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    records = []
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        token_path = token_dir / item["token_path"]
        records.append(
            ManifestRecord(
                split=str(item["split"]),
                split_index=int(item["split_index"]),
                dataset_index=int(item["dataset_index"]),
                folder=str(item["folder"]),
                token_path=token_path,
                target=float(item["target"]),
                prediction=float(item["prediction"]),
            )
        )
    if not records:
        raise ValueError(f"No records found in {manifest_path}")
    return records


def _finite_stats(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        float(finite.mean()),
        float(finite.std()),
        float(finite.min()),
        float(finite.max()),
        float(np.quantile(finite, 0.5)),
    ]


def _maybe_log1p(values: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return values
    return np.sign(values) * np.log1p(np.abs(values))


def _profile_stats(path: Path, dataset_name: str, *, log1p: bool) -> list[float]:
    with h5py.File(path, "r") as h5_file:
        values = np.asarray(h5_file[dataset_name][...], dtype=np.float32)
    values = _maybe_log1p(values, log1p)
    frames = values.reshape(values.shape[0], -1)
    stats: list[float] = []
    for frame in frames:
        stats.extend(_finite_stats(frame))
    stats.extend(_finite_stats(values.reshape(-1)))
    return stats


def _field_stats(path: Path, dataset_names: Sequence[str]) -> list[float]:
    stats: list[float] = []
    with h5py.File(path, "r") as h5_file:
        for dataset_name in dataset_names:
            values = np.asarray(h5_file[dataset_name][...], dtype=np.float32)
            stats.extend(_finite_stats(values.reshape(-1)))
    return stats


def _read_condition_vector(
    folder: Path,
    *,
    include_profile_stats: bool,
    include_bfield_stats: bool,
    profile_log1p: bool,
) -> Tensor:
    values: list[float] = []
    with h5py.File(folder / "desc_equilibrium.h5", "r") as equilibrium_file:
        for dataset_name in ("_R_lmn", "_Z_lmn"):
            coeffs = np.asarray(equilibrium_file[dataset_name][...], dtype=np.float32)
            values.extend(np.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0).tolist())

    if include_profile_stats:
        analysis_path = folder / "analysis_results.h5"
        values.extend(
            _profile_stats(analysis_path, "profiles/prs_para", log1p=profile_log1p)
        )
        values.extend(
            _profile_stats(analysis_path, "profiles/prs_perp", log1p=profile_log1p)
        )

    if include_bfield_stats:
        values.extend(_field_stats(folder / "bfield.h5", ("br", "bphi", "bz")))

    return torch.tensor(values, dtype=torch.float32)


def _select_latent(payload: dict[str, Any], token_kind: TokenKind, layers: LayerSelection) -> Tensor:
    tokens = payload[token_kind].float()
    if tokens.ndim != 4:
        raise ValueError(f"Expected token tensor [layers, heads, slices, dim], got {tokens.shape}")

    if layers == "last":
        selected = tokens[-1:].contiguous()
    elif layers == "all":
        selected = tokens
    else:
        raise ValueError(f"Unsupported layer selection: {layers}")

    # [L, H, S, D] -> [L, S, H * D]
    selected = selected.permute(0, 2, 1, 3).contiguous()
    return selected.reshape(selected.shape[0] * selected.shape[1], -1)


def _load_latent(path: Path, token_kind: TokenKind, layers: LayerSelection) -> Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return _select_latent(payload, token_kind, layers)


def _records_for_split(records: Sequence[ManifestRecord], split: str) -> list[ManifestRecord]:
    selected = [record for record in records if record.split == split]
    if not selected:
        raise ValueError(f"No records found for split '{split}'")
    return selected


def _split_train_records(
    records: Sequence[ManifestRecord],
    *,
    fraction: float,
    seed: int,
) -> tuple[list[ManifestRecord], list[ManifestRecord]]:
    train_records = _records_for_split(records, "train")
    val_records = [record for record in records if record.split == "val"]
    if val_records:
        return train_records, val_records

    shuffled = list(train_records)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2:
        return shuffled, shuffled
    split = max(1, min(len(shuffled) - 1, int(len(shuffled) * fraction)))
    return shuffled[:split], shuffled[split:]


def _stack_conditions(
    records: Sequence[ManifestRecord],
    *,
    include_profile_stats: bool,
    include_bfield_stats: bool,
    profile_log1p: bool,
) -> dict[str, Tensor]:
    cache: dict[str, Tensor] = {}
    for record in records:
        if record.folder in cache:
            continue
        cache[record.folder] = _read_condition_vector(
            Path(record.folder),
            include_profile_stats=include_profile_stats,
            include_bfield_stats=include_bfield_stats,
            profile_log1p=profile_log1p,
        )

    lengths = {condition.numel() for condition in cache.values()}
    if len(lengths) != 1:
        raise ValueError(f"Condition vectors have inconsistent lengths: {sorted(lengths)}")
    return cache


def _compute_condition_stats(records: Sequence[ManifestRecord], condition_cache: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    conditions = torch.stack([condition_cache[record.folder] for record in records], dim=0)
    mean = conditions.mean(dim=0)
    std = conditions.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    return mean, std


def _compute_target_stats(records: Sequence[ManifestRecord], target_name: str) -> tuple[Tensor, Tensor]:
    targets = torch.tensor(
        [getattr(record, target_name) for record in records],
        dtype=torch.float32,
    )
    mean = targets.mean()
    std = targets.std(unbiased=False).clamp_min(1.0e-6)
    return mean.reshape(1), std.reshape(1)


def _compute_latent_stats(
    records: Sequence[ManifestRecord],
    *,
    token_kind: TokenKind,
    layers: LayerSelection,
) -> tuple[Tensor, Tensor]:
    count = 0
    total: Tensor | None = None
    total_sq: Tensor | None = None
    for record in records:
        latent = _load_latent(record.token_path, token_kind, layers)
        flat = latent.reshape(-1, latent.shape[-1])
        if total is None:
            total = torch.zeros(flat.shape[-1], dtype=torch.float64)
            total_sq = torch.zeros(flat.shape[-1], dtype=torch.float64)
        total += flat.double().sum(dim=0)
        total_sq += flat.double().pow(2).sum(dim=0)
        count += flat.shape[0]

    if total is None or total_sq is None or count == 0:
        raise ValueError("Cannot compute latent stats from an empty record set.")
    mean = total / count
    variance = (total_sq / count - mean.pow(2)).clamp_min(1.0e-12)
    return mean.float(), variance.sqrt().float().clamp_min(1.0e-6)


class LatentTokenDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ManifestRecord],
        *,
        condition_cache: dict[str, Tensor],
        stats: NormalizationStats,
        token_kind: TokenKind,
        layers: LayerSelection,
        target_name: str,
    ) -> None:
        self.records = list(records)
        self.condition_cache = condition_cache
        self.stats = stats
        self.token_kind = token_kind
        self.layers = layers
        self.target_name = target_name

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        condition = self.condition_cache[record.folder]
        condition = (condition - self.stats.condition_mean) / self.stats.condition_std

        latent = _load_latent(record.token_path, self.token_kind, self.layers)
        latent = (latent - self.stats.latent_mean) / self.stats.latent_std

        target_value = torch.tensor([getattr(record, self.target_name)], dtype=torch.float32)
        target = (target_value - self.stats.target_mean) / self.stats.target_std
        return {
            "condition": condition,
            "latent": latent,
            "target": target,
            "folder": record.folder,
        }


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TokenScalarHead(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        num_tokens: int,
        width: int,
        depth: int,
        heads: int,
        dropout: float,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(token_dim, width)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.zeros(1, num_tokens + 1, width))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, latent: Tensor) -> Tensor:
        x = self.input_proj(latent)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.position[:, : x.shape[1]]
        x = self.encoder(x)
        return self.head(x[:, 0])


class ConditionalLatentGenerator(nn.Module):
    def __init__(
        self,
        *,
        condition_dim: int,
        latent_dim: int,
        num_tokens: int,
        noise_dim: int,
        width: int,
        condition_depth: int,
        transformer_depth: int,
        heads: int,
        dropout: float,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.num_tokens = num_tokens
        self.condition_encoder = MLP(
            condition_dim + noise_dim,
            width,
            width,
            depth=condition_depth,
            dropout=dropout,
        )
        self.query = nn.Parameter(torch.zeros(1, num_tokens, width))
        self.position = nn.Parameter(torch.zeros(1, num_tokens, width))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, latent_dim),
        )
        nn.init.trunc_normal_(self.query, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, condition: Tensor, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            noise = torch.randn(condition.shape[0], self.noise_dim, device=condition.device)
        context = self.condition_encoder(torch.cat([condition, noise], dim=-1))
        x = self.query.expand(condition.shape[0], -1, -1)
        x = x + self.position + context[:, None, :]
        x = self.encoder(x)
        return self.output_proj(x)


def _to_device(batch: dict[str, Any], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    return (
        batch["condition"].to(device),
        batch["latent"].to(device),
        batch["target"].to(device),
    )


def _unnormalize_target(value: Tensor, stats: NormalizationStats) -> Tensor:
    return value * stats.target_std.to(value.device) + stats.target_mean.to(value.device)


def _scalar_epoch(
    model: TokenScalarHead,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    stats: NormalizationStats,
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses = []
    maes = []
    rmses = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            _, latent, target = _to_device(batch, device)
            prediction = model(latent)
            loss = F.mse_loss(prediction, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            with torch.no_grad():
                pred_real = _unnormalize_target(prediction, stats)
                target_real = _unnormalize_target(target, stats)
                error = pred_real - target_real
                losses.append(float(loss.detach().cpu()))
                maes.append(float(error.abs().mean().detach().cpu()))
                rmses.append(float(error.pow(2).mean().sqrt().detach().cpu()))
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "mae": sum(maes) / max(len(maes), 1),
        "rmse": sum(rmses) / max(len(rmses), 1),
    }


def _generator_epoch(
    generator: ConditionalLatentGenerator,
    scalar_head: TokenScalarHead,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    stats: NormalizationStats,
    device: torch.device,
    latent_weight: float,
    scalar_weight: float,
    consistency_weight: float,
    noise_std: float,
) -> dict[str, float]:
    training = optimizer is not None
    generator.train(training)
    scalar_head.eval()
    losses = []
    latent_losses = []
    scalar_losses = []
    consistency_losses = []
    maes = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            condition, latent, target = _to_device(batch, device)
            noise = noise_std * torch.randn(
                condition.shape[0],
                generator.noise_dim,
                device=device,
            )
            generated = generator(condition, noise)
            latent_loss = F.mse_loss(generated, latent)
            generated_scalar = scalar_head(generated)
            with torch.no_grad():
                teacher_scalar = scalar_head(latent)
            scalar_loss = F.mse_loss(generated_scalar, target)
            consistency_loss = F.mse_loss(generated_scalar, teacher_scalar)
            loss = (
                latent_weight * latent_loss
                + scalar_weight * scalar_loss
                + consistency_weight * consistency_loss
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                optimizer.step()
            with torch.no_grad():
                pred_real = _unnormalize_target(generated_scalar, stats)
                target_real = _unnormalize_target(target, stats)
                losses.append(float(loss.detach().cpu()))
                latent_losses.append(float(latent_loss.detach().cpu()))
                scalar_losses.append(float(scalar_loss.detach().cpu()))
                consistency_losses.append(float(consistency_loss.detach().cpu()))
                maes.append(float((pred_real - target_real).abs().mean().detach().cpu()))
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "latent_mse": sum(latent_losses) / max(len(latent_losses), 1),
        "scalar_mse": sum(scalar_losses) / max(len(scalar_losses), 1),
        "consistency_mse": sum(consistency_losses) / max(len(consistency_losses), 1),
        "mae": sum(maes) / max(len(maes), 1),
    }


def _save_checkpoint(
    path: Path,
    *,
    scalar_head: TokenScalarHead,
    generator: ConditionalLatentGenerator | None,
    stats: NormalizationStats,
    config: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "scalar_head_state_dict": scalar_head.state_dict(),
        "stats": stats.state_dict(),
        "config": config,
        "metrics": metrics,
    }
    if generator is not None:
        payload["generator_state_dict"] = generator.state_dict()
    torch.save(payload, path)


def _make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return " ".join(f"{prefix}_{key}={value:.6g}" for key, value in metrics.items())


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for filename in (
        "config.json",
        "metrics.jsonl",
        "scalar_head_best.pt",
        "generator_best.pt",
        "last.pt",
    ):
        path = output_dir / filename
        if path.exists():
            path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token-dir",
        type=Path,
        required=True,
        help="Directory written by export_transolver_slice_tokens.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Training output directory. Defaults under --token-dir.",
    )
    parser.add_argument(
        "--token-kind",
        choices=("slice_tokens", "out_slice_tokens"),
        default="out_slice_tokens",
    )
    parser.add_argument("--layers", choices=("last", "all"), default="last")
    parser.add_argument(
        "--scalar-target",
        choices=("target", "prediction"),
        default="target",
        help="Train scalar head against ASCOT target or Transolver checkpoint prediction.",
    )
    parser.add_argument("--no-profile-condition", action="store_true")
    parser.add_argument("--no-bfield-condition", action="store_true")
    parser.add_argument("--no-profile-log1p", action="store_true")
    parser.add_argument("--fallback-val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scalar-epochs", type=int, default=200)
    parser.add_argument("--generator-epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--scalar-depth", type=int, default=2)
    parser.add_argument("--generator-depth", type=int, default=4)
    parser.add_argument("--condition-depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--noise-dim", type=int, default=64)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--latent-weight", type=float, default=1.0)
    parser.add_argument("--scalar-weight", type=float, default=0.1)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--skip-generator", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    token_dir = args.token_dir.expanduser().resolve()
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = token_dir / f"latent_generator_{timestamp}"
    output_dir = output_dir.expanduser().resolve()
    _prepare_output_dir(output_dir, overwrite=args.overwrite)

    records = _read_manifest(token_dir)
    train_records, val_records = _split_train_records(
        records,
        fraction=args.fallback_val_fraction,
        seed=args.seed,
    )
    condition_cache = _stack_conditions(
        records,
        include_profile_stats=not args.no_profile_condition,
        include_bfield_stats=not args.no_bfield_condition,
        profile_log1p=not args.no_profile_log1p,
    )
    condition_mean, condition_std = _compute_condition_stats(train_records, condition_cache)
    latent_mean, latent_std = _compute_latent_stats(
        train_records,
        token_kind=args.token_kind,
        layers=args.layers,
    )
    target_mean, target_std = _compute_target_stats(train_records, args.scalar_target)
    stats = NormalizationStats(
        condition_mean=condition_mean,
        condition_std=condition_std,
        latent_mean=latent_mean,
        latent_std=latent_std,
        target_mean=target_mean,
        target_std=target_std,
    )

    first_latent = _load_latent(train_records[0].token_path, args.token_kind, args.layers)
    num_tokens, latent_dim = int(first_latent.shape[0]), int(first_latent.shape[1])
    condition_dim = int(condition_mean.numel())
    config = {
        "token_dir": str(token_dir),
        "output_dir": str(output_dir),
        "token_kind": args.token_kind,
        "layers": args.layers,
        "scalar_target": args.scalar_target,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "condition_dim": condition_dim,
        "num_tokens": num_tokens,
        "latent_dim": latent_dim,
        "args": vars(args),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    train_dataset = LatentTokenDataset(
        train_records,
        condition_cache=condition_cache,
        stats=stats,
        token_kind=args.token_kind,
        layers=args.layers,
        target_name=args.scalar_target,
    )
    val_dataset = LatentTokenDataset(
        val_records,
        condition_cache=condition_cache,
        stats=stats,
        token_kind=args.token_kind,
        layers=args.layers,
        target_name=args.scalar_target,
    )
    train_loader = _make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    scalar_head = TokenScalarHead(
        token_dim=latent_dim,
        num_tokens=num_tokens,
        width=args.width,
        depth=args.scalar_depth,
        heads=args.heads,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
    ).to(device)
    scalar_optimizer = torch.optim.AdamW(
        scalar_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    metrics_path = output_dir / "metrics.jsonl"
    best_scalar_val = math.inf
    for epoch in range(1, args.scalar_epochs + 1):
        train_metrics = _scalar_epoch(
            scalar_head,
            train_loader,
            optimizer=scalar_optimizer,
            stats=stats,
            device=device,
        )
        val_metrics = _scalar_epoch(
            scalar_head,
            val_loader,
            optimizer=None,
            stats=stats,
            device=device,
        )
        metrics = {
            "stage": "scalar",
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        with metrics_path.open("a") as file:
            file.write(json.dumps(metrics) + "\n")
        if val_metrics["loss"] < best_scalar_val:
            best_scalar_val = val_metrics["loss"]
            _save_checkpoint(
                output_dir / "scalar_head_best.pt",
                scalar_head=scalar_head,
                generator=None,
                stats=stats,
                config=config,
                metrics=metrics,
            )
        if args.log_every > 0 and (epoch == 1 or epoch % args.log_every == 0):
            print(
                f"scalar epoch={epoch} "
                f"{_format_metrics('train', train_metrics)} "
                f"{_format_metrics('val', val_metrics)}"
            )

    scalar_checkpoint = torch.load(
        output_dir / "scalar_head_best.pt",
        map_location=device,
        weights_only=False,
    )
    scalar_head.load_state_dict(scalar_checkpoint["scalar_head_state_dict"])
    for parameter in scalar_head.parameters():
        parameter.requires_grad_(False)

    if args.skip_generator:
        print(f"Wrote scalar head checkpoint to {output_dir / 'scalar_head_best.pt'}")
        return

    generator = ConditionalLatentGenerator(
        condition_dim=condition_dim,
        latent_dim=latent_dim,
        num_tokens=num_tokens,
        noise_dim=args.noise_dim,
        width=args.width,
        condition_depth=args.condition_depth,
        transformer_depth=args.generator_depth,
        heads=args.heads,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
    ).to(device)
    generator_optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_generator_val = math.inf
    for epoch in range(1, args.generator_epochs + 1):
        train_metrics = _generator_epoch(
            generator,
            scalar_head,
            train_loader,
            optimizer=generator_optimizer,
            stats=stats,
            device=device,
            latent_weight=args.latent_weight,
            scalar_weight=args.scalar_weight,
            consistency_weight=args.consistency_weight,
            noise_std=args.noise_std,
        )
        val_metrics = _generator_epoch(
            generator,
            scalar_head,
            val_loader,
            optimizer=None,
            stats=stats,
            device=device,
            latent_weight=args.latent_weight,
            scalar_weight=args.scalar_weight,
            consistency_weight=args.consistency_weight,
            noise_std=args.noise_std,
        )
        metrics = {
            "stage": "generator",
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        with metrics_path.open("a") as file:
            file.write(json.dumps(metrics) + "\n")
        if val_metrics["loss"] < best_generator_val:
            best_generator_val = val_metrics["loss"]
            _save_checkpoint(
                output_dir / "generator_best.pt",
                scalar_head=scalar_head,
                generator=generator,
                stats=stats,
                config=config,
                metrics=metrics,
            )
        if args.log_every > 0 and (epoch == 1 or epoch % args.log_every == 0):
            print(
                f"generator epoch={epoch} "
                f"{_format_metrics('train', train_metrics)} "
                f"{_format_metrics('val', val_metrics)}"
            )

    _save_checkpoint(
        output_dir / "last.pt",
        scalar_head=scalar_head,
        generator=generator,
        stats=stats,
        config=config,
        metrics=metrics,
    )
    print(f"Wrote latent training outputs to {output_dir}")


if __name__ == "__main__":
    main()
