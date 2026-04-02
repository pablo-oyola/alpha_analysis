"""PyTorch dataset utilities for DESC/analysis result folders.

Each sample folder is expected to contain:
    - ``desc_equilibrium.h5``
    - ``analysis_results.h5``

The dataset reads:
    - ``desc_equilibrium.h5``: ``_R_lmn`` and ``_Z_lmn``
    - ``analysis_results.h5``: ``profiles/prs_para``, ``profiles/prs_perp``, and
      the loss-energy target. The observed target path in this workspace is
      ``losses/losses/energy``, but ``losses/energy`` is also supported as a
      fallback.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import h5py
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

PathLike = Union[str, Path]

DEFAULT_ANALYSIS_FILENAME = "analysis_results.h5"
DEFAULT_EQUILIBRIUM_FILENAME = "desc_equilibrium.h5"
TARGET_DATASET_CANDIDATES = ("losses/losses/energy", "losses/energy")


@dataclass(frozen=True)
class SamplePaths:
    """Filesystem locations for a single training sample."""

    folder: Path
    analysis_path: Path
    equilibrium_path: Path


def _resolve_sample_paths(
    folders: Iterable[PathLike],
    analysis_filename: str,
    equilibrium_filename: str,
    strict: bool,
) -> Tuple[List[SamplePaths], List[str]]:
    samples: List[SamplePaths] = []
    skipped: List[str] = []

    for folder in folders:
        folder_path = Path(folder).expanduser()
        try:
            if not folder_path.is_dir():
                raise FileNotFoundError(f"Sample folder does not exist: {folder_path}")
        except PermissionError as exc:
            if strict:
                raise PermissionError(f"Cannot access sample folder: {folder_path}") from exc
            skipped.append(f"{folder_path}: cannot access folder")
            continue

        analysis_path = folder_path / analysis_filename
        equilibrium_path = folder_path / equilibrium_filename

        try:
            missing_files = [
                str(path.name)
                for path in (analysis_path, equilibrium_path)
                if not path.is_file()
            ]
        except PermissionError as exc:
            message = "{}: cannot access required files ({}, {})".format(
                folder_path,
                analysis_filename,
                equilibrium_filename,
            )
            if strict:
                raise PermissionError(message) from exc
            skipped.append(message)
            continue
        if missing_files:
            message = f"{folder_path}: missing required files: {', '.join(missing_files)}"
            if strict:
                raise FileNotFoundError(message)
            skipped.append(message)
            continue

        samples.append(
            SamplePaths(
                folder=folder_path,
                analysis_path=analysis_path,
                equilibrium_path=equilibrium_path,
            )
        )

    if not samples:
        details = f" Skipped entries: {skipped}" if skipped else ""
        raise ValueError(f"No readable samples were found from the provided folders.{details}")

    return samples, skipped


def _read_required_dataset(h5_file: h5py.File, dataset_name: str) -> Tensor:
    if dataset_name not in h5_file:
        raise KeyError(f"Missing dataset '{dataset_name}' in {h5_file.filename}")
    return torch.as_tensor(h5_file[dataset_name][...], dtype=torch.float32)


def _read_target_dataset(h5_file: h5py.File) -> Tensor:
    for dataset_name in TARGET_DATASET_CANDIDATES:
        if dataset_name in h5_file:
            return torch.as_tensor(h5_file[dataset_name][...], dtype=torch.float32)
    raise KeyError(
        f"Missing target dataset in {h5_file.filename}. "
        f"Tried: {', '.join(TARGET_DATASET_CANDIDATES)}"
    )


def _pad_tensor_batch(
    tensors: Sequence[Tensor],
    pad_value: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pad a list of equal-rank tensors to the maximum shape in the batch."""

    if not tensors:
        raise ValueError("Cannot pad an empty tensor batch.")

    rank = tensors[0].ndim
    for tensor in tensors:
        if tensor.ndim != rank:
            raise ValueError(
                "All tensors in a batch must have the same rank. "
                f"Observed ranks: {[item.ndim for item in tensors]}"
            )

    max_shape = [max(tensor.shape[dim] for tensor in tensors) for dim in range(rank)]

    padded_tensors: List[Tensor] = []
    masks: List[Tensor] = []
    shapes: List[Tensor] = []
    for tensor in tensors:
        shapes.append(torch.tensor(tensor.shape, dtype=torch.long))

        pad_widths: List[int] = []
        for dim in reversed(range(rank)):
            pad_widths.extend((0, max_shape[dim] - tensor.shape[dim]))
        padded_tensors.append(F.pad(tensor, pad_widths, value=pad_value))

        mask = torch.zeros(max_shape, dtype=torch.bool)
        mask[tuple(slice(0, size) for size in tensor.shape)] = True
        masks.append(mask)

    return (
        torch.stack(padded_tensors, dim=0),
        torch.stack(masks, dim=0),
        torch.stack(shapes, dim=0),
    )


class Ascot5Dataset(Dataset):
    """Lazy dataset for simulation folders containing DESC and analysis HDF5 files."""

    def __init__(
        self,
        folders: Iterable[PathLike],
        *,
        analysis_filename: str = DEFAULT_ANALYSIS_FILENAME,
        equilibrium_filename: str = DEFAULT_EQUILIBRIUM_FILENAME,
        strict: bool = True,
    ) -> None:
        self.samples, self.skipped_folders = _resolve_sample_paths(
            folders=folders,
            analysis_filename=analysis_filename,
            equilibrium_filename=equilibrium_filename,
            strict=strict,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_paths = self.samples[index]
        with h5py.File(sample_paths.equilibrium_path, "r") as equilibrium_file, h5py.File(
            sample_paths.analysis_path, "r"
        ) as analysis_file:
            r_lmn = _read_required_dataset(equilibrium_file, "_R_lmn")
            z_lmn = _read_required_dataset(equilibrium_file, "_Z_lmn")
            prs_para = _read_required_dataset(analysis_file, "profiles/prs_para")
            prs_perp = _read_required_dataset(analysis_file, "profiles/prs_perp")
            target = _read_target_dataset(analysis_file)

        return {
            "folder": str(sample_paths.folder),
            "prs_para": prs_para,
            "prs_perp": prs_perp,
            "context": {
                "R_lmn": r_lmn,
                "Z_lmn": z_lmn,
            },
            "target": target,
        }


def simulation_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad variable-length tensors and create masks for batching."""

    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    prs_para_batch, prs_para_mask, prs_para_shapes = _pad_tensor_batch(
        [item["prs_para"] for item in batch]  # type: ignore[index]
    )
    prs_perp_batch, prs_perp_mask, prs_perp_shapes = _pad_tensor_batch(
        [item["prs_perp"] for item in batch]  # type: ignore[index]
    )
    r_lmn_batch, r_lmn_mask, r_lmn_shapes = _pad_tensor_batch(
        [item["context"]["R_lmn"] for item in batch]  # type: ignore[index]
    )
    z_lmn_batch, z_lmn_mask, z_lmn_shapes = _pad_tensor_batch(
        [item["context"]["Z_lmn"] for item in batch]  # type: ignore[index]
    )
    target_batch, target_mask, target_shapes = _pad_tensor_batch(
        [item["target"] for item in batch]  # type: ignore[index]
    )

    if prs_para_batch.shape != prs_perp_batch.shape:
        raise ValueError(
            "profiles/prs_para and profiles/prs_perp must have matching shapes within a batch. "
            "Received padded shapes {} and {}.".format(prs_para_batch.shape, prs_perp_batch.shape)
        )

    return {
        "folders": [item["folder"] for item in batch],
        "input_channels": ("prs_para", "prs_perp"),
        "inputs": torch.stack((prs_para_batch, prs_perp_batch), dim=1),
        "input_mask": torch.stack((prs_para_mask, prs_perp_mask), dim=1),
        "context": {
            "R_lmn": r_lmn_batch,
            "R_lmn_mask": r_lmn_mask,
            "Z_lmn": z_lmn_batch,
            "Z_lmn_mask": z_lmn_mask,
        },
        "target": target_batch,
        "target_mask": target_mask,
        "shapes": {
            "prs_para": prs_para_shapes,
            "prs_perp": prs_perp_shapes,
            "R_lmn": r_lmn_shapes,
            "Z_lmn": z_lmn_shapes,
            "target": target_shapes,
        },
    }


def build_simulation_dataloader(
    folders: Iterable[PathLike],
    *,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    strict: bool = True,
    analysis_filename: str = DEFAULT_ANALYSIS_FILENAME,
    equilibrium_filename: str = DEFAULT_EQUILIBRIUM_FILENAME,
) -> DataLoader:
    """Create a DataLoader for a list of simulation result folders."""

    dataset = Ascot5Dataset(
        folders,
        analysis_filename=analysis_filename,
        equilibrium_filename=equilibrium_filename,
        strict=strict,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=simulation_collate_fn,
    )


__all__ = [
    "DEFAULT_ANALYSIS_FILENAME",
    "DEFAULT_EQUILIBRIUM_FILENAME",
    "Ascot5Dataset",
    "TARGET_DATASET_CANDIDATES",
    "build_simulation_dataloader",
    "simulation_collate_fn",
]
