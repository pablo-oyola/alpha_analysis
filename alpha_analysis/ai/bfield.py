"""Helpers to evaluate DESC magnetic field components on analysis profile grids."""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import h5py
import numpy as np

PathLike = Union[str, Path]

DEFAULT_ASCOT_FILENAME = "ascot_output.h5"
ASCOT_FILENAME_CANDIDATES = ("ascot_output.h5", "ascot_results.h5")
PROFILE_GRID_DATASETS = ("profiles/rho", "profiles/theta", "profiles/phi")


def _load_desc_equilibrium(equilibrium_path: PathLike):
    try:
        import desc.io as dscio
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "desc is required to evaluate magnetic fields from the DESC equilibrium. "
            "Install that dependency in the active environment first."
        ) from exc

    equilibrium_path = Path(equilibrium_path).expanduser()
    try:
        fam = dscio.load(str(equilibrium_path), file_format="hdf5")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load DESC equilibrium from {equilibrium_path}. "
            "Ensure the active environment has the full DESC runtime installed."
        ) from exc
    try:
        return fam[-1]
    except Exception:
        return fam


def read_profile_grid(analysis_path: PathLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the profile-grid center coordinates from ``analysis_results.h5``."""

    analysis_path = Path(analysis_path).expanduser()
    with h5py.File(analysis_path, "r") as analysis_file:
        missing = [name for name in PROFILE_GRID_DATASETS if name not in analysis_file]
        if missing:
            raise KeyError(
                f"Missing profile-grid datasets in {analysis_path}: {', '.join(missing)}"
            )
        rho = np.asarray(analysis_file["profiles/rho"][...], dtype=np.float64)
        theta = np.asarray(analysis_file["profiles/theta"][...], dtype=np.float64)
        phi = np.asarray(analysis_file["profiles/phi"][...], dtype=np.float64)
    return rho, theta, phi


def interpolate_desc_bfield(
    equilibrium_path: PathLike,
    rho: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    dtype: np.dtype = np.float32,
) -> Dict[str, np.ndarray]:
    """Evaluate ``br``, ``bphi``, and ``bz`` at flux coordinates from a DESC equilibrium."""

    device = "gpu" if Path("/dev/nvidia0").exists() else "cpu"
    os.environ.setdefault("JAX_PLATFORMS", "gpu,cpu" if device == "gpu" else "cpu")
    os.environ.setdefault("JAX_PLATFORM_NAME", device)
    if device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    try:
        import desc.grid as dscg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "desc is required to evaluate magnetic fields from the DESC equilibrium. "
            "Install that dependency in the active environment first."
        ) from exc

    rho = np.asarray(rho, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    if not (rho.shape == theta.shape == phi.shape):
        raise ValueError(
            "rho, theta, and phi must have matching shapes. "
            f"Received {rho.shape}, {theta.shape}, and {phi.shape}."
        )

    eq = _load_desc_equilibrium(equilibrium_path)
    grid = dscg.Grid(np.stack([rho.ravel(), theta.ravel(), phi.ravel()], axis=-1))
    data = eq.compute(["B_R", "B_phi", "B_Z"], grid=grid)
    shape = rho.shape

    return {
        "br": np.asarray(data["B_R"], dtype=dtype).reshape(shape),
        "bphi": np.asarray(data["B_phi"], dtype=dtype).reshape(shape),
        "bz": np.asarray(data["B_Z"], dtype=dtype).reshape(shape),
    }


def interpolate_desc_bfield_on_profile_grid(
    analysis_path: PathLike,
    equilibrium_path: PathLike,
    *,
    dtype: np.dtype = np.float32,
) -> Dict[str, np.ndarray]:
    """Evaluate DESC magnetic field components on the ``analysis_results.h5`` profile grid."""

    rho_1d, theta_1d, phi_1d = read_profile_grid(analysis_path)
    rho, theta, phi = np.meshgrid(rho_1d, theta_1d, phi_1d, indexing="ij")
    return interpolate_desc_bfield(equilibrium_path, rho, theta, phi, dtype=dtype)


def sample_bfield_on_profile_grid(
    ascot_path: Optional[PathLike],
    analysis_path: PathLike,
    equilibrium_path: PathLike,
    *,
    dtype: np.dtype = np.float32,
) -> Dict[str, np.ndarray]:
    """Backward-compatible wrapper for DESC field evaluation on the profile grid."""

    if ascot_path is not None:
        Path(ascot_path).expanduser()

    return interpolate_desc_bfield_on_profile_grid(
        analysis_path=analysis_path,
        equilibrium_path=equilibrium_path,
        dtype=dtype,
    )


__all__ = [
    "DEFAULT_ASCOT_FILENAME",
    "ASCOT_FILENAME_CANDIDATES",
    "PROFILE_GRID_DATASETS",
    "interpolate_desc_bfield",
    "interpolate_desc_bfield_on_profile_grid",
    "read_profile_grid",
    "sample_bfield_on_profile_grid",
]
