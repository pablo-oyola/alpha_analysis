"""Helpers to sample magnetic field components on analysis profile grids."""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import h5py
import numpy as np

PathLike = Union[str, Path]

DEFAULT_ASCOT_FILENAME = "ascot_output.h5"
ASCOT_FILENAME_CANDIDATES = ("ascot_output.h5", "ascot_results.h5")
PROFILE_GRID_DATASETS = ("profiles/rho", "profiles/theta", "profiles/phi")


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


def sample_bfield_on_profile_grid(
    ascot_path: Optional[PathLike],
    analysis_path: PathLike,
    equilibrium_path: PathLike,
    *,
    dtype: np.dtype = np.float32,
) -> Dict[str, np.ndarray]:
    """Evaluate ``br``, ``bphi``, and ``bz`` on the profile grid stored in analysis results.

    The current implementation samples the field directly from the DESC
    equilibrium on the analysis profile grid. ``ascot_path`` is accepted for
    forwards compatibility with a future direct ASCOT-file backend.
    """

    device = "gpu" if Path("/dev/nvidia0").exists() else "cpu"
    os.environ.setdefault("JAX_PLATFORMS", "gpu,cpu" if device == "gpu" else "cpu")
    os.environ.setdefault("JAX_PLATFORM_NAME", device)
    if device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    try:
        import desc.grid as dscg
        import desc.io as dscio
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "desc is required to sample magnetic fields on the analysis profile "
            "grid. Install that dependency in the active environment first."
        ) from exc

    analysis_path = Path(analysis_path).expanduser()
    equilibrium_path = Path(equilibrium_path).expanduser()
    if ascot_path is not None:
        ascot_path = Path(ascot_path).expanduser()

    rho, theta, phi_flux = read_profile_grid(analysis_path)
    rho_3d, theta_3d, phi_3d = np.meshgrid(rho, theta, phi_flux, indexing="ij")

    fam = dscio.load(str(equilibrium_path), file_format="hdf5")
    try:
        eq = fam[-1]
    except Exception:
        eq = fam
    grid = dscg.Grid(np.stack([rho_3d.ravel(), theta_3d.ravel(), phi_3d.ravel()], axis=-1))
    data = eq.compute(["B_R", "B_phi", "B_Z"], grid=grid)
    shape = rho_3d.shape

    return {
        "br": np.asarray(data["B_R"], dtype=dtype).reshape(shape),
        "bphi": np.asarray(data["B_phi"], dtype=dtype).reshape(shape),
        "bz": np.asarray(data["B_Z"], dtype=dtype).reshape(shape),
    }


__all__ = [
    "DEFAULT_ASCOT_FILENAME",
    "ASCOT_FILENAME_CANDIDATES",
    "PROFILE_GRID_DATASETS",
    "read_profile_grid",
    "sample_bfield_on_profile_grid",
]
