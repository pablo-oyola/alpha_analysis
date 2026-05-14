"""Helpers to evaluate DESC magnetic field components on analysis profile grids."""

import os


def _configure_jax_env() -> None:
    """Set conservative JAX defaults without overriding an explicit user choice."""

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("JAX_LOG_COMPILES", "0")
    if "CUDA_VISIBLE_DEVICES" not in os.environ and os.path.exists("/dev/nvidia0"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if "CUDA_VISIBLE_DEVICES" not in os.environ and not os.path.exists("/dev/nvidia0"):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


_configure_jax_env()

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import h5py
import numpy as np

PathLike = Union[str, Path]

DEFAULT_ASCOT_FILENAME = "ascot_output.h5"
ASCOT_FILENAME_CANDIDATES = ("ascot_output.h5", "ascot_results.h5")
PROFILE_GRID_DATASETS = ("profiles/rho", "profiles/theta", "profiles/phi")
DEFAULT_THETA_INVERSION_SCAN_POINTS = 256
DEFAULT_THETA_INVERSION_ITERATIONS = 32

_TWO_PI = 2.0 * np.pi
_ROOT_TOL = 1.0e-12
_AXIS_RHO_TOL = 1.0e-12
_DESC_THETA_COORDINATES = {"desc", "flux", "theta_desc"}
_GEOMETRIC_THETA_COORDINATES = {
    "geometric",
    "cylindrical",
    "cyl",
    "theta_cyl",
    "theta_geom",
}


def get_runtime_diagnostics() -> Dict[str, object]:
    """Return lightweight runtime diagnostics for DESC/JAX backend selection."""

    diagnostics: Dict[str, object] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS"),
        "jax_platform_name_env": os.environ.get("JAX_PLATFORM_NAME"),
        "nvidia0_exists": os.path.exists("/dev/nvidia0"),
    }
    try:
        import jax
    except Exception as exc:  # pragma: no cover - purely diagnostic path
        diagnostics["jax_import_error"] = repr(exc)
        return diagnostics

    try:
        diagnostics["jax_default_backend"] = jax.default_backend()
        diagnostics["jax_devices"] = [str(device) for device in jax.devices()]
    except Exception as exc:  # pragma: no cover - purely diagnostic path
        diagnostics["jax_backend_error"] = repr(exc)
    return diagnostics


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


def _broadcast_coordinates(
    rho: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        return np.broadcast_arrays(
            np.asarray(rho, dtype=np.float64),
            np.asarray(theta, dtype=np.float64),
            np.asarray(phi, dtype=np.float64),
        )
    except ValueError as exc:
        raise ValueError(
            "rho, theta, and phi must have matching or broadcast-compatible shapes. "
            f"Received {np.shape(rho)}, {np.shape(theta)}, and {np.shape(phi)}."
        ) from exc


def _normalize_theta_coordinate(theta_coordinate: str) -> str:
    normalized = theta_coordinate.strip().lower()
    if normalized in _GEOMETRIC_THETA_COORDINATES:
        return "geometric"
    if normalized in _DESC_THETA_COORDINATES:
        return "desc"
    allowed = sorted(_GEOMETRIC_THETA_COORDINATES | _DESC_THETA_COORDINATES)
    raise ValueError(
        f"theta_coordinate must be one of {allowed}. Received {theta_coordinate!r}."
    )


def _wrapped_angle_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return wrapped angle difference ``a - b`` in ``[-pi, pi)``."""

    return (np.asarray(a) - np.asarray(b) + np.pi) % _TWO_PI - np.pi


def _compute_axis_rz(eq, phi: np.ndarray, grid_cls) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate magnetic-axis R/Z at the requested cylindrical toroidal angles."""

    if phi.size == 0:
        empty = np.empty_like(phi, dtype=np.float64)
        return empty, empty

    unique_phi, inverse = np.unique(phi, return_inverse=True)
    axis_nodes = np.stack(
        [
            np.zeros_like(unique_phi),
            np.zeros_like(unique_phi),
            unique_phi,
        ],
        axis=-1,
    )
    data = eq.compute(["R", "Z"], grid=grid_cls(axis_nodes))
    r_axis = np.asarray(data["R"], dtype=np.float64)[inverse]
    z_axis = np.asarray(data["Z"], dtype=np.float64)[inverse]
    return r_axis, z_axis


def _compute_geometric_theta(
    eq,
    rho: np.ndarray,
    theta_desc: np.ndarray,
    phi: np.ndarray,
    r_axis: np.ndarray,
    z_axis: np.ndarray,
    grid_cls,
) -> np.ndarray:
    nodes = np.stack([rho, theta_desc, phi], axis=-1)
    data = eq.compute(["R", "Z"], grid=grid_cls(nodes))
    r_point = np.asarray(data["R"], dtype=np.float64)
    z_point = np.asarray(data["Z"], dtype=np.float64)
    return np.arctan2(z_point - z_axis, r_point - r_axis)


def geometric_theta_from_desc(
    eq,
    rho: np.ndarray,
    theta_desc: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Return geometric/cylindrical poloidal angle for DESC coordinates.

    The geometric angle is measured in the local R-Z plane around the magnetic
    axis at the same toroidal angle ``phi``.
    """

    try:
        import desc.grid as dscg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "desc is required to evaluate magnetic fields from the DESC equilibrium. "
            "Install that dependency in the active environment first."
        ) from exc

    rho, theta_desc, phi = _broadcast_coordinates(rho, theta_desc, phi)
    shape = rho.shape
    rho_flat = rho.ravel()
    theta_flat = theta_desc.ravel()
    phi_flat = phi.ravel()
    r_axis, z_axis = _compute_axis_rz(eq, phi_flat, dscg.Grid)
    theta_geom = _compute_geometric_theta(
        eq,
        rho_flat,
        theta_flat,
        phi_flat,
        r_axis,
        z_axis,
        dscg.Grid,
    )
    return theta_geom.reshape(shape)


def find_desc_theta_for_geometric_theta(
    eq,
    rho: np.ndarray,
    theta_cyl: np.ndarray,
    phi: np.ndarray,
    *,
    n_scan: int = DEFAULT_THETA_INVERSION_SCAN_POINTS,
    n_refine: int = DEFAULT_THETA_INVERSION_ITERATIONS,
) -> np.ndarray:
    """Invert cylindrical/geometric poloidal angle to DESC poloidal coordinate.

    ``rho`` remains DESC's flux-surface radial coordinate. For each requested
    ``(rho, theta_cyl, phi)``, this solves for the DESC ``theta`` whose point on
    that flux surface has geometric angle ``theta_cyl`` about the magnetic axis
    at the same ``phi``. If no clean bracket is found, the closest scan point is
    used as a fallback, which is useful near angle branch cuts or non-star-shaped
    surfaces.
    """

    if n_scan < 4:
        raise ValueError("n_scan must be at least 4.")
    if n_refine < 0:
        raise ValueError("n_refine must be non-negative.")

    try:
        import desc.grid as dscg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "desc is required to evaluate magnetic fields from the DESC equilibrium. "
            "Install that dependency in the active environment first."
        ) from exc

    rho, theta_cyl, phi = _broadcast_coordinates(rho, theta_cyl, phi)
    shape = rho.shape
    theta_desc = np.zeros(rho.size, dtype=np.float64)
    if rho.size == 0:
        return theta_desc.reshape(shape)

    rho_flat = rho.ravel()
    theta_cyl_flat = np.mod(theta_cyl.ravel(), _TWO_PI)
    phi_flat = phi.ravel()

    active_indices = np.nonzero(np.abs(rho_flat) > _AXIS_RHO_TOL)[0]
    if active_indices.size == 0:
        return theta_desc.reshape(shape)

    rho_work = rho_flat[active_indices]
    theta_cyl_work = theta_cyl_flat[active_indices]
    phi_work = phi_flat[active_indices]
    r_axis, z_axis = _compute_axis_rz(eq, phi_work, dscg.Grid)

    def residual(theta_desc_values: np.ndarray) -> np.ndarray:
        theta_geom = _compute_geometric_theta(
            eq,
            rho_work,
            theta_desc_values,
            phi_work,
            r_axis,
            z_axis,
            dscg.Grid,
        )
        return _wrapped_angle_difference(theta_geom, theta_cyl_work)

    theta_scan = np.linspace(0.0, _TWO_PI, n_scan + 1)
    previous_theta = theta_scan[0]
    previous_values = residual(np.full_like(rho_work, previous_theta))

    best_theta = np.full_like(rho_work, previous_theta)
    best_abs_value = np.abs(previous_values)

    found = np.zeros(rho_work.shape, dtype=bool)
    lower = np.zeros_like(rho_work)
    upper = np.zeros_like(rho_work)
    lower_values = np.zeros_like(rho_work)

    for current_theta in theta_scan[1:]:
        current_values = residual(np.full_like(rho_work, current_theta))
        current_abs = np.abs(current_values)
        improve = current_abs < best_abs_value
        best_theta[improve] = current_theta
        best_abs_value[improve] = current_abs[improve]

        previous_exact = np.abs(previous_values) <= _ROOT_TOL
        current_exact = current_abs <= _ROOT_TOL
        wrapped_branch_jump = (
            (np.abs(previous_values) > 0.5 * np.pi)
            & (np.abs(current_values) > 0.5 * np.pi)
        )
        sign_change = (previous_values * current_values < 0.0) & ~wrapped_branch_jump
        bracket = (~found) & (previous_exact | current_exact | sign_change)
        if np.any(bracket):
            exact_at_previous = bracket & previous_exact
            exact_at_current = bracket & current_exact & ~previous_exact
            bracketed = bracket & ~(exact_at_previous | exact_at_current)

            lower[exact_at_previous] = previous_theta
            upper[exact_at_previous] = previous_theta
            lower_values[exact_at_previous] = 0.0

            lower[exact_at_current] = current_theta
            upper[exact_at_current] = current_theta
            lower_values[exact_at_current] = 0.0

            lower[bracketed] = previous_theta
            upper[bracketed] = current_theta
            lower_values[bracketed] = previous_values[bracketed]
            found[bracket] = True

        previous_theta = current_theta
        previous_values = current_values

    for _ in range(n_refine):
        refine = found & (upper > lower)
        if not np.any(refine):
            break

        refine_indices = np.nonzero(refine)[0]
        midpoint = 0.5 * (lower[refine_indices] + upper[refine_indices])
        midpoint_geom = _compute_geometric_theta(
            eq,
            rho_work[refine_indices],
            midpoint,
            phi_work[refine_indices],
            r_axis[refine_indices],
            z_axis[refine_indices],
            dscg.Grid,
        )
        midpoint_values = _wrapped_angle_difference(
            midpoint_geom,
            theta_cyl_work[refine_indices],
        )

        exact = np.abs(midpoint_values) <= _ROOT_TOL
        exact_indices = refine_indices[exact]
        lower[exact_indices] = midpoint[exact]
        upper[exact_indices] = midpoint[exact]
        lower_values[exact_indices] = 0.0

        remaining = ~exact
        remaining_indices = refine_indices[remaining]
        if remaining_indices.size == 0:
            continue

        midpoint_remaining = midpoint[remaining]
        midpoint_values_remaining = midpoint_values[remaining]
        keep_left = (
            lower_values[remaining_indices] * midpoint_values_remaining <= 0.0
        )

        left_indices = remaining_indices[keep_left]
        upper[left_indices] = midpoint_remaining[keep_left]

        right_indices = remaining_indices[~keep_left]
        lower[right_indices] = midpoint_remaining[~keep_left]
        lower_values[right_indices] = midpoint_values_remaining[~keep_left]

    theta_desc_work = best_theta.copy()
    theta_desc_work[found] = 0.5 * (lower[found] + upper[found])
    theta_desc[active_indices] = np.mod(theta_desc_work, _TWO_PI)
    return theta_desc.reshape(shape)


def interpolate_desc_bfield(
    equilibrium_path: PathLike,
    rho: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    dtype: np.dtype = np.float32,
    theta_coordinate: str = "geometric",
    theta_inversion_scan_points: int = DEFAULT_THETA_INVERSION_SCAN_POINTS,
    theta_inversion_iterations: int = DEFAULT_THETA_INVERSION_ITERATIONS,
) -> Dict[str, np.ndarray]:
    """Evaluate ``br``, ``bphi``, and ``bz`` from a DESC equilibrium.

    By default, ``theta`` is treated as a geometric/cylindrical poloidal angle
    measured about the magnetic axis at each ``phi``. It is inverted to DESC's
    internal poloidal coordinate before evaluating the field. Pass
    ``theta_coordinate="desc"`` if ``theta`` is already DESC's flux-surface
    poloidal coordinate.
    """

    try:
        import desc.grid as dscg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "desc is required to evaluate magnetic fields from the DESC equilibrium. "
            "Install that dependency in the active environment first."
        ) from exc

    rho, theta, phi = _broadcast_coordinates(rho, theta, phi)
    shape = rho.shape
    if rho.size == 0:
        return {
            "br": np.empty(shape, dtype=dtype),
            "bphi": np.empty(shape, dtype=dtype),
            "bz": np.empty(shape, dtype=dtype),
            "theta_desc": np.empty(shape, dtype=np.float64),
        }

    eq = _load_desc_equilibrium(equilibrium_path)
    theta_coordinate = _normalize_theta_coordinate(theta_coordinate)
    if theta_coordinate == "geometric":
        theta_desc = find_desc_theta_for_geometric_theta(
            eq,
            rho,
            theta,
            phi,
            n_scan=theta_inversion_scan_points,
            n_refine=theta_inversion_iterations,
        )
    else:
        theta_desc = np.mod(theta, _TWO_PI)

    grid = dscg.Grid(
        np.stack([rho.ravel(), theta_desc.ravel(), phi.ravel()], axis=-1)
    )
    data = eq.compute(["B_R", "B_phi", "B_Z"], grid=grid)

    return {
        "br": np.asarray(data["B_R"], dtype=dtype).reshape(shape),
        "bphi": np.asarray(data["B_phi"], dtype=dtype).reshape(shape),
        "bz": np.asarray(data["B_Z"], dtype=dtype).reshape(shape),
        "theta_desc": theta_desc.reshape(shape),
    }


def interpolate_desc_bfield_on_profile_grid(
    analysis_path: PathLike,
    equilibrium_path: PathLike,
    *,
    dtype: np.dtype = np.float32,
    theta_coordinate: str = "geometric",
    theta_inversion_scan_points: int = DEFAULT_THETA_INVERSION_SCAN_POINTS,
    theta_inversion_iterations: int = DEFAULT_THETA_INVERSION_ITERATIONS,
) -> Dict[str, np.ndarray]:
    """Evaluate DESC magnetic field components on the ``analysis_results.h5`` profile grid."""

    rho_1d, theta_1d, phi_1d = read_profile_grid(analysis_path)
    rho, theta, phi = np.meshgrid(rho_1d, theta_1d, phi_1d, indexing="ij")
    return interpolate_desc_bfield(
        equilibrium_path,
        rho,
        theta,
        phi,
        dtype=dtype,
        theta_coordinate=theta_coordinate,
        theta_inversion_scan_points=theta_inversion_scan_points,
        theta_inversion_iterations=theta_inversion_iterations,
    )


def sample_bfield_on_profile_grid(
    ascot_path: Optional[PathLike],
    analysis_path: PathLike,
    equilibrium_path: PathLike,
    *,
    dtype: np.dtype = np.float32,
    theta_coordinate: str = "geometric",
    theta_inversion_scan_points: int = DEFAULT_THETA_INVERSION_SCAN_POINTS,
    theta_inversion_iterations: int = DEFAULT_THETA_INVERSION_ITERATIONS,
) -> Dict[str, np.ndarray]:
    """Backward-compatible wrapper for DESC field evaluation on the profile grid."""

    if ascot_path is not None:
        Path(ascot_path).expanduser()

    return interpolate_desc_bfield_on_profile_grid(
        analysis_path=analysis_path,
        equilibrium_path=equilibrium_path,
        dtype=dtype,
        theta_coordinate=theta_coordinate,
        theta_inversion_scan_points=theta_inversion_scan_points,
        theta_inversion_iterations=theta_inversion_iterations,
    )


__all__ = [
    "DEFAULT_ASCOT_FILENAME",
    "ASCOT_FILENAME_CANDIDATES",
    "DEFAULT_THETA_INVERSION_ITERATIONS",
    "DEFAULT_THETA_INVERSION_SCAN_POINTS",
    "PROFILE_GRID_DATASETS",
    "find_desc_theta_for_geometric_theta",
    "geometric_theta_from_desc",
    "get_runtime_diagnostics",
    "interpolate_desc_bfield",
    "interpolate_desc_bfield_on_profile_grid",
    "read_profile_grid",
    "sample_bfield_on_profile_grid",
]
