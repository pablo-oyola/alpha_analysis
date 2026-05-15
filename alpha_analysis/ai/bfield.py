"""Helpers to evaluate DESC magnetic field components on analysis profile grids."""

import os


def _configure_jax_env() -> None:
    """Set conservative JAX defaults without overriding an explicit user choice."""

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("JAX_LOG_COMPILES", "0")

    platforms = os.environ.get("JAX_PLATFORMS", "")
    platform_name = os.environ.get("JAX_PLATFORM_NAME", "")
    explicit_jax_platform = (
        "JAX_PLATFORMS" in os.environ or "JAX_PLATFORM_NAME" in os.environ
    )
    use_gpu = os.environ.get("ALPHA_ANALYSIS_USE_GPU", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    cpu_requested = (
        platform_name.lower() == "cpu"
        or any(platform.strip().lower() == "cpu" for platform in platforms.split(","))
    )

    if cpu_requested or (not explicit_jax_platform and not use_gpu):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        os.environ.setdefault("JAX_SKIP_CUDA_CONSTRAINTS_CHECK", "1")
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
DEFAULT_COORDINATE_MAPPING_TOL = 1.0e-8
DEFAULT_COORDINATE_MAPPING_MAXITER = 100
DEFAULT_GEOMETRIC_THETA_SCAN_POINTS = 64
DEFAULT_GEOMETRIC_THETA_ITERATIONS = 10

_TWO_PI = 2.0 * np.pi
_ROOT_TOL = 1.0e-12
_AXIS_RHO_TOL = 1.0e-12
_DESC_COORDINATES = {"rtz", "desc", "flux", "rho_theta_zeta"}
_PROFILE_COORDINATES = {"rtp", "rho_theta_phi"}
_PROFILE_CYLINDRICAL_THETA_COORDINATES = {
    "rtcp",
    "rho_theta_cyl_phi",
    "rho_theta_cylindrical_phi",
    "rho_geometric_theta_phi",
}
_CYLINDRICAL_COORDINATES = {
    "rpz",
    "cylindrical",
    "physical",
    "r_phi_z",
    "rphiz",
    "R_phi_Z",
}


def get_runtime_diagnostics() -> Dict[str, object]:
    """Return lightweight runtime diagnostics for DESC/JAX backend selection."""

    diagnostics: Dict[str, object] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS"),
        "jax_platform_name_env": os.environ.get("JAX_PLATFORM_NAME"),
        "jax_skip_cuda_constraints_check_env": os.environ.get(
            "JAX_SKIP_CUDA_CONSTRAINTS_CHECK"
        ),
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
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    names: Tuple[str, str, str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        return np.broadcast_arrays(
            np.asarray(first, dtype=np.float64),
            np.asarray(second, dtype=np.float64),
            np.asarray(third, dtype=np.float64),
        )
    except ValueError as exc:
        first_name, second_name, third_name = names
        raise ValueError(
            f"{first_name}, {second_name}, and {third_name} must have matching "
            f"or broadcast-compatible shapes. Received {np.shape(first)}, "
            f"{np.shape(second)}, and {np.shape(third)}."
        ) from exc


def _normalize_coordinates(coordinates: str) -> str:
    normalized = coordinates.strip().lower()
    if normalized in _DESC_COORDINATES:
        return "rtz"
    if normalized in _PROFILE_COORDINATES:
        return "rtp"
    if normalized in _PROFILE_CYLINDRICAL_THETA_COORDINATES:
        return "rtcp"
    if normalized in {value.lower() for value in _CYLINDRICAL_COORDINATES}:
        return "rpz"
    allowed = sorted(
        _DESC_COORDINATES
        | _PROFILE_COORDINATES
        | _PROFILE_CYLINDRICAL_THETA_COORDINATES
        | _CYLINDRICAL_COORDINATES
    )
    raise ValueError(
        f"coordinates must be one of {allowed}. Received {coordinates!r}."
    )


def _coordinate_names(coordinates: str) -> Tuple[str, str, str]:
    if coordinates == "rpz":
        return "R", "phi", "Z"
    if coordinates == "rtcp":
        return "rho", "theta_cyl", "phi"
    if coordinates == "rtp":
        return "rho", "theta", "phi"
    return "rho", "theta", "zeta"


def _nfp(eq) -> int:
    return max(int(getattr(eq, "NFP", 1)), 1)


def _wrapped_angle_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return wrapped angle difference ``a - b`` in ``[-pi, pi)``."""

    return (np.asarray(a) - np.asarray(b) + np.pi) % _TWO_PI - np.pi


def _compute_rz_from_rtp(
    eq,
    rho: np.ndarray,
    theta_desc: np.ndarray,
    phi: np.ndarray,
    *,
    tol: float = DEFAULT_COORDINATE_MAPPING_TOL,
    maxiter: int = DEFAULT_COORDINATE_MAPPING_MAXITER,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate physical R/Z from ``(rho, theta_DESC, phi_cyl)``."""

    try:
        import desc.grid as dscg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "desc is required to evaluate magnetic fields from the DESC equilibrium. "
            "Install that dependency in the active environment first."
        ) from exc

    desc_nodes = _map_to_desc_coordinates(
        eq,
        rho,
        theta_desc,
        phi,
        "rtp",
        tol=tol,
        maxiter=maxiter,
    )
    data = eq.compute(["R", "Z"], grid=dscg.Grid(desc_nodes, NFP=_nfp(eq)))
    return (
        np.asarray(data["R"], dtype=np.float64),
        np.asarray(data["Z"], dtype=np.float64),
    )


def _axis_rz_from_phi(
    eq,
    phi: np.ndarray,
    *,
    tol: float = DEFAULT_COORDINATE_MAPPING_TOL,
    maxiter: int = DEFAULT_COORDINATE_MAPPING_MAXITER,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate magnetic-axis R/Z at physical toroidal angle ``phi``."""

    unique_phi, inverse = np.unique(phi, return_inverse=True)
    r_axis, z_axis = _compute_rz_from_rtp(
        eq,
        np.zeros_like(unique_phi),
        np.zeros_like(unique_phi),
        unique_phi,
        tol=tol,
        maxiter=maxiter,
    )
    return r_axis[inverse], z_axis[inverse]


def find_desc_theta_for_cylindrical_theta(
    eq,
    rho: np.ndarray,
    theta_cyl: np.ndarray,
    phi: np.ndarray,
    *,
    n_scan: int = DEFAULT_GEOMETRIC_THETA_SCAN_POINTS,
    n_refine: int = DEFAULT_GEOMETRIC_THETA_ITERATIONS,
    mapping_tol: float = DEFAULT_COORDINATE_MAPPING_TOL,
    mapping_maxiter: int = DEFAULT_COORDINATE_MAPPING_MAXITER,
) -> np.ndarray:
    """Find DESC ``theta`` for ``(rho, theta_cyl, phi_cyl)`` points."""

    if n_scan < 4:
        raise ValueError("n_scan must be at least 4.")
    if n_refine < 0:
        raise ValueError("n_refine must be non-negative.")

    rho, theta_cyl, phi = _broadcast_coordinates(
        rho,
        theta_cyl,
        phi,
        names=("rho", "theta_cyl", "phi"),
    )
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
    r_axis, z_axis = _axis_rz_from_phi(
        eq,
        phi_work,
        tol=mapping_tol,
        maxiter=mapping_maxiter,
    )

    def residual(theta_desc_values: np.ndarray) -> np.ndarray:
        r_point, z_point = _compute_rz_from_rtp(
            eq,
            rho_work,
            theta_desc_values,
            phi_work,
            tol=mapping_tol,
            maxiter=mapping_maxiter,
        )
        theta_geom = np.arctan2(z_point - z_axis, r_point - r_axis)
        return _wrapped_angle_difference(theta_geom, theta_cyl_work)

    theta_scan = np.linspace(0.0, _TWO_PI, n_scan + 1)
    previous_theta = theta_scan[0]
    previous_values = residual(np.full_like(rho_work, previous_theta))
    previous_abs = np.abs(previous_values)

    best_theta = np.full_like(rho_work, previous_theta)
    best_abs_value = np.where(np.isfinite(previous_abs), previous_abs, np.inf)

    found = np.zeros(rho_work.shape, dtype=bool)
    lower = np.zeros_like(rho_work)
    upper = np.zeros_like(rho_work)
    lower_values = np.zeros_like(rho_work)

    for current_theta in theta_scan[1:]:
        current_values = residual(np.full_like(rho_work, current_theta))
        current_abs = np.abs(current_values)
        current_abs_finite = np.where(np.isfinite(current_abs), current_abs, np.inf)
        improve = current_abs_finite < best_abs_value
        best_theta[improve] = current_theta
        best_abs_value[improve] = current_abs_finite[improve]

        previous_finite = np.isfinite(previous_values)
        current_finite = np.isfinite(current_values)
        previous_exact = previous_finite & (np.abs(previous_values) <= _ROOT_TOL)
        current_exact = current_finite & (current_abs <= _ROOT_TOL)
        wrapped_branch_jump = (
            (np.abs(previous_values) > 0.5 * np.pi)
            & (np.abs(current_values) > 0.5 * np.pi)
        )
        sign_change = (
            previous_finite
            & current_finite
            & (previous_values * current_values < 0.0)
            & ~wrapped_branch_jump
        )
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
        r_point, z_point = _compute_rz_from_rtp(
            eq,
            rho_work[refine_indices],
            midpoint,
            phi_work[refine_indices],
            tol=mapping_tol,
            maxiter=mapping_maxiter,
        )
        midpoint_geom = np.arctan2(
            z_point - z_axis[refine_indices],
            r_point - r_axis[refine_indices],
        )
        midpoint_values = _wrapped_angle_difference(
            midpoint_geom,
            theta_cyl_work[refine_indices],
        )

        exact = np.isfinite(midpoint_values) & (
            np.abs(midpoint_values) <= _ROOT_TOL
        )
        exact_indices = refine_indices[exact]
        lower[exact_indices] = midpoint[exact]
        upper[exact_indices] = midpoint[exact]
        lower_values[exact_indices] = 0.0

        remaining = ~exact & np.isfinite(midpoint_values)
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


def find_desc_theta_for_profile_grid(
    eq,
    rho: np.ndarray,
    theta_cyl: np.ndarray,
    phi: np.ndarray,
    *,
    n_scan: int = DEFAULT_GEOMETRIC_THETA_SCAN_POINTS,
    n_refine: int = DEFAULT_GEOMETRIC_THETA_ITERATIONS,
    mapping_tol: float = DEFAULT_COORDINATE_MAPPING_TOL,
    mapping_maxiter: int = DEFAULT_COORDINATE_MAPPING_MAXITER,
) -> np.ndarray:
    """Find DESC ``theta`` on a tensor-product ``rho/theta_cyl/phi`` profile grid."""

    rho = np.asarray(rho, dtype=np.float64)
    theta_cyl = np.asarray(theta_cyl, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    if rho.ndim != 1 or theta_cyl.ndim != 1 or phi.ndim != 1:
        raise ValueError("rho, theta_cyl, and phi profile coordinates must be 1D.")
    if n_scan < 4:
        raise ValueError("n_scan must be at least 4.")
    if n_refine < 0:
        raise ValueError("n_refine must be non-negative.")

    out_shape = (rho.size, theta_cyl.size, phi.size)
    if 0 in out_shape:
        return np.empty(out_shape, dtype=np.float64)

    rho2, phi2 = np.meshgrid(rho, phi, indexing="ij")
    pair_shape = rho2.shape
    rho_pairs = rho2.ravel()
    phi_pairs = phi2.ravel()

    r_axis_phi, z_axis_phi = _axis_rz_from_phi(
        eq,
        phi,
        tol=mapping_tol,
        maxiter=mapping_maxiter,
    )
    r_axis = np.broadcast_to(r_axis_phi[np.newaxis, :], pair_shape)
    z_axis = np.broadcast_to(z_axis_phi[np.newaxis, :], pair_shape)
    theta_cyl_target = np.mod(theta_cyl, _TWO_PI)[np.newaxis, :, np.newaxis]

    theta_desc = np.zeros(out_shape, dtype=np.float64)
    active_rho = np.abs(rho) > _AXIS_RHO_TOL
    if not np.any(active_rho):
        return theta_desc

    def residual(theta_desc_values: np.ndarray) -> np.ndarray:
        r_point, z_point = _compute_rz_from_rtp(
            eq,
            rho_pairs,
            theta_desc_values,
            phi_pairs,
            tol=mapping_tol,
            maxiter=mapping_maxiter,
        )
        theta_geom = np.arctan2(
            z_point.reshape(pair_shape) - z_axis,
            r_point.reshape(pair_shape) - r_axis,
        )
        return _wrapped_angle_difference(
            theta_geom[:, np.newaxis, :],
            theta_cyl_target,
        )

    theta_scan = np.linspace(0.0, _TWO_PI, n_scan + 1)
    previous_theta = theta_scan[0]
    previous_values = residual(np.full_like(rho_pairs, previous_theta))
    previous_values[~active_rho, :, :] = np.nan
    previous_abs = np.abs(previous_values)

    best_theta = np.full(out_shape, previous_theta, dtype=np.float64)
    best_abs_value = np.where(np.isfinite(previous_abs), previous_abs, np.inf)

    found = np.zeros(out_shape, dtype=bool)
    lower = np.zeros(out_shape, dtype=np.float64)
    upper = np.zeros(out_shape, dtype=np.float64)
    lower_values = np.zeros(out_shape, dtype=np.float64)

    for current_theta in theta_scan[1:]:
        current_values = residual(np.full_like(rho_pairs, current_theta))
        current_values[~active_rho, :, :] = np.nan
        current_abs = np.abs(current_values)
        current_abs_finite = np.where(np.isfinite(current_abs), current_abs, np.inf)
        improve = current_abs_finite < best_abs_value
        best_theta[improve] = current_theta
        best_abs_value[improve] = current_abs_finite[improve]

        previous_finite = np.isfinite(previous_values)
        current_finite = np.isfinite(current_values)
        previous_exact = previous_finite & (np.abs(previous_values) <= _ROOT_TOL)
        current_exact = current_finite & (current_abs <= _ROOT_TOL)
        wrapped_branch_jump = (
            (np.abs(previous_values) > 0.5 * np.pi)
            & (np.abs(current_values) > 0.5 * np.pi)
        )
        sign_change = (
            previous_finite
            & current_finite
            & (previous_values * current_values < 0.0)
            & ~wrapped_branch_jump
        )
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

        refine_indices = np.nonzero(refine)
        midpoint = 0.5 * (lower[refine_indices] + upper[refine_indices])
        rho_refine = rho[refine_indices[0]]
        phi_refine = phi[refine_indices[2]]
        r_axis_refine = r_axis[refine_indices[0], refine_indices[2]]
        z_axis_refine = z_axis[refine_indices[0], refine_indices[2]]
        r_point, z_point = _compute_rz_from_rtp(
            eq,
            rho_refine,
            midpoint,
            phi_refine,
            tol=mapping_tol,
            maxiter=mapping_maxiter,
        )
        midpoint_geom = np.arctan2(
            z_point - z_axis_refine,
            r_point - r_axis_refine,
        )
        midpoint_values = _wrapped_angle_difference(
            midpoint_geom,
            theta_cyl[refine_indices[1]],
        )

        exact = np.isfinite(midpoint_values) & (
            np.abs(midpoint_values) <= _ROOT_TOL
        )
        exact_indices = tuple(index[exact] for index in refine_indices)
        lower[exact_indices] = midpoint[exact]
        upper[exact_indices] = midpoint[exact]
        lower_values[exact_indices] = 0.0

        remaining = ~exact & np.isfinite(midpoint_values)
        if not np.any(remaining):
            continue

        remaining_indices = tuple(index[remaining] for index in refine_indices)
        midpoint_remaining = midpoint[remaining]
        midpoint_values_remaining = midpoint_values[remaining]
        keep_left = (
            lower_values[remaining_indices] * midpoint_values_remaining <= 0.0
        )

        left_indices = tuple(index[keep_left] for index in remaining_indices)
        upper[left_indices] = midpoint_remaining[keep_left]

        right_indices = tuple(index[~keep_left] for index in remaining_indices)
        lower[right_indices] = midpoint_remaining[~keep_left]
        lower_values[right_indices] = midpoint_values_remaining[~keep_left]

    theta_desc = best_theta
    theta_desc[found] = 0.5 * (lower[found] + upper[found])
    theta_desc[~active_rho, :, :] = 0.0
    return np.mod(theta_desc, _TWO_PI)


def _map_to_desc_coordinates(
    eq,
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    coordinates: str,
    *,
    tol: float = DEFAULT_COORDINATE_MAPPING_TOL,
    maxiter: int = DEFAULT_COORDINATE_MAPPING_MAXITER,
) -> np.ndarray:
    """Map supported coordinate triples to DESC ``(rho, theta, zeta)`` nodes."""

    nodes = np.stack([first.ravel(), second.ravel(), third.ravel()], axis=-1)
    if coordinates == "rtz" or nodes.size == 0:
        return nodes

    nfp = _nfp(eq)
    if coordinates == "rpz":
        inbasis = ("R", "phi", "Z")
        period = (np.inf, _TWO_PI / nfp, np.inf)
        guess = np.zeros_like(nodes)
        guess[:, 0] = 0.5
        guess[:, 1] = np.mod(np.arctan2(nodes[:, 2], nodes[:, 0]), _TWO_PI)
        guess[:, 2] = nodes[:, 1]
    else:
        inbasis = ("rho", "theta", "phi")
        period = (np.inf, _TWO_PI, _TWO_PI / nfp)
        guess = nodes.copy()
        if coordinates == "rtcp":
            theta_desc = find_desc_theta_for_cylindrical_theta(
                eq,
                first,
                second,
                third,
                mapping_tol=tol,
                mapping_maxiter=maxiter,
            )
            return _map_to_desc_coordinates(
                eq,
                first,
                theta_desc,
                third,
                "rtp",
                tol=tol,
                maxiter=maxiter,
            )

    mapped = eq.map_coordinates(
        coords=nodes,
        inbasis=inbasis,
        outbasis=("rho", "theta", "zeta"),
        period=period,
        guess=guess,
        tol=tol,
        maxiter=maxiter,
    )
    return np.asarray(mapped, dtype=np.float64)


def _evaluate_bfield_from_desc_nodes(
    eq,
    desc_nodes: np.ndarray,
    shape: Tuple[int, ...],
    dtype: np.dtype,
) -> Dict[str, np.ndarray]:
    try:
        import desc.grid as dscg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "desc is required to evaluate magnetic fields from the DESC equilibrium. "
            "Install that dependency in the active environment first."
        ) from exc

    grid = dscg.Grid(desc_nodes, coordinates="rtz", NFP=_nfp(eq))
    data = eq.compute(["B_R", "B_phi", "B_Z", "rho", "theta", "zeta"], grid=grid)
    return {
        "br": np.asarray(data["B_R"], dtype=dtype).reshape(shape),
        "bphi": np.asarray(data["B_phi"], dtype=dtype).reshape(shape),
        "bz": np.asarray(data["B_Z"], dtype=dtype).reshape(shape),
        "rho_desc": np.asarray(data["rho"], dtype=np.float64).reshape(shape),
        "theta_desc": np.asarray(data["theta"], dtype=np.float64).reshape(shape),
        "zeta_desc": np.asarray(data["zeta"], dtype=np.float64).reshape(shape),
    }


def interpolate_desc_bfield(
    equilibrium_path: PathLike,
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    *,
    dtype: np.dtype = np.float32,
    coordinates: str = "rtz",
    mapping_tol: float = DEFAULT_COORDINATE_MAPPING_TOL,
    mapping_maxiter: int = DEFAULT_COORDINATE_MAPPING_MAXITER,
) -> Dict[str, np.ndarray]:
    """Evaluate DESC magnetic field components at requested coordinates.

    ``coordinates`` controls how ``first``, ``second``, and ``third`` are
    interpreted:

    - ``"rtz"``: DESC computational coordinates ``(rho, theta, zeta)``.
    - ``"rtp"``: ``(rho, theta_DESC, phi)`` with physical toroidal angle ``phi``.
    - ``"rtcp"``: ``(rho, theta_cyl, phi)`` where ``theta_cyl`` is geometric
      cylindrical poloidal angle about the magnetic axis.
    - ``"rpz"``: physical cylindrical coordinates ``(R, phi, Z)``.

    Non-``rtz`` coordinates are mapped to DESC coordinates with DESC's native
    ``map_coordinates`` implementation before evaluating ``B_R``, ``B_phi``,
    and ``B_Z``.
    """

    coordinates = _normalize_coordinates(coordinates)
    first_name, second_name, third_name = _coordinate_names(coordinates)
    first, second, third = _broadcast_coordinates(
        first,
        second,
        third,
        names=(first_name, second_name, third_name),
    )
    shape = first.shape
    if first.size == 0:
        return {
            "br": np.empty(shape, dtype=dtype),
            "bphi": np.empty(shape, dtype=dtype),
            "bz": np.empty(shape, dtype=dtype),
            "rho_desc": np.empty(shape, dtype=np.float64),
            "theta_desc": np.empty(shape, dtype=np.float64),
            "zeta_desc": np.empty(shape, dtype=np.float64),
        }

    eq = _load_desc_equilibrium(equilibrium_path)
    desc_nodes = _map_to_desc_coordinates(
        eq,
        first,
        second,
        third,
        coordinates,
        tol=mapping_tol,
        maxiter=mapping_maxiter,
    )
    return _evaluate_bfield_from_desc_nodes(eq, desc_nodes, shape, dtype)


def interpolate_desc_bfield_rpz(
    equilibrium_path: PathLike,
    R: np.ndarray,
    phi: np.ndarray,
    Z: np.ndarray,
    *,
    dtype: np.dtype = np.float32,
    mapping_tol: float = DEFAULT_COORDINATE_MAPPING_TOL,
    mapping_maxiter: int = DEFAULT_COORDINATE_MAPPING_MAXITER,
) -> Dict[str, np.ndarray]:
    """Evaluate DESC magnetic field at physical cylindrical ``(R, phi, Z)`` points."""

    return interpolate_desc_bfield(
        equilibrium_path,
        R,
        phi,
        Z,
        dtype=dtype,
        coordinates="rpz",
        mapping_tol=mapping_tol,
        mapping_maxiter=mapping_maxiter,
    )


def interpolate_desc_bfield_on_profile_grid(
    analysis_path: PathLike,
    equilibrium_path: PathLike,
    *,
    dtype: np.dtype = np.float32,
    coordinates: str = "rtcp",
    mapping_tol: float = DEFAULT_COORDINATE_MAPPING_TOL,
    mapping_maxiter: int = DEFAULT_COORDINATE_MAPPING_MAXITER,
    theta_scan_points: int = DEFAULT_GEOMETRIC_THETA_SCAN_POINTS,
    theta_iterations: int = DEFAULT_GEOMETRIC_THETA_ITERATIONS,
) -> Dict[str, np.ndarray]:
    """Evaluate DESC magnetic field components on the ``analysis_results.h5`` profile grid.

    The profile axes are stored as ``(rho, theta_cyl, phi_cyl)``. By default
    ``theta`` is treated as the geometric/cylindrical poloidal angle about the
    magnetic axis, while ``phi`` is treated as the physical cylindrical toroidal
    angle. DESC handles the final mapping to computational ``(rho, theta, zeta)``.
    """

    rho_1d, theta_1d, phi_1d = read_profile_grid(analysis_path)
    rho, theta, phi = np.meshgrid(rho_1d, theta_1d, phi_1d, indexing="ij")
    coordinates = _normalize_coordinates(coordinates)
    eq = _load_desc_equilibrium(equilibrium_path)
    if coordinates == "rtcp":
        theta = find_desc_theta_for_profile_grid(
            eq,
            rho_1d,
            theta_1d,
            phi_1d,
            n_scan=theta_scan_points,
            n_refine=theta_iterations,
            mapping_tol=mapping_tol,
            mapping_maxiter=mapping_maxiter,
        )
        coordinates = "rtp"
    desc_nodes = _map_to_desc_coordinates(
        eq,
        rho,
        theta,
        phi,
        coordinates,
        tol=mapping_tol,
        maxiter=mapping_maxiter,
    )
    return _evaluate_bfield_from_desc_nodes(eq, desc_nodes, rho.shape, dtype)


def sample_bfield_on_profile_grid(
    ascot_path: Optional[PathLike],
    analysis_path: PathLike,
    equilibrium_path: PathLike,
    *,
    dtype: np.dtype = np.float32,
    coordinates: str = "rtcp",
    mapping_tol: float = DEFAULT_COORDINATE_MAPPING_TOL,
    mapping_maxiter: int = DEFAULT_COORDINATE_MAPPING_MAXITER,
    theta_scan_points: int = DEFAULT_GEOMETRIC_THETA_SCAN_POINTS,
    theta_iterations: int = DEFAULT_GEOMETRIC_THETA_ITERATIONS,
) -> Dict[str, np.ndarray]:
    """Backward-compatible wrapper for DESC field evaluation on the profile grid."""

    if ascot_path is not None:
        Path(ascot_path).expanduser()

    return interpolate_desc_bfield_on_profile_grid(
        analysis_path=analysis_path,
        equilibrium_path=equilibrium_path,
        dtype=dtype,
        coordinates=coordinates,
        mapping_tol=mapping_tol,
        mapping_maxiter=mapping_maxiter,
        theta_scan_points=theta_scan_points,
        theta_iterations=theta_iterations,
    )


__all__ = [
    "DEFAULT_ASCOT_FILENAME",
    "ASCOT_FILENAME_CANDIDATES",
    "DEFAULT_COORDINATE_MAPPING_MAXITER",
    "DEFAULT_COORDINATE_MAPPING_TOL",
    "DEFAULT_GEOMETRIC_THETA_ITERATIONS",
    "DEFAULT_GEOMETRIC_THETA_SCAN_POINTS",
    "PROFILE_GRID_DATASETS",
    "find_desc_theta_for_cylindrical_theta",
    "get_runtime_diagnostics",
    "interpolate_desc_bfield",
    "interpolate_desc_bfield_rpz",
    "interpolate_desc_bfield_on_profile_grid",
    "read_profile_grid",
    "sample_bfield_on_profile_grid",
]
