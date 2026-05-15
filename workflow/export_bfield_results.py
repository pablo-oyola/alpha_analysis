#!/usr/bin/env python3
"""Export DESC magnetic-field samples onto analysis profile grids."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import traceback
from pathlib import Path

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alpha_analysis.ai.bfield import (  # noqa: E402
    DEFAULT_COORDINATE_MAPPING_MAXITER,
    DEFAULT_GEOMETRIC_THETA_ITERATIONS,
    DEFAULT_GEOMETRIC_THETA_SCAN_POINTS,
    get_runtime_diagnostics,
    interpolate_desc_bfield_on_profile_grid,
    read_profile_grid,
)

LOGGER = logging.getLogger("export_bfield_results")
TAG_READY = 1
TAG_WORK = 2
TAG_STOP = 3
TAG_RESULT = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Walk a results tree, evaluate DESC B_R/B_phi/B_Z on each "
            "analysis_results.h5 profile (rho, theta, phi) grid, and write "
            "bfield.h5 in each sample directory."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/global/cfs/projectdirs/m5300/results"),
        help="Top-level results directory to walk.",
    )
    parser.add_argument(
        "--analysis-filename",
        default="analysis_results.h5",
        help="Name of the analysis file expected in each sample directory.",
    )
    parser.add_argument(
        "--equilibrium-filename",
        default="desc_equilibrium.h5",
        help="Name of the DESC equilibrium file expected in each sample directory.",
    )
    parser.add_argument(
        "--output-filename",
        default="bfield.h5",
        help="Name of the HDF5 file to write in each sample directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    parser.add_argument(
        "--mpi",
        action="store_true",
        help=(
            "Run as an MPI master/worker job. Launch with srun/mpiexec; rank 0 "
            "dispatches sample directories dynamically to worker ranks."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many sample directories after discovery/sharding.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Optional shard index for array jobs (0-based).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Optional total number of shards for array jobs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--theta-scan-points",
        type=int,
        default=DEFAULT_GEOMETRIC_THETA_SCAN_POINTS,
        help=(
            "Number of coarse DESC-theta samples used to invert cylindrical "
            "profile theta. Lower is faster; higher is more robust."
        ),
    )
    parser.add_argument(
        "--theta-iterations",
        type=int,
        default=DEFAULT_GEOMETRIC_THETA_ITERATIONS,
        help=(
            "Number of bisection refinement steps after the coarse theta scan."
        ),
    )
    parser.add_argument(
        "--mapping-maxiter",
        type=int,
        default=DEFAULT_COORDINATE_MAPPING_MAXITER,
        help="Maximum Newton iterations for DESC coordinate mapping.",
    )
    return parser


def configure_logging(verbose: bool, rank: int | None = None) -> None:
    prefix = "" if rank is None else f"rank={rank} "
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=f"%(asctime)s %(levelname)s {prefix}%(message)s",
        force=True,
    )


def log_runtime_diagnostics() -> None:
    diagnostics = get_runtime_diagnostics()
    summary = " ".join(
        f"{key}={value!r}" for key, value in sorted(diagnostics.items())
    )
    LOGGER.info("Runtime diagnostics: %s", summary)


def sample_directories(
    results_root: Path,
    analysis_filename: str,
    equilibrium_filename: str,
) -> list[Path]:
    directories: list[Path] = []

    def onerror(err: OSError) -> None:
        LOGGER.warning("Skipping inaccessible path during walk: %s", err)

    for dirpath, _, filenames in os.walk(results_root, onerror=onerror):
        if analysis_filename not in filenames:
            continue
        folder = Path(dirpath)
        if (folder / equilibrium_filename).is_file():
            directories.append(folder)

    directories.sort()
    return directories


def keep_for_shard(folder: Path, shard_index: int, num_shards: int) -> bool:
    digest = hashlib.sha1(str(folder.resolve()).encode("utf-8")).hexdigest()
    shard = int(digest[:16], 16) % num_shards
    return shard == shard_index


def resolve_sharding(args: argparse.Namespace) -> tuple[int | None, int | None]:
    shard_index = args.shard_index
    num_shards = args.num_shards

    if shard_index is None and num_shards is None:
        env_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        env_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
        if env_index is not None and env_count is not None:
            shard_index = int(env_index)
            num_shards = int(env_count)

    if (shard_index is None) != (num_shards is None):
        raise ValueError("Provide both --shard-index and --num-shards, or neither.")
    if num_shards is not None:
        if num_shards <= 0:
            raise ValueError("--num-shards must be positive.")
        if shard_index is None or shard_index < 0 or shard_index >= num_shards:
            raise ValueError("Require 0 <= shard-index < num-shards.")

    return shard_index, num_shards


def write_bfield_file(
    output_path: Path,
    *,
    analysis_path: Path,
    equilibrium_path: Path,
    theta_scan_points: int,
    theta_iterations: int,
    mapping_maxiter: int,
) -> None:
    rho, theta, phi = read_profile_grid(analysis_path)
    fields = interpolate_desc_bfield_on_profile_grid(
        analysis_path=analysis_path,
        equilibrium_path=equilibrium_path,
        theta_scan_points=theta_scan_points,
        theta_iterations=theta_iterations,
        mapping_maxiter=mapping_maxiter,
    )

    tmp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        with h5py.File(tmp_path, "w") as h5f:
            h5f.create_dataset("rho", data=rho)
            h5f.create_dataset("theta", data=theta)
            h5f.create_dataset("phi", data=phi)
            for name in ("br", "bphi", "bz", "rho_desc", "theta_desc", "zeta_desc"):
                h5f.create_dataset(
                    name,
                    data=fields[name],
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
            h5f.attrs["analysis_results_path"] = str(analysis_path)
            h5f.attrs["desc_equilibrium_path"] = str(equilibrium_path)
            h5f.attrs["coordinate_system"] = "rho, theta_cyl, phi_cyl"
            h5f.attrs["theta_description"] = (
                "Geometric/cylindrical poloidal angle from analysis_results.h5."
            )
            h5f.attrs["phi_description"] = (
                "Physical cylindrical toroidal angle from analysis_results.h5."
            )
        os.replace(tmp_path, output_path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            LOGGER.warning("Could not remove temporary file %s", tmp_path)


def discover_work(args: argparse.Namespace) -> list[Path]:
    shard_index, num_shards = resolve_sharding(args)
    results_root = args.results_root.expanduser().resolve()
    directories = sample_directories(
        results_root,
        args.analysis_filename,
        args.equilibrium_filename,
    )

    if shard_index is not None and num_shards is not None:
        directories = [
            folder
            for folder in directories
            if keep_for_shard(folder, shard_index, num_shards)
        ]
        LOGGER.info(
            "Running shard %d/%d on %d sample directories.",
            shard_index,
            num_shards,
            len(directories),
        )
    else:
        LOGGER.info("Found %d sample directories.", len(directories))

    if args.limit is not None:
        directories = directories[: max(args.limit, 0)]

    return directories


def build_task(folder: Path, args: argparse.Namespace) -> dict[str, object]:
    return {
        "folder": str(folder),
        "analysis_filename": args.analysis_filename,
        "equilibrium_filename": args.equilibrium_filename,
        "output_filename": args.output_filename,
        "overwrite": bool(args.overwrite),
        "theta_scan_points": int(args.theta_scan_points),
        "theta_iterations": int(args.theta_iterations),
        "mapping_maxiter": int(args.mapping_maxiter),
    }


def process_task(task: dict[str, object]) -> dict[str, object]:
    folder = Path(str(task["folder"]))
    analysis_path = folder / str(task["analysis_filename"])
    equilibrium_path = folder / str(task["equilibrium_filename"])
    output_path = folder / str(task["output_filename"])
    overwrite = bool(task["overwrite"])

    try:
        if output_path.exists() and not overwrite:
            return {
                "status": "skipped",
                "folder": str(folder),
                "output_path": str(output_path),
                "message": "output exists",
            }

        write_bfield_file(
            output_path,
            analysis_path=analysis_path,
            equilibrium_path=equilibrium_path,
            theta_scan_points=int(task["theta_scan_points"]),
            theta_iterations=int(task["theta_iterations"]),
            mapping_maxiter=int(task["mapping_maxiter"]),
        )
        return {
            "status": "wrote",
            "folder": str(folder),
            "output_path": str(output_path),
            "message": "",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "folder": str(folder),
            "output_path": str(output_path),
            "message": repr(exc),
            "traceback": traceback.format_exc(),
        }


def log_task_result(result: dict[str, object]) -> None:
    status = result["status"]
    output_path = result["output_path"]
    if status == "wrote":
        LOGGER.info("Wrote %s", output_path)
    elif status == "skipped":
        LOGGER.info("Skipping existing %s", output_path)
    else:
        LOGGER.error(
            "Failed to process %s: %s\n%s",
            result["folder"],
            result.get("message", ""),
            result.get("traceback", ""),
        )


def summarize_results(results: list[dict[str, object]]) -> int:
    wrote = sum(1 for result in results if result["status"] == "wrote")
    skipped = sum(1 for result in results if result["status"] == "skipped")
    failed = sum(1 for result in results if result["status"] == "failed")
    LOGGER.info(
        "Done: wrote=%d skipped=%d failed=%d",
        wrote,
        skipped,
        failed,
    )
    return 1 if failed else 0


def run_serial(args: argparse.Namespace) -> int:
    log_runtime_diagnostics()
    try:
        directories = discover_work(args)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2

    wrote = 0
    skipped = 0
    failed = 0

    for folder in directories:
        result = process_task(build_task(folder, args))
        log_task_result(result)
        if result["status"] == "wrote":
            wrote += 1
        elif result["status"] == "skipped":
            skipped += 1
        else:
            failed += 1

    LOGGER.info(
        "Done: wrote=%d skipped=%d failed=%d",
        wrote,
        skipped,
        failed,
    )
    return 1 if failed else 0


def run_mpi(args: argparse.Namespace) -> int:
    configure_logging(args.verbose)
    try:
        from mpi4py import MPI
    except ModuleNotFoundError:
        LOGGER.error(
            "MPI mode requires mpi4py in the active Python environment. "
            "Install mpi4py or run without --mpi."
        )
        return 2

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    configure_logging(args.verbose, rank=rank)

    if size == 1:
        LOGGER.warning("MPI mode launched with one rank; running serially.")
        return run_serial(args)

    if rank == 0:
        return run_mpi_master(comm, args, size)

    run_mpi_worker(comm)
    return 0


def run_mpi_master(comm, args: argparse.Namespace, size: int) -> int:
    from mpi4py import MPI

    log_runtime_diagnostics()
    try:
        directories = discover_work(args)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        for worker in range(1, size):
            comm.send(None, dest=worker, tag=TAG_STOP)
        return 2

    total = len(directories)
    LOGGER.info("MPI run with %d workers for %d sample directories.", size - 1, total)
    task_iter = iter(directories)
    ready_workers = 0
    active_workers = 0
    completed = 0
    results: list[dict[str, object]] = []

    while ready_workers < size - 1:
        status = MPI.Status()
        comm.recv(source=MPI.ANY_SOURCE, tag=TAG_READY, status=status)
        worker = status.Get_source()
        ready_workers += 1
        try:
            folder = next(task_iter)
        except StopIteration:
            comm.send(None, dest=worker, tag=TAG_STOP)
        else:
            comm.send(build_task(folder, args), dest=worker, tag=TAG_WORK)
            active_workers += 1

    while active_workers:
        status = MPI.Status()
        result = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status)
        worker = status.Get_source()
        active_workers -= 1
        completed += 1
        results.append(result)
        log_task_result(result)
        if completed % 10 == 0 or completed == total:
            LOGGER.info("Progress: %d/%d complete.", completed, total)

        try:
            folder = next(task_iter)
        except StopIteration:
            comm.send(None, dest=worker, tag=TAG_STOP)
        else:
            comm.send(build_task(folder, args), dest=worker, tag=TAG_WORK)
            active_workers += 1

    return summarize_results(results)


def run_mpi_worker(comm) -> None:
    from mpi4py import MPI

    comm.send(None, dest=0, tag=TAG_READY)
    while True:
        status = MPI.Status()
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == TAG_STOP:
            return
        result = process_task(task)
        comm.send(result, dest=0, tag=TAG_RESULT)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mpi:
        return run_mpi(args)

    configure_logging(args.verbose)
    return run_serial(args)


if __name__ == "__main__":
    raise SystemExit(main())
