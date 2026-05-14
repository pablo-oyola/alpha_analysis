#!/usr/bin/env python3
"""Export DESC magnetic-field samples onto analysis profile grids."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alpha_analysis.ai.bfield import (  # noqa: E402
    get_runtime_diagnostics,
    interpolate_desc_bfield_on_profile_grid,
    read_profile_grid,
)

LOGGER = logging.getLogger("export_bfield_results")


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
    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
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
) -> None:
    rho, theta, phi = read_profile_grid(analysis_path)
    fields = interpolate_desc_bfield_on_profile_grid(
        analysis_path=analysis_path,
        equilibrium_path=equilibrium_path,
    )

    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("rho", data=rho)
        h5f.create_dataset("theta", data=theta)
        h5f.create_dataset("phi", data=phi)
        for name in ("br", "bphi", "bz"):
            h5f.create_dataset(
                name,
                data=fields[name],
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
        h5f.attrs["analysis_results_path"] = str(analysis_path)
        h5f.attrs["desc_equilibrium_path"] = str(equilibrium_path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    log_runtime_diagnostics()

    try:
        shard_index, num_shards = resolve_sharding(args)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2

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

    wrote = 0
    skipped = 0
    failed = 0

    for folder in directories:
        analysis_path = folder / args.analysis_filename
        equilibrium_path = folder / args.equilibrium_filename
        output_path = folder / args.output_filename

        if output_path.exists() and not args.overwrite:
            skipped += 1
            LOGGER.info("Skipping existing %s", output_path)
            continue

        try:
            write_bfield_file(
                output_path,
                analysis_path=analysis_path,
                equilibrium_path=equilibrium_path,
            )
        except Exception:
            failed += 1
            LOGGER.exception("Failed to process %s", folder)
            continue

        wrote += 1
        LOGGER.info("Wrote %s", output_path)

    LOGGER.info(
        "Done: wrote=%d skipped=%d failed=%d",
        wrote,
        skipped,
        failed,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
