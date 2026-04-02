"""Time a single sample read from the AI dataloader."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from alpha_analysis.ai.dataloader import Ascot5Dataset, DEFAULT_ASCOT_FILENAME
else:  # pragma: no cover - exercised when run as module
    from .dataloader import Ascot5Dataset, DEFAULT_ASCOT_FILENAME


def _describe_value(value: Any) -> Any:
    shape = getattr(value, "shape", None)
    if shape is None:
        return type(value).__name__
    return tuple(int(dim) for dim in shape)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Time a single Ascot5Dataset sample read.")
    parser.add_argument("folder", type=Path, help="Path to one sample folder.")
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset index to read.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of timed reads to perform.")
    parser.add_argument("--warmup", type=int, default=0, help="Number of untimed warmup reads.")
    parser.add_argument(
        "--include-bfield",
        action="store_true",
        help="Also sample br/bphi/bz from the DESC equilibrium on the profile grid.",
    )
    parser.add_argument(
        "--analysis-filename",
        default="analysis_results.h5",
        help="Name of the analysis results file inside the sample folder.",
    )
    parser.add_argument(
        "--equilibrium-filename",
        default="desc_equilibrium.h5",
        help="Name of the DESC equilibrium file inside the sample folder.",
    )
    parser.add_argument(
        "--ascot-filename",
        default=DEFAULT_ASCOT_FILENAME,
        help="Reserved for future ASCOT-file sampling; ignored by the current backend.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")

    dataset = Ascot5Dataset(
        [args.folder],
        analysis_filename=args.analysis_filename,
        equilibrium_filename=args.equilibrium_filename,
        ascot_filename=args.ascot_filename,
        include_bfield=args.include_bfield,
        strict=True,
    )

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(
            f"--sample-index {args.sample_index} is out of bounds for dataset length {len(dataset)}"
        )

    for _ in range(args.warmup):
        _ = dataset[args.sample_index]

    timings = []
    last_sample = None
    for _ in range(args.repeat):
        start = time.perf_counter()
        last_sample = dataset[args.sample_index]
        timings.append(time.perf_counter() - start)

    assert last_sample is not None

    print(f"folder: {args.folder}")
    print(f"sample_index: {args.sample_index}")
    print(f"include_bfield: {args.include_bfield}")
    print(f"warmup_reads: {args.warmup}")
    print(f"timed_reads: {args.repeat}")
    print(f"timings_s: {[round(dt, 6) for dt in timings]}")
    print(f"min_s: {min(timings):.6f}")
    print(f"max_s: {max(timings):.6f}")
    print(f"mean_s: {sum(timings) / len(timings):.6f}")
    print(
        "sample_shapes: "
        + str(
            {
                "prs_para": _describe_value(last_sample["prs_para"]),
                "prs_perp": _describe_value(last_sample["prs_perp"]),
                "R_lmn": _describe_value(last_sample["context"]["R_lmn"]),
                "Z_lmn": _describe_value(last_sample["context"]["Z_lmn"]),
                "target": _describe_value(last_sample["target"]),
                **(
                    {
                        "br": _describe_value(last_sample["bfield"]["br"]),
                        "bphi": _describe_value(last_sample["bfield"]["bphi"]),
                        "bz": _describe_value(last_sample["bfield"]["bz"]),
                    }
                    if "bfield" in last_sample
                    else {}
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
