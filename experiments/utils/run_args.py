"""Command-line options shared by the experiment scripts.

Besides the device and the dataset cache directory, every list in a preset
(seeds, learning seeds, swept values) can be replaced from the command line,
which allows running a reduced subset of a preset, e.g.

    python fig3_4_8_9_10_hardness.py --preset bi_cvr --cvr 1 3 5 --seeds 1
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


def _number(text: str) -> int | float:
    """Parse ``"1"`` as int and ``"1.5"`` as float, matching how presets are written."""
    value = float(text)
    return int(value) if value.is_integer() and "." not in text else value


def _hidden_layers(text: str) -> tuple[int, int]:
    """Parse ``"128x3"`` as (hidden_dim=128, num_layers=3)."""
    hidden_dim, num_layers = text.lower().split("x")
    return int(hidden_dim), int(num_layers)


# preset key -> (flag, parser, help)
SUBSET_OPTIONS = {
    "seeds": ("--seeds", int, "dataset seeds"),
    "learning_seeds": ("--learning-seeds", int, "learning seeds"),
    "cvr": ("--cvr", _number, "clause-to-variable ratios"),
    "c": ("--c", int, "numbers of constraint regions"),
    "n_terms": ("--n-terms", int, "numbers of terms per linear inequality"),
    "hidden_layers": ("--hidden-layers", _hidden_layers, "learner sizes, e.g. 128x3"),
    "n_samples": ("--n-samples", int, "dataset sizes"),
}


def add_run_args(
    parser: argparse.ArgumentParser, subset_keys: list[str]
) -> argparse.ArgumentParser:
    """Add --device, --results-dir and one list option per key in *subset_keys*."""
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for dataset generation and training (default: cpu)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where generated datasets are stored "
        "(default: experiments/results)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel worker processes (default: 3)",
    )
    for key in subset_keys:
        flag, parse, what = SUBSET_OPTIONS[key]
        parser.add_argument(
            flag,
            dest=key,
            type=parse,
            nargs="+",
            default=None,
            help=f"Run only these {what} instead of the preset's",
        )
    return parser


def parse_distant_args(description: str, values: str) -> argparse.Namespace:
    """Options of the distant-supervision scripts (fig7a-d_distant_*.py).

    *values* describes the swept parameter of the script (for --values).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for dataset generation and training (default: cpu; "
        "the paper's runs used a GPU)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="Run only these dataset seeds"
    )
    parser.add_argument(
        "--learning-seeds", type=int, nargs="+", help="Run only these learning seeds"
    )
    parser.add_argument(
        "--values", type=_number, nargs="+", help=f"Run only these {values}"
    )
    return parser.parse_args()


def apply_subset(preset: dict, args: argparse.Namespace) -> dict:
    """Copy of *preset* with every list given on the command line replaced."""
    preset = copy.deepcopy(preset)
    for key in SUBSET_OPTIONS:
        value = getattr(args, key, None)
        if value is not None:
            preset[key] = value
    return preset
