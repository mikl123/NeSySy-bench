"""Run the paper's experiments and draw the paper's figures and tables from them.

    python run.py smoke                 quick check that everything works
    python run.py subset [WHAT ...]     representative subset of the results
    python run.py full   [WHAT ...]     all runs behind the paper's results

WHAT selects paper results (fig3 fig4 fig7 fig8 fig9 fig10 tab1 tab2 tab3 tab4) or
single experiments (cnf_cvr bi_cvr li_cvr cnf_regions bi_regions li_regions cnf_io_cvr
bi_io_cvr li_io_cvr li_n_terms cnf_methods cnf_train_size bi_methods tab1 distant_cvr
distant_weight distant_gen_depth distant_learner_depth); default: all.
Results go to output/<mode>/: mlflow.db (all runs), datasets/, logs/ (one log per
experiment), timings.csv and paper_results/ (the paper's figures and tables, drawn by
the notebooks in experiments/analysis/).
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = {  # experiments/<file>
    1: "fig3_4_8_9_10_hardness.py",
    2: "tables2_3_4_methods.py",
    5: "fig7a_distant_cvr.py",
    6: "table1_fixing_vs_filtering.py",
    7: "fig7b_distant_constraint_weight.py",
    8: "fig7c_distant_generator_depth.py",
    9: "fig7d_distant_learner_depth.py",
}
NOTEBOOKS = sorted((ROOT / "experiments" / "analysis").glob("*.ipynb"))
RESULTS = [
    "fig3",
    "fig4",
    "fig7",
    "fig8",
    "fig9",
    "fig10",
    "tab1",
    "tab2",
    "tab3",
    "tab4",
]


@dataclass
class Job:
    results: tuple[str, ...]  # paper results this job feeds
    script: int  # key of SCRIPTS
    preset: str
    args: list[str] = field(default_factory=list)
    minutes: int = 0  # rough duration on a laptop (Apple M4, 3 worker processes)

    @property
    def name(self) -> str:
        return self.preset


# Every preset without overrides: exactly the runs behind the paper's results.
FULL = [
    Job(("fig3", "fig8", "fig9"), 1, "cnf_cvr", minutes=15),
    Job(("fig3", "fig8"), 1, "bi_cvr", minutes=400),
    Job(("fig3", "fig8"), 1, "li_cvr", minutes=200),
    Job(("fig4",), 1, "cnf_regions", minutes=8),
    Job(("fig4",), 1, "bi_regions", minutes=120),
    Job(("fig4",), 1, "li_regions", minutes=40),
    Job(("fig8",), 1, "cnf_io_cvr", minutes=8),
    Job(("fig8",), 1, "bi_io_cvr", minutes=110),
    Job(("fig8",), 1, "li_io_cvr", minutes=25),
    Job(("fig10",), 1, "li_n_terms", minutes=50),
    Job(("tab2",), 2, "cnf_methods", minutes=480),
    Job(("tab3",), 2, "cnf_train_size", minutes=720),
    Job(("tab4",), 2, "bi_methods", minutes=1200),
    Job(("tab1",), 6, "tab1", minutes=90),
    Job(("fig7",), 5, "distant_cvr", minutes=600),
    Job(("fig7",), 7, "distant_weight", minutes=240),
    Job(("fig7",), 8, "distant_gen_depth", minutes=300),
    Job(("fig7",), 9, "distant_learner_depth", minutes=300),
]

# Representative subset: Figs. 3a/3b, 7a and Tables 2 and 4 with every x value (row) of
# the paper but fewer dataset seeds, training seeds and learner sizes; the CNF sweeps of
# Figs. 3a, 4a, 8d and 9 in full. The other results are reproduced by the full run only.
SUBSET = [
    Job(("fig3", "fig8", "fig9"), 1, "cnf_cvr", minutes=15),
    Job(
        ("fig3", "fig8"),
        1,
        "bi_cvr",
        ["--seeds", "1", "2", "--learning-seeds", "1", "2"],
        minutes=135,
    ),
    Job(("fig4",), 1, "cnf_regions", minutes=8),
    Job(("fig8",), 1, "cnf_io_cvr", minutes=8),
    Job(
        ("tab2",),
        2,
        "cnf_methods",
        ["--hidden-layers", "128x3", "--learning-seeds", "1"],
        minutes=30,
    ),
    Job(
        ("tab4",),
        2,
        "bi_methods",
        [
            "--seeds",
            "10",
            "20",
            "30",
            "--hidden-layers",
            "128x3",
            "--learning-seeds",
            "1",
        ],
        minutes=145,
    ),
    Job(
        ("fig7",),
        5,
        "distant_cvr",
        [
            "--values",
            *["0.4", "0.8", "1.2", "1.6", "2.0", "2.4", "2.8", "3.2", "3.6", "4.0"],
            "--seeds",
            "10",
            "20",
            "30",
            "--learning-seeds",
            "1",
        ],
        minutes=60,
    ),
]

# Smoke test: one run each of a Boolean, a real-valued and a method-comparison experiment.
SMOKE = [
    Job(
        ("fig3",),
        1,
        "cnf_cvr",
        ["--cvr", "2", "--seeds", "1", "--learning-seeds", "1"],
        minutes=1,
    ),
    Job(
        ("fig3",),
        1,
        "li_cvr",
        ["--cvr", "1", "--seeds", "1", "--learning-seeds", "1"],
        minutes=5,
    ),
    Job(
        ("tab2",),
        2,
        "cnf_methods",
        [
            "--cvr",
            "1",
            "--seeds",
            "10",
            "--hidden-layers",
            "128x3",
            "--learning-seeds",
            "1",
        ],
        minutes=5,
    ),
]
MODES = {"smoke": SMOKE, "subset": SUBSET, "full": FULL}


TEST_2_PRESETS = {"cnf_methods", "cnf_train_size", "bi_methods"}
IO_PRESETS = {"cnf_io_cvr", "bi_io_cvr", "li_io_cvr"}


def count_failures(db: Path, preset: str) -> tuple[int, int]:
    """Failed tasks of a preset: (expected, unexpected).

    Expected (as in the paper): for input-output constraints with a large cvr no
    constraint satisfiable for all inputs exists, so the dataset cannot be generated
    and the models of that configuration cannot run. Everything else is an error.
    """
    base = preset if preset in TEST_2_PRESETS else f"sup_train_{preset}"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        errors = [
            row[0] or ""
            for row in con.execute(
                "select (select value from tags t2 where t2.run_uuid = r.run_uuid "
                "and t2.key = 'error') from runs r "
                "join experiments e on e.experiment_id = r.experiment_id "
                "join tags t on t.run_uuid = r.run_uuid and t.key = 'status' "
                "and t.value = 'failed' where e.name in (?, ?)",
                (f"{base}_datasets", base),
            )
        ]
    finally:
        con.close()
    expected = sum(
        e.startswith("Failed to generate")
        or (
            e.startswith("[Errno 2] No such file or directory") and preset in IO_PRESETS
        )
        for e in errors
    )
    return expected, len(errors) - expected


def run_notebooks(out: Path, mode: str) -> list[str]:
    """Execute the notebooks of experiments/analysis on the runs in *out*."""
    dest = out / "paper_results"
    dest.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "NESYSY_RUNS": str(out),
        # the paper's figures show only points with all 25 runs (full runs)
        "NESYSY_MIN_RUNS": "25" if mode == "full" else "1",
    }
    failed = []
    for nb in NOTEBOOKS:
        cmd = [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--output-dir",
            str(dest),
            str(nb),
        ]
        with open(out / "logs" / f"notebook_{nb.stem}.log", "w") as fh:
            if subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT) != 0:
                failed.append(nb.name)
    return failed


def run_job(
    job: Job, out: Path, device: str, workers: int, threads: int
) -> tuple[int, float]:
    script = [sys.executable, SCRIPTS[job.script]]
    if job.script in (1, 2):
        cmd = [
            *script,
            "--preset",
            job.preset,
            *job.args,
            "--device",
            device,
            "--results-dir",
            str(out / "datasets"),
            "--workers",
            str(workers),
        ]
    elif job.script == 6:  # Table 1 writes its runs to a CSV file
        cmd = [*script, *job.args, "--out-csv", str(out / "tab1_runs.csv")]
    else:  # Figure 7, distant supervision
        cmd = [*script, *job.args, "--device", device]
    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": f"sqlite:///{out / 'mlflow.db'}",
        "PYTHONUNBUFFERED": "1",
        # same number of threads per worker process on every machine
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
    }
    log = out / "logs" / f"{job.name}.log"
    start = time.time()
    with open(log, "w") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        rc = subprocess.call(
            cmd, cwd=ROOT / "experiments", env=env, stdout=fh, stderr=subprocess.STDOUT
        )
    return rc, time.time() - start


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=MODES)
    parser.add_argument(
        "what",
        nargs="*",
        metavar="WHAT",
        help="paper results or experiments to run (default: all)",
    )
    parser.add_argument(
        "--out", type=Path, help="output directory (default: output/<mode>)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="the paper's Figure 7 was computed on a GPU, all other results on CPU; "
        "everything also runs on CPU (default: cpu)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="parallel worker processes per experiment (default: 3)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="threads per worker process (default: 4, as for the paper's runs)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted run: skip the experiments that already finished",
    )
    args = parser.parse_args()
    presets = [j.preset for j in FULL]
    unknown = sorted(set(args.what) - set(RESULTS) - set(presets))
    if unknown:
        parser.error(
            f"unknown {' '.join(unknown)}; choose results from {' '.join(RESULTS)} "
            f"or experiments from {' '.join(presets)}"
        )

    out = (args.out or ROOT / "output" / args.mode).resolve()
    wanted = set(args.what)
    jobs = [
        j for j in MODES[args.mode] if not wanted or wanted & {*j.results, j.preset}
    ]
    if not jobs:
        sys.exit(
            f"nothing to run: {args.mode} has no experiment for {' '.join(sorted(wanted))}; "
            "./run_full.sh runs every result"
        )
    timings = out / "timings.csv"
    done = set()
    if timings.exists():
        if not args.resume:
            sys.exit(
                f"{out} already holds results. Use --resume to continue, or --out for a new directory."
            )
        with open(timings) as fh:
            done = {
                row["experiment"]
                for row in csv.DictReader(fh)
                if row["exit_code"] == "0"
            }
    (out / "logs").mkdir(parents=True, exist_ok=True)
    if not timings.exists():
        timings.write_text("experiment,arguments,seconds,exit_code\n")

    todo = [j for j in jobs if j.name not in done]
    total = sum(j.minutes for j in todo)
    print(
        f"{args.mode}: {len(todo)} experiment(s): {' '.join(j.name for j in todo)}; roughly "
        f"{total // 60} h {total % 60} min on a recent laptop with 3 workers. Output: {out}"
    )
    failed = []
    for i, job in enumerate(todo, 1):
        print(
            f"[{i}/{len(todo)}] {time.strftime('%H:%M:%S')} {job.preset} {' '.join(job.args)} "
            f"(~{job.minutes} min, log: logs/{job.name}.log)",
            flush=True,
        )
        rc, seconds = run_job(job, out, args.device, args.workers, args.threads)
        with open(timings, "a") as fh:
            fh.write(f"{job.name},\"{' '.join(job.args)}\",{seconds:.0f},{rc}\n")
        n_expected, n_errors = (
            count_failures(out / "mlflow.db", job.preset)
            if (out / "mlflow.db").exists()
            else (0, 0)
        )
        if rc != 0 or n_errors:
            failed.append(job.name)
            print(
                f"    FAILED: exit code {rc}, {n_errors} failed run(s); see logs/{job.name}.log"
            )
        else:
            note = (
                f" ({n_expected} configuration(s) without a satisfiable constraint, as in the paper)"
                if n_expected
                else ""
            )
            print(f"    done in {seconds / 60:.1f} min{note}")

    print("\nDrawing the paper's figures and tables ...", flush=True)
    failed += run_notebooks(out, args.mode)
    if failed:
        sys.exit(f"\nERROR: {len(failed)} step(s) had errors: {' '.join(failed)}")
    print(f"\nAll {len(todo)} experiment(s) finished without errors.")
    print(f"The paper's figures and tables are in {out / 'paper_results'}")


if __name__ == "__main__":
    main()
