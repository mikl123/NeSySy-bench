"""Example: Effect of base mapping capacity on accuracy and constraint violation.

Varies hidden_dim and num_hidden of the diversity model, then trains a plain_nn
on each generated dataset and reports accuracy + constraint violation.
Runs N_RUNS per config with different seeds and plots mean +/- stderr.
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write the plot to a file, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesy_approaches.methods import ALL_METHODS, evaluate
from nesy_approaches.utils import generate_dataset
from nesy_data_gen.constraints import CNFConstraint, SMTBoundedIntervalConstraint

parser = argparse.ArgumentParser(
    description="Table 1: fixing vs filtering for different widths of the generator mapping."
)
parser.add_argument(
    "--out-csv",
    type=Path,
    default=PROJECT_ROOT / "experiments" / "results" / "tab1_runs.csv",
    help="CSV file with one row per run (default: experiments/results/tab1_runs.csv)",
)
parser.add_argument(
    "--seeds", type=int, nargs="+", help="problem seeds (default: 41-45)"
)
parser.add_argument(
    "--learning-seeds", type=int, nargs="+", help="learning seeds (default: 1 2 3)"
)
parser.add_argument(
    "--widths",
    type=int,
    nargs="+",
    help="hidden dimensions of the generator mapping (default: 128 256 512 1024)",
)
parser.add_argument(
    "--generators", choices=["filtering", "fixing"], nargs="+", help="(default: both)"
)
parser.add_argument(
    "--n-samples",
    type=int,
    default=10000,
    help="samples per dataset (default: 10000, as for the paper's Table 1)",
)
parser.add_argument(
    "--generator-lr",
    type=float,
    default=0.001,
    help="learning rate of the generator mapping (default: 0.001)",
)
ARGS = parser.parse_args()

# Choose constraint family: "cnf" or "boundary"
CONSTRAINT_TYPE = "cnf"

# One seed per problem instance (seeds both constraint generation AND dataset generation)
PROBLEM_SEEDS = ARGS.seeds or [41, 42, 43, 44, 45]
# Separate seeds for model training/initialization
LEARNING_SEEDS = ARGS.learning_seeds or [1, 2, 3]
N_RUNS = len(PROBLEM_SEEDS) * len(LEARNING_SEEDS)


def set_learning_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Base learner as in the runs of the paper's Table 1: batch 64, up to 1000 epochs,
# no dropout; the learner is seeded by set_learning_seed() before training.
TRAIN_PARAMS = {
    "batch_size": 64,
    "epochs": 1000,
    "lr": 0.001,
    "hidden_dim": 64,
    "num_layers": 3,
    "patience": 20,
    "device": "cpu",
}

CNF_PARAMS = {
    "n_outputs": 15,  # Number of output variables (boolean)
    "n_inputs": 10,  # Number of input features (not part of the CNF)
    "c": 1,  # Number of unique constraint formulas
    "b": 1,  # Regions per constraint
    "seed": 41,
    "k": 3,  # Literals per clause (3-SAT)
    "r": 1,  # Clause ratio (n_clauses = n_outputs * r); cvr = 1 in the paper's Table 1
    "use_input_variables": False,
}

BOUNDARY_PARAMS = {
    "n_outputs": 10,
    "n_inputs": 10,
    "c": 1,
    "b": 1,
    "seed": 41,
    "r": 1,  # Clause ratio (n_clauses = n_vars * r)
    "min_gap": 0.2,  # Minimum gap between interval bounds
    "use_input_variables": False,
}

# Dispatch based on CONSTRAINT_TYPE
if CONSTRAINT_TYPE == "cnf":
    CONSTRAINT_CLASS = CNFConstraint
    CONSTRAINT_PARAMS = CNF_PARAMS
elif CONSTRAINT_TYPE == "boundary":
    CONSTRAINT_CLASS = SMTBoundedIntervalConstraint
    CONSTRAINT_PARAMS = BOUNDARY_PARAMS
else:
    raise ValueError(f"Unknown CONSTRAINT_TYPE: {CONSTRAINT_TYPE}")

# Base dataset params (shared)
_COMMON_DS = {
    "n_inputs": CONSTRAINT_PARAMS["n_inputs"],
    "n_outputs": CONSTRAINT_PARAMS["n_outputs"],
    "n_samples": ARGS.n_samples,
    "exploration_set_size": 10000,  # as in the paper's Table 1 runs
    "epochs": 1000,
    "lr": ARGS.generator_lr,
    "batch_size": 128,
    "losses": [
        {"name": "soft_repulsion", "weight": 1.0},
        {"name": "constraint", "weight": 1.0, "params": {"recalc_start": 20}},
    ],
}

ALL_GENERATOR_CONFIGS = {
    "filtering": {
        "generator_type": "filtering",
        "ds_extra": {"noise_std": 0.1, "max_iterations": 100},
    },
    "fixing": {
        "generator_type": "fixing",
        "ds_extra": {"fix_noise_std": 0.1, "fix_iterations": 10},
    },
}

GENERATOR_CONFIGS = {
    name: cfg
    for name, cfg in ALL_GENERATOR_CONFIGS.items()
    if ARGS.generators is None or name in ARGS.generators
}

# Capacity configurations to sweep
CAPACITY_CONFIGS = [
    {"hidden_dim": width, "num_hidden": 3}
    for width in (ARGS.widths or [128, 256, 512, 1024])
    # earlier sweeps:
    # {"hidden_dim": 32, "num_hidden": 3},
    # {"hidden_dim": 64, "num_hidden": 3},
    # {"hidden_dim": 50, "num_hidden": 3},
    # {"hidden_dim": 64, "num_hidden": 2},
    # {"hidden_dim": 64, "num_hidden": 3},
    # {"hidden_dim": 64, "num_hidden": 4},
    # {"hidden_dim": 64, "num_hidden": 5},
]

# Collect per-run metrics: {(gen_type, label): {metric_name: [values]}}
all_runs = defaultdict(lambda: defaultdict(list))
csv_rows = []

for gen_name, gen_cfg in GENERATOR_CONFIGS.items():
    for cfg in CAPACITY_CONFIGS:
        label = f"h{cfg['hidden_dim']}_l{cfg['num_hidden']}"
        key = (gen_name, label)
        print(f"\n{'='*60}")
        print(f"Generator: {gen_name} | Config: {label}")
        print(f"{'='*60}")

        run_idx = 0
        for problem_seed in PROBLEM_SEEDS:
            # Problem seed drives BOTH constraint generation and dataset generation
            cnf = CONSTRAINT_CLASS(**{**CONSTRAINT_PARAMS, "seed": problem_seed})
            cnf.generate()

            ds_params = {
                "generator_type": gen_cfg["generator_type"],
                "ds_params": {**_COMMON_DS, **gen_cfg["ds_extra"], **cfg},
                "seed": problem_seed,
                "test_size": 0.2,
                "valid_size": 0.1,
            }

            x_train, y_train, x_valid, y_valid, x_test, y_test, _, dataset_info = (
                generate_dataset(ds_params, cnf)
            )

            n_total = len(x_train) + len(x_valid) + len(x_test)
            n_expected = _COMMON_DS["n_samples"]
            deficit = n_total < n_expected

            for learning_seed in LEARNING_SEEDS:
                run_idx += 1
                print(
                    f"  Run {run_idx}/{N_RUNS} (problem_seed={problem_seed}, learning_seed={learning_seed})"
                )

                set_learning_seed(learning_seed)
                config = TRAIN_PARAMS
                predicted, _ = ALL_METHODS["plain_nn"](
                    x_train,
                    y_train,
                    x_valid,
                    y_valid,
                    x_test,
                    cnf,
                    config,
                )
                metrics = evaluate(predicted, y_test, x_test, cnf)

                all_runs[key]["accuracy"].append(metrics.get("accuracy", 0))
                all_runs[key]["constraint_violation_ratio"].append(
                    metrics.get("constraint_violation_ratio", 0)
                )
                all_runs[key]["hamming_distance"].append(
                    metrics.get("hamming_distance", 0)
                )
                all_runs[key]["n_unique_outputs"].append(
                    dataset_info.get("n_unique_outputs", 0)
                )
                all_runs[key]["n_train"].append(len(x_train))
                all_runs[key]["n_total"].append(n_total)
                all_runs[key]["deficit"].append(deficit)

                csv_rows.append(
                    {
                        "generator": gen_name,
                        "hidden_dim": cfg["hidden_dim"],
                        "num_hidden": cfg["num_hidden"],
                        "problem_seed": problem_seed,
                        "learning_seed": learning_seed,
                        "n_train": len(x_train),
                        "n_valid": len(x_valid),
                        "n_test": len(x_test),
                        "n_total": n_total,
                        "n_unique_outputs": dataset_info.get("n_unique_outputs", 0),
                        "deficit": deficit,
                        "accuracy": metrics.get("accuracy", 0),
                        "constraint_violation_ratio": metrics.get(
                            "constraint_violation_ratio", 0
                        ),
                        "hamming_distance": metrics.get("hamming_distance", 0),
                        "binary_cross_entropy": metrics.get("binary_cross_entropy", 0),
                    }
                )

                deficit_str = f" *** DEFICIT {n_total}/{n_expected}" if deficit else ""
                print(
                    f"    acc={metrics.get('accuracy', 0):.4f}  cvr={metrics.get('constraint_violation_ratio', 0):.4f}  unique={dataset_info.get('n_unique_outputs', 0)}  total={n_total}{deficit_str}"
                )

# Save all runs to CSV
csv_path = ARGS.out_csv
csv_path.parent.mkdir(parents=True, exist_ok=True)
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"\nAll runs saved to {csv_path} ({len(csv_rows)} rows)")

# Summary table
summary = {}
for gen_name in GENERATOR_CONFIGS:
    print(f"\n{'='*80}")
    print(
        f"Summary [{gen_name}] ({N_RUNS} runs: {len(PROBLEM_SEEDS)} problem seeds x {len(LEARNING_SEEDS)} learning seeds)"
    )
    print(f"{'='*80}")
    print(
        f"{'Config':<15} {'Unique':>12} {'Train':>12} {'Accuracy':>14} {'CVR':>14} {'Hamming':>14} {'Deficit':>8}"
    )
    print("-" * 95)

    for cfg in CAPACITY_CONFIGS:
        label = f"h{cfg['hidden_dim']}_l{cfg['num_hidden']}"
        key = (gen_name, label)
        runs = all_runs[key]
        s = {}
        for m in [
            "accuracy",
            "constraint_violation_ratio",
            "hamming_distance",
            "n_unique_outputs",
            "n_train",
        ]:
            vals = np.array(runs[m])
            s[m] = {"mean": vals.mean(), "sem": vals.std(ddof=1) / np.sqrt(len(vals))}
        n_deficit = sum(runs["deficit"])
        s["n_deficit"] = n_deficit
        summary[key] = s
        deficit_str = f"{n_deficit}/{N_RUNS}" if n_deficit > 0 else "0"
        print(
            f"{label:<15} "
            f"{s['n_unique_outputs']['mean']:>5.1f}±{s['n_unique_outputs']['sem']:.1f} "
            f"{s['n_train']['mean']:>6.0f}±{s['n_train']['sem']:.0f} "
            f"{s['accuracy']['mean']:>7.4f}±{s['accuracy']['sem']:.4f} "
            f"{s['constraint_violation_ratio']['mean']:>7.4f}±{s['constraint_violation_ratio']['sem']:.4f} "
            f"{s['hamming_distance']['mean']:>7.4f}±{s['hamming_distance']['sem']:.4f} "
            f"{deficit_str:>8}"
        )

# Plot: grouped bars for filtering vs fixing side by side
cap_labels = [f"h{c['hidden_dim']}_l{c['num_hidden']}" for c in CAPACITY_CONFIGS]
gen_names = list(GENERATOR_CONFIGS.keys())
n_caps = len(cap_labels)
n_gens = len(gen_names)
x = np.arange(n_caps)
bar_width = 0.8 / (n_gens * 2)  # 2 metrics per generator
colors_acc = {"filtering": "steelblue", "fixing": "cornflowerblue"}
colors_cvr = {"filtering": "salmon", "fixing": "lightsalmon"}

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

for i, gen_name in enumerate(gen_names):
    acc_means = [summary[(gen_name, l)]["accuracy"]["mean"] for l in cap_labels]
    acc_errs = [summary[(gen_name, l)]["accuracy"]["sem"] for l in cap_labels]
    cvr_means = [
        summary[(gen_name, l)]["constraint_violation_ratio"]["mean"] for l in cap_labels
    ]
    cvr_errs = [
        summary[(gen_name, l)]["constraint_violation_ratio"]["sem"] for l in cap_labels
    ]

    offset_acc = (i * 2) * bar_width - 0.4 + bar_width / 2
    offset_cvr = (i * 2 + 1) * bar_width - 0.4 + bar_width / 2

    ax1.bar(
        x + offset_acc,
        acc_means,
        bar_width,
        yerr=acc_errs,
        capsize=3,
        label=f"Accuracy ({gen_name})",
        color=colors_acc[gen_name],
    )
    ax2.bar(
        x + offset_cvr,
        cvr_means,
        bar_width,
        yerr=cvr_errs,
        capsize=3,
        label=f"CVR ({gen_name})",
        color=colors_cvr[gen_name],
    )

ax1.set_ylabel("Accuracy")
ax1.set_ylim(0, 1.05)
ax2.set_ylabel("Constraint Violation Ratio")
all_cvr = [summary[k]["constraint_violation_ratio"]["mean"] for k in summary]
ax2.set_ylim(0, max(0.1, max(all_cvr) * 1.5))

ax1.set_xlabel("Base Mapping Capacity")
ax1.set_xticks(x)
ax1.set_xticklabels(cap_labels, rotation=45, ha="right")
ax1.set_title(
    f"Filtering vs Fixing ({len(PROBLEM_SEEDS)} problem seeds x {len(LEARNING_SEEDS)} learning seeds)"
)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig(csv_path.with_suffix(".png"), dpi=150)
print(f"\nPlot saved to {csv_path.with_suffix('.png')}")
