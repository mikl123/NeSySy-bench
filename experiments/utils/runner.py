"""Shared parallel experiment runner with MLflow logging."""

from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable

from experiments.utils.mlflow_logger import MLflowLogger


def run_sweep(
    experiment_name: str,
    tasks: list[tuple],
    run_fn: Callable[..., dict | list[dict]],
    tags: dict[str, str] | None = None,
    max_workers: int = 3,
) -> None:
    """Execute *run_fn* for each task in parallel, logging results to MLflow.

    Parameters
    ----------
    experiment_name:
        MLflow experiment name.
    tasks:
        List of argument tuples passed to *run_fn* via ``run_fn(*task)``.
    run_fn:
        Worker function.  Must return either:
        - A single dict with ``{"metrics": {...}, "params": {...}}``
        - A list of such dicts (for sweeps that produce multiple results per task)
        Each dict may optionally include ``"tags": {...}`` and ``"result_dir": str``.
    tags:
        Extra tags added to every MLflow run.
    max_workers:
        Maximum parallel processes.
    """
    logger = MLflowLogger(experiment_name=experiment_name)
    max_workers = min(len(tasks), max_workers)
    failures: list[tuple[tuple, str]] = []

    with ProcessPoolExecutor(
        max_workers=max_workers, max_tasks_per_child=1
    ) as executor:
        futures = {executor.submit(run_fn, *t): t for t in tasks}

        for future in as_completed(futures):
            task_args = futures[future]
            try:
                result = future.result()
                results = result if isinstance(result, list) else [result]

                for r in results:
                    sub = r.get("sub_experiment", _default_sub_experiment(r["params"]))
                    logger.start_run(
                        sub_experiment=sub,
                        params=r["params"],
                    )
                    logger.log_metrics(r["metrics"])
                    run_tags = {**(tags or {}), **r.get("tags", {})}
                    if run_tags:
                        logger.log_tags(run_tags)
                    logger.end_run()

                    _print_result(sub, r["metrics"])

            except Exception as e:
                tb = traceback.format_exc()
                failures.append((task_args, str(e)))
                print(f"FAILED {task_args}: {e}\n{tb}")
                logger.start_run(
                    sub_experiment=f"FAILED_{task_args}",
                    params={"task_args": str(task_args)},
                )
                logger.log_tags(
                    {"status": "failed", "error": str(e)[:250], "traceback": tb[:500]}
                )
                logger.end_run()

    if failures:
        print(f"\n{'='*60}")
        print(f"SUMMARY — {len(failures)}/{len(tasks)} tasks FAILED:")
        print(f"{'='*60}")
        for args, err in failures:
            print(f"  {args}  error={err}")
        print(f"{'='*60}")
    else:
        print(f"\nAll {len(tasks)} tasks completed successfully.")


def _default_sub_experiment(params: dict) -> str:
    """Build a sub-experiment name from params."""
    parts = [
        f"{k}={v}"
        for k, v in params.items()
        if isinstance(v, (int, float, str, bool)) and k != "result_dir"
    ]
    return "_".join(parts[:6])


def _print_result(sub: str, metrics: dict[str, Any]) -> None:
    metrics_str = ", ".join(
        f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))
    )
    print(f"[{sub}] {metrics_str}")
