import json
from typing import Dict, Optional

import mlflow


class MLflowLogger:
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """
        Parameters
        ----------
        experiment_name : str
            Name of the main experiment.
        tracking_uri : str, optional
            MLflow tracking server URI.
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.active_run = None

    def start_run(self, sub_experiment: str, params: Dict):
        """
        Start a new run inside the experiment.

        Parameters
        ----------
        sub_experiment : str
            Name of the specific run (sub-experiment).
        params : dict
            Dictionary of parameters to log.
        """
        if mlflow.active_run() is not None:
            mlflow.end_run()
        self.active_run = mlflow.start_run(run_name=sub_experiment)

        if params:
            mlflow.log_params(self._flatten_dict(params))

    @staticmethod
    def _flatten_dict(d: Dict, prefix: str = "", sep: str = ".") -> Dict:
        flat = {}
        for k, v in d.items():
            key = f"{prefix}{sep}{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(MLflowLogger._flatten_dict(v, prefix=key, sep=sep))
            elif isinstance(v, (list, tuple)):
                flat[key] = json.dumps(v, default=str)
            else:
                flat[key] = v
        return flat

    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """
        Log metrics.

        Parameters
        ----------
        metrics : dict
            Dictionary of metric_name -> value
        step : int, optional
            Step index
        """
        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=step)

    def log_tags(self, tags: Dict):
        """Log tags."""
        mlflow.set_tags(tags)

    def log_artifact(self, path: str):
        """Log a file artifact."""
        mlflow.log_artifact(path)

    def end_run(self):
        """Finish the active run."""
        if self.active_run:
            mlflow.end_run()
            self.active_run = None
