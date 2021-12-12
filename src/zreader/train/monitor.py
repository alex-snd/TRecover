from pathlib import Path
from typing import Dict, Union, Optional, Any

import mlflow
import wandb
from mlflow.tracking.fluent import ActiveRun, end_run
from wandb.wandb_run import Run

from config import var


class BaseMonitor(object):
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        raise NotImplementedError

    def log_artifact(self, path: str) -> None:
        raise NotImplementedError

    def log_variables(self, variables: Dict[str, Union[float, int]]) -> None:
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def finish(self):
        raise NotImplementedError

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError


class IdentityMonitor(BaseMonitor):
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        pass

    def log_artifact(self, path: str) -> None:
        pass

    def log_variables(self, variables: Dict[str, Union[float, int]]) -> None:
        pass

    def start(self):
        pass

    def finish(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class WandbMonitor(BaseMonitor):
    def __init__(self,
                 project_name: str,
                 experiment_name: str,
                 config: Dict[str, Any],
                 registry: Union[str, Path] = var.WANDB_REGISTRY_DIR.absolute()):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        self.registry = registry

    # TODO alter steps
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        wandb.log(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        wandb.log_artifact(path)

    def log_variables(self, variables: Dict[str, Union[float, int]]) -> None:
        for var_name, var_value in variables.items():
            wandb.summary[var_name] = var_value

    def start(self) -> Run:
        return wandb.init(project=self.project_name,
                          name=self.experiment_name,
                          config=self.config,
                          dir=self.registry)

    def finish(self) -> None:
        wandb.finish()

    def __enter__(self) -> Run:
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return self.finish()


class MlflowMonitor(BaseMonitor):
    def __init__(self,
                 project_name: str,
                 experiment_name: str,
                 config: Dict[str, Any],
                 registry_uri: Union[str, Path] = var.MLFLOW_REGISTRY_DIR.absolute().as_uri()):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        self.registry_uri = registry_uri

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        mlflow.log_artifact(path)

    def log_variables(self, variables: Dict[str, Union[float, int]]) -> None:
        mlflow.log_params(variables)

    def start(self) -> ActiveRun:
        mlflow.set_tracking_uri(self.registry_uri)
        mlflow.set_experiment(self.project_name)
        mlflow.log_params(self.config)

        return mlflow.start_run(run_name=self.experiment_name)

    def finish(self):
        end_run()

    def __enter__(self) -> Run:
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return self.finish()
