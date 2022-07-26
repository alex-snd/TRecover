import uuid
from pathlib import Path
from typing import Dict, Union, Optional, Any

import mlflow
import wandb
from mlflow.tracking.fluent import ActiveRun, end_run
from wandb.wandb_run import Run

from trecover.config import exp_var
from trecover.utils.train import get_experiment_mark


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

    def __enter__(self) -> Union[Run, ActiveRun]:
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return self.finish()


class IdentityMonitor(BaseMonitor):
    def __init__(self,
                 project_name: str = None,
                 experiment_name: str = None,
                 config: Dict[str, Any] = None):
        self.project_name = project_name or f'DefaultProject-{uuid.uuid4().hex}'
        self.experiment_name = experiment_name or get_experiment_mark()
        self.config = config or dict()

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
                 registry: Union[str, Path] = exp_var.WANDB_REGISTRY_DIR.absolute()):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        self.registry = registry

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        wandb.log(metrics)

    def log_artifact(self, path: str) -> None:
        wandb.log_artifact(artifact_or_path=path, type='experiment_artifact')

    def log_variables(self, variables: Dict[str, Union[float, int]]) -> None:
        for var_name, var_value in variables.items():
            wandb.summary[var_name] = var_value

    def start(self) -> Run:
        return wandb.init(project=self.project_name,
                          name=self.experiment_name,
                          config=self.config,
                          anonymous='allow',
                          dir=self.registry)

    def finish(self) -> None:
        wandb.finish()


class MlflowMonitor(BaseMonitor):
    def __init__(self,
                 project_name: str,
                 experiment_name: str,
                 config: Dict[str, Any],
                 tracking_uri: Union[str, Path] = exp_var.MLFLOW_REGISTRY_DIR.absolute().as_uri()):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        self.tracking_uri = tracking_uri

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        mlflow.log_artifact(path)

    def log_variables(self, variables: Dict[str, Union[float, int]]) -> None:
        mlflow.log_params(variables)

    def start(self) -> ActiveRun:
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.project_name)
        mlflow.log_params(self.config)

        with mlflow.start_run(run_name=self.experiment_name, nested=True) as run:
            return run

    def finish(self):
        end_run()
