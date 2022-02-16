import os

from zreader.config import var

TRAIN_DATA = var.DATA_DIR / 'train'
VAL_DATA = var.DATA_DIR / 'validation'
VIS_DATA = var.DATA_DIR / 'visualization'

var.DATA_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DATA.mkdir(parents=True, exist_ok=True)
VAL_DATA.mkdir(parents=True, exist_ok=True)
VIS_DATA.mkdir(parents=True, exist_ok=True)

MLFLOW_REGISTRY_DIR = var.EXPERIMENTS_DIR / 'mlflow_registry'
WANDB_REGISTRY_DIR = var.EXPERIMENTS_DIR / 'wandb_registry'

var.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
WANDB_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_HOST = os.getenv('MLFLOW_HOST', default='localhost')
MLFLOW_PORT = int(os.getenv('MLFLOW_PORT', default=8002))
MLFLOW_BACKEND = os.getenv('MLFLOW_BACKEND',
                           default=f'sqlite:{(MLFLOW_REGISTRY_DIR.absolute() / "mlflow.db").as_uri()[5:]}')
MLFLOW_WORKERS = int(os.getenv('MLFLOW_WORKERS', default=1))
MLFLOW_PID = var.CONFIG_DIR / 'mlflow.pid'
