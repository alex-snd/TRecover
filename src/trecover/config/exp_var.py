import os

from trecover.config import var

TRAIN_DATA = var.DATA_DIR / 'train'
VAL_DATA = var.DATA_DIR / 'validation'
VIS_DATA = var.DATA_DIR / 'visualization'

var.DATA_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DATA.mkdir(parents=True, exist_ok=True)
VAL_DATA.mkdir(parents=True, exist_ok=True)
VIS_DATA.mkdir(parents=True, exist_ok=True)

WANDB_REGISTRY_DIR = var.EXPERIMENTS_DIR / 'wandb_registry'
MLFLOW_REGISTRY_DIR = var.EXPERIMENTS_DIR / 'mlflow_registry'
MLFLOW_BACKEND = os.getenv('MLFLOW_BACKEND',
                           default=f'sqlite:{(MLFLOW_REGISTRY_DIR.absolute() / "mlflow.db").as_uri()[5:]}')

var.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
WANDB_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
