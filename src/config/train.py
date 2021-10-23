from pathlib import Path

from config.vars import BASE_DIR

DATA_DIR = Path(BASE_DIR, 'data')
TRAIN_DATA = Path(DATA_DIR, 'train')
VAL_DATA = Path(DATA_DIR, 'validation')
VIS_DATA = Path(DATA_DIR, 'visualization')
EXPERIMENTS_DIR = Path(BASE_DIR, 'experiments')
MODEL_REGISTRY_DIR = Path(EXPERIMENTS_DIR, 'mlflow_registry')

DATA_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DATA.mkdir(parents=True, exist_ok=True)
VAL_DATA.mkdir(parents=True, exist_ok=True)
VIS_DATA.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

