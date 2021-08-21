from pathlib import Path

import mlflow

# Repository
AUTHOR = "Alex-Snd"
REPO = "ZReader"

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
INFERENCE_DIR = Path(BASE_DIR, 'inference')  # for inference zreaderapi params
DATA_DIR = Path(BASE_DIR, 'data')
TRAIN_DATA = Path(DATA_DIR, 'train')
VAL_DATA = Path(DATA_DIR, 'validation')
VIS_DATA = Path(DATA_DIR, 'visualization')
EXAMPLES_DIR = Path(BASE_DIR, 'examples')

# Local stores
EXPERIMENTS_DIR = Path(BASE_DIR, 'experiments')
MODEL_REGISTRY_DIR = Path(EXPERIMENTS_DIR, 'mlruns')

# Create dirs
INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DATA.mkdir(parents=True, exist_ok=True)
VAL_DATA.mkdir(parents=True, exist_ok=True)
VIS_DATA.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# MLFlow model registry
mlflow.set_tracking_uri(MODEL_REGISTRY_DIR.absolute().as_uri())
