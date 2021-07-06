from pathlib import Path
import mlflow

# Repository
AUTHOR = "Alex-Snd"
REPO = "Zreader"

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")  # for inference model params
DATA_DIR = Path(BASE_DIR, "data")
TRAIN_DATA = Path(DATA_DIR, "train")
VAL_DATA = Path(DATA_DIR, "validation")
VIS_DATA = Path(DATA_DIR, "visualization")
MODEL_DIR = Path(BASE_DIR, "model")  # for inference weights
# TESTS_DIR = Path(BASE_DIR, "tests")
EXAMPLES_DIR = Path(BASE_DIR, "examples")

# Local stores
EXPERIMENTS_DIR = Path(BASE_DIR, "experiments")
MODEL_REGISTRY_DIR = EXPERIMENTS_DIR / 'mlruns'

# Create dirs
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DATA.mkdir(parents=True, exist_ok=True)
VAL_DATA.mkdir(parents=True, exist_ok=True)
VIS_DATA.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
# TESTS_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# MLFlow model registry
mlflow.set_tracking_uri(MODEL_REGISTRY_DIR.absolute().as_uri())
