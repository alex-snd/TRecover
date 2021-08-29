import logging
from pathlib import Path

import mlflow
from rich.logging import RichHandler
from rich.console import Console

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
LOGS_DIR = Path(BASE_DIR, 'logs')

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
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# MLFlow model registry
mlflow.set_tracking_uri(MODEL_REGISTRY_DIR.absolute().as_uri())

# Create console instance
console = Console(record=True)

# Create loggers
train_logger = logging.getLogger('train')
train_logger.setLevel(logging.DEBUG)
project_logger = logging.getLogger('project')  # TODO use it everywhere
project_logger.setLevel(logging.DEBUG)

# Create handlers
console_handler = RichHandler(console=console, show_time=False, show_level=False, show_path=False, markup=True,
                              rich_tracebacks=True, tracebacks_show_locals=True)
console_handler.setLevel(logging.DEBUG)

error_handler = logging.handlers.RotatingFileHandler(
    filename=Path(LOGS_DIR, "error.log"),
    maxBytes=10485760,  # 1 MB
    backupCount=10)
error_handler.setLevel(logging.ERROR)

# Create formatters
minimal_formatter = logging.Formatter(fmt="%(message)s")
detailed_formatter = logging.Formatter(
    fmt="%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n")

# Hook it all up
console_handler.setFormatter(fmt=minimal_formatter)
error_handler.setFormatter(fmt=detailed_formatter)

train_logger.addHandler(hdlr=console_handler)
train_logger.addHandler(hdlr=error_handler)
