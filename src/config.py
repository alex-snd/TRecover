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

# # Create Trainer logger
# train_logger = logging.getLogger('trainer')
# train_logger.setLevel(logging.DEBUG)
# train_console = Console(record=True)
# train_console_handler = RichHandler(console=train_console, show_time=False, show_level=False, show_path=False,
#                                     markup=True, rich_tracebacks=True, tracebacks_show_locals=True)
# train_console_handler.setLevel(logging.DEBUG)
# train_logger.addHandler(hdlr=train_console_handler)

# Create Project logger
project_logger = logging.getLogger('project')  # TODO use it everywhere
project_logger.setLevel(logging.DEBUG)

error_console = Console(file=Path(LOGS_DIR, 'error.log').open(mode='a'))
error_handler = RichHandler(console=error_console, markup=True, rich_tracebacks=True, tracebacks_show_locals=True)
error_handler.setLevel(logging.ERROR)
project_logger.addHandler(hdlr=error_handler)

info_console = Console(file=Path(LOGS_DIR, 'info.log').open(mode='a'))
info_handler = RichHandler(console=info_console, markup=True, rich_tracebacks=True, tracebacks_show_locals=True)
info_handler.setLevel(logging.INFO)
project_logger.addHandler(hdlr=info_handler)
