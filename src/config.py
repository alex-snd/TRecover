import logging
import os
from pathlib import Path

import mlflow
from celery.signals import after_setup_task_logger, after_setup_logger
from rich.console import Console
from rich.logging import RichHandler

# Repository
AUTHOR = "Alex-Snd"
REPO = "ZReader"

# Environment variables
CELERY_BROKER = os.getenv('CELERY_BROKER') or 'pyamqp://guest@localhost'
CELERY_BACKEND = os.getenv('CELERY_BACKEND') or 'redis://localhost:6379'
FASTAPI_HOST = os.getenv('FASTAPI_HOST') or 'localhost'
FASTAPI_PORT = os.getenv('FASTAPI_PORT') or 8001
CUDA = False if os.getenv('CUDA') == 'False' else True
MAX_NOISE = 13

FASTAPI_URL = f'http://{FASTAPI_HOST}:{FASTAPI_PORT}'

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

# Create Project logger
project_logger = logging.getLogger('project')
project_logger.setLevel(logging.DEBUG)

project_console = Console(force_terminal=True, record=True)
console_handler = RichHandler(console=project_console, markup=True, show_time=False, show_level=False, show_path=False)
console_handler.setLevel(logging.DEBUG)
project_logger.addHandler(hdlr=console_handler)

error_console = Console(file=Path(LOGS_DIR, 'error.log').open(mode='a'))
error_handler = RichHandler(console=error_console, markup=True, rich_tracebacks=True, tracebacks_show_locals=True)
error_handler.setLevel(logging.ERROR)
project_logger.addHandler(hdlr=error_handler)

info_console = Console(file=Path(LOGS_DIR, 'info.log').open(mode='a'))
info_handler = RichHandler(console=info_console, markup=True, rich_tracebacks=True, tracebacks_show_locals=True)
info_handler.setLevel(logging.INFO)
project_logger.addHandler(hdlr=info_handler)


# Configure Celery logger
@after_setup_task_logger.connect
def setup_task_logger(logger, *args, **kwargs):
    logger.addHandler(hdlr=error_handler)
    logger.addHandler(hdlr=info_handler)


@after_setup_logger.connect
def setup_task_logger(logger, *args, **kwargs):
    logger.addHandler(hdlr=error_handler)
    logger.addHandler(hdlr=info_handler)
