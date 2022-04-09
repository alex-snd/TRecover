import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from trecover.config import var

var.LOGS_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_LOG = var.LOGS_DIR / 'mlflow.log'
DASHBOARD_LOG = var.LOGS_DIR / 'dashboard.log'
API_LOG = var.LOGS_DIR / 'api.log'
WORKER_LOG = var.LOGS_DIR / 'worker.log'

project_logger = logging.getLogger('project')
project_logger.setLevel(logging.DEBUG)

project_console = Console(force_terminal=True, record=True)
console_handler = RichHandler(console=project_console, markup=True, show_time=False, show_level=False, show_path=False)
console_handler.setLevel(logging.DEBUG)
project_logger.addHandler(hdlr=console_handler)

error_console = Console(file=Path(var.LOGS_DIR, 'error.log').open(mode='a'))
error_handler = RichHandler(console=error_console, markup=True, rich_tracebacks=True, tracebacks_show_locals=True)
error_handler.setLevel(logging.ERROR)
project_logger.addHandler(hdlr=error_handler)

info_console = Console(file=Path(var.LOGS_DIR, 'info.log').open(mode='a'))
info_handler = RichHandler(console=info_console, markup=True, rich_tracebacks=True, tracebacks_show_locals=True)
info_handler.setLevel(logging.INFO)
project_logger.addHandler(hdlr=info_handler)
