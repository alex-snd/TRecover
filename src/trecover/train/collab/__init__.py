import logging
import warnings

from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from rich.logging import RichHandler

from .arguments import (
    get_monitor_parser, get_visualization_parser, get_tune_parser,
    get_train_parser, get_auxiliary_parser, sync_base_args
)
from .entrypoint import monitor, visualize, tune, train, auxiliary
from ...config import log

warnings.filterwarnings(action='ignore', message=r'The given NumPy array is not writable', category=UserWarning)

use_hivemind_log_handler('nowhere')
hivemind_logger = get_logger('hivemind')
root_logger = get_logger()

hivemind_handler = RichHandler(console=log.project_console, markup=True, show_path=False,
                               log_time_format='%b %d %H:%M:%S.%f')
hivemind_logger.addHandler(hdlr=hivemind_handler)
hivemind_logger.propagate = False
hivemind_logger.setLevel(logging.INFO)

root_logger.addHandler(hdlr=hivemind_handler)
root_logger.setLevel(logging.INFO)
