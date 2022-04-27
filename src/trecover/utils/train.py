import re
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer


class ExperimentParams(dict):
    """ Container for experiment parameters """

    def __init__(self, params: Union[Namespace, Dict, None] = None):
        super(ExperimentParams, self).__init__()

        if isinstance(params, Namespace):
            self.__dict__.update(vars(params))
        if isinstance(params, dict):
            self.__dict__.update(params)

    def __getitem__(self, key: Any) -> Any:
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def jsonify(self) -> Dict:
        """
        Simplify experiment parameters for further json serialization.

        Returns
        -------
            Experiment parameters as a dictionary with python built-in types values.

        """
        simplified = dict()
        simplified.update(self.__dict__)

        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                simplified[key] = str(value)

        return simplified

    def update(self, *args: Any, **kwargs: Any) -> None:
        self.__dict__.update(*args, **kwargs)


def set_seeds(seed: int = 2531) -> None:
    """ Set seeds for experiment reproducibility """

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU


def get_experiment_mark() -> str:
    """ Generate a string mark based on current date """

    date = datetime.now()
    return f'{date.month:0>2}-{date.day:0>2}-{date.hour:0>2}-{date.minute:0>2}'


def optimizer_to_str(optimizer: Optimizer) -> str:
    """
    Get optimizer object string representation.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer object for representation.

    Returns
    -------
    str:
        Optimizer string representation.

    """

    optimizer_str_repr = str(optimizer)

    optimizer_name = optimizer_str_repr.split()[0]
    optimizer_params = re.findall(r'(.+): (.+)', optimizer_str_repr)

    return f"{optimizer_name}({', '.join([f'{param.strip()}={value.strip()}' for param, value in optimizer_params])})"


def transfer(tensors: Tuple[Optional[Tensor], ...], to_device: torch.device) -> Tuple[Optional[Tensor], ...]:
    """
    Transfer the tensors to a specified device.

    Parameters
    ----------
    tensors : Tuple
        Sequence of tensors to transfer.
    to_device : torch.device
        The desired device of returned tensors.

    Returns
    -------
    Tuple[Optional[Tensor], ...]:
        Tuple of tensors allocated on the specified device.

    """
    return tuple([
        tensor.to(to_device) if isinstance(tensor, Tensor) else tensor
        for tensor in tensors
    ])
