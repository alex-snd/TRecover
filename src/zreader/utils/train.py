import re
from argparse import Namespace
from typing import Any, Optional

import numpy as np
import torch
from torch.optim import Optimizer


class ExperimentParams(dict):
    def __init__(self, params: Optional[Namespace] = None):
        super(ExperimentParams, self).__init__()

        if params:
            self.__dict__.update(vars(params))

    def __getitem__(self, key: Any) -> Any:
        return self.__dict__[key]

    def update(self, *args: Any, **kwargs: Any) -> None:
        self.__dict__.update(*args, **kwargs)


def set_seeds(seed: int = 2531) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU


def optimizer_to_str(optimizer: Optimizer) -> str:
    optimizer_str_repr = str(optimizer)

    optimizer_name = optimizer_str_repr.split()[0]
    optimizer_params = re.findall(r'(.+): (.+)', optimizer_str_repr)

    return f"{optimizer_name}({', '.join([f'{param.strip()}={value.strip()}' for param, value in optimizer_params])})"
