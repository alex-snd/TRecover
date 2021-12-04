from typing import List

import torch
from torch import Tensor

from config import var


def columns_to_tensor(columns: List[str], device: torch.device = torch.device('cpu')) -> Tensor:
    tensor = torch.zeros((len(columns), len(var.ALPHABET)), dtype=torch.float, device=device)

    for col in range(len(columns)):
        for symbol in columns[col]:
            tensor[col, var.ALPHABET2NUM[symbol]] = 1

    return tensor


def files_columns_to_tensors(files_columns: List[List[str]],
                             device: torch.device = torch.device('cpu')
                             ) -> List[Tensor]:
    return [columns_to_tensor(columns, device) for columns in files_columns]


def tensor_to_columns(grid: Tensor) -> List[str]:
    return [
        ''.join([var.NUM2ALPHABET[pos] for pos in range(grid.shape[1]) if grid[c, pos]])
        for c in range(grid.shape[0])
    ]


def tensor_to_target(tgt: Tensor) -> List[str]:
    return [var.NUM2ALPHABET[ch_id] for ch_id in tgt.tolist()]
