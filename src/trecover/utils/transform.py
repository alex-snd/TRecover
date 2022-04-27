from typing import List

import torch
from torch import Tensor

from trecover.config import var


def columns_to_tensor(columns: List[str], device: torch.device = torch.device('cpu')) -> Tensor:
    """
    Convert the columns to a torch tensor.

    Parameters
    ----------
    columns : List[str]
        Columns for keyless reading.
    device : torch.device, default=torch.device('cpu')
        The desired device of returned tensor.

    Returns
    -------
    tensor : Tensor[SEQUENCE_LEN, len(var.ALPHABET)]
        Columns as a torch tensor.

    """

    tensor = torch.zeros((len(columns), len(var.ALPHABET)), dtype=torch.float, device=device)

    for col in range(len(columns)):
        for symbol in columns[col]:
            tensor[col, var.ALPHABET2NUM[symbol]] = 1

    return tensor


def files_columns_to_tensors(files_columns: List[List[str]],
                             device: torch.device = torch.device('cpu')
                             ) -> List[Tensor]:
    """
    Convert the batch of columns to torch tensors.

    Parameters
    ----------
    files_columns : List[List[str]]
        Batch of columns for keyless reading.
    device : torch.device, default=torch.device('cpu')
        The desired device of returned tensor.

    Returns
    -------
    List[Tensor]:
        Columns batch as a list of torch tensors.

    """

    return [columns_to_tensor(columns, device) for columns in files_columns]


def tensor_to_columns(grid: Tensor) -> List[str]:
    """
    Convert the columns' tensor representation to a list of strings with alphabet symbols.

    Parameters
    ----------
    grid : Tensor[SEQUENCE_LEN, len(var.ALPHABET)]
        Columns for keyless reading as a tensor.

    Returns
    -------
    List[str]:
        Columns' tensor representation as a list of strings with alphabet symbols.

    """

    return [
        ''.join([var.NUM2ALPHABET[pos] for pos in range(grid.shape[1]) if grid[c, pos]])
        for c in range(grid.shape[0])
    ]


def tensor_to_target(tgt: Tensor) -> List[str]:
    """
    Convert the target tensor representation to list of alphabet symbols.

    Parameters
    ----------
    tgt : Tensor[SEQUENCE_LEN]
        Target tensor representation of columns' correct symbols.

    Returns
    -------
    List[str]:
        Target tensor representation to list of alphabet symbols.

    """

    return [var.NUM2ALPHABET[ch_id] for ch_id in tgt.tolist()]
