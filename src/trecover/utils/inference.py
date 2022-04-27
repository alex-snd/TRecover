import re
from pathlib import Path
from typing import List, Union

import numpy as np

from trecover.config import var


# ---------------------------------Plain inference data cleaning & preparation utils------------------------------------


def create_noisy_columns(data: str, min_noise: int, max_noise: int) -> List[str]:
    """
    Generate columns for keyless reading from plain data with defined noise range.

    Parameters
    ----------
    data : str
        Plain text.
    min_noise : int
        Minimum noise range value.
    max_noise : int
        Maximum noise range value.

    Returns
    -------
    columns : List[str]
        Columns for keyless reading.

    """

    np.random.seed(None)
    columns = list()

    data = re.sub(r'[^A-Za-z]', '', data).lower()

    for symbol in data:
        noise_size = np.random.randint(low=min_noise, high=max_noise, size=1)[0]
        noise_indexes = np.random.choice(list(var.ALPHABET.difference(symbol)), size=noise_size, replace=False)
        columns.append(f"{symbol}{''.join(noise_indexes)}")

    return columns


def create_files_noisy_columns(files: List[Union[str, Path]],
                               min_noise: int,
                               max_noise: int,
                               n_to_show: int = 0
                               ) -> List[List[str]]:
    """
    Generate columns for keyless reading from plain data contained in the files with defined noise range.

    Parameters
    ----------
    files : List[Union[str, Path]]
        Paths to files that contain plain data to create noised columns for keyless reading.
    min_noise : int
        Minimum noise range value.
    max_noise : int
        Maximum noise range value.
    n_to_show : int, default=0
        Maximum number of columns. Zero means no restrictions.

    Returns
    -------
    files_columns : List[List[str]]
        Batch of columns for keyless reading.

    """

    files_columns = list()

    for file in files:
        with open(file) as f:
            data = f.read()

        if n_to_show > 0:
            data = data[:n_to_show]

        columns = create_noisy_columns(data, min_noise, max_noise)

        files_columns.append(columns)

    return files_columns


# --------------------------------Noised inference data cleaning & preparation utils-----------------------------------


def data_to_columns(data: str, separator: str = ' ') -> List[str]:
    """
    Clean and split noised data.

    Parameters
    ----------
    data : str
        Noised columns for keyless reading.
    separator : str, default=' '
        Separator to split the data into columns.

    Returns
    -------
    List[str]:
        Columns for keyless reading.

    """

    data = re.sub(separator, ' ', data)
    cleaned_data = re.sub(r'[^A-Za-z ]', '', data).lower()

    return cleaned_data.split(' ')


def read_files_columns(files: List[Union[str, Path]], separator: str, n_to_show: int = 0) -> List[List[str]]:
    """
    Read, clean and split noised data contained in the files.

    Parameters
    ----------
    files : List[Union[str, Path]]
        Paths to files that contain noised data for keyless reading.
    separator : str
        Separator to split the data into columns.
    n_to_show : int, default=0
        Maximum number of columns. Zero means no restrictions.

    Returns
    -------
    files_columns : List[List[str]]
        Batch of columns for keyless reading.

    """

    files_columns = list()

    for file in files:
        with open(file) as f:
            data = f.read()

        columns = data_to_columns(data, separator)

        if n_to_show > 0:
            columns = columns[:n_to_show]

        files_columns.append(columns)

    return files_columns
