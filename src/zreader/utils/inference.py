import re
from pathlib import Path
from typing import List, Union

import numpy as np

from config import var


# ---------------------------------Plain inference data cleaning & preparation utils------------------------------------


def create_noisy_columns(data: str, min_noise: int, max_noise: int) -> List[str]:
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
    data = re.sub(separator, ' ', data)
    cleaned_data = re.sub(r'[^A-Za-z ]', '', data).lower()

    return cleaned_data.split(' ')


def read_files_columns(files: List[Union[str, Path]], separator: str, n_to_show: int = 0) -> List[List[str]]:
    files_columns = list()

    for file in files:
        with open(file) as f:
            data = f.read()

        columns = data_to_columns(data, separator)

        if n_to_show > 0:
            columns = columns[:n_to_show]

        files_columns.append(columns)

    return files_columns
