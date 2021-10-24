import re
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from torch import Tensor

from config import var


# ----------------------------------------Data cleaning & preparation utils---------------------------------------------


def clean_wiki_text(filename: Path, drop_threshold: float = 0.62) -> None:
    result_size = 0

    with open(filename, mode='r') as wiki_file, open(f'{filename}.clean', mode='w') as writer:

        while True:
            try:
                line = wiki_file.readline()
            except UnicodeDecodeError:
                continue

            if not line:
                break

            if not line.strip() or line.startswith(' = '):
                continue

            cleaned_line = re.sub(r'[^A-Za-z]', '', line)

            if len(cleaned_line) / len(line) < drop_threshold:
                continue

            writer.write(cleaned_line.lower())
            result_size += len(cleaned_line)

    print(f'Result size: {result_size}')


def clean_wiki_qa(filename: Path) -> None:
    result_size = 0
    questions = set()

    with open(filename, mode='r') as wiki_file, open(f'{filename}.clean', mode='w') as writer:

        while True:
            try:
                line = wiki_file.readline()
            except UnicodeDecodeError:
                continue

            if not line:
                break

            try:
                question, answer = line.split('\t')[:2]
            except ValueError:
                continue

            cleaned_question = re.sub(r'[^A-Za-z]', '', question).lower()
            cleaned_answer = re.sub(r'[^A-Za-z]', '', answer).lower()

            if cleaned_question not in questions:
                writer.write(cleaned_question)

                questions.add(cleaned_question)
                result_size += len(cleaned_question)

            writer.write(cleaned_answer)
            result_size += len(cleaned_answer)

    print(f'Result size: {result_size}')


def clean_blogs(blogs_folder: Path) -> None:
    result_size = 0

    with open(blogs_folder / 'blogs.clean', mode='w') as writer:
        for blog_file in [Path(blogs_folder, file) for file in blogs_folder.iterdir()]:

            text_flag = False

            with open(blog_file, mode='r') as f:
                while True:
                    try:
                        line = f.readline()
                    except UnicodeDecodeError:
                        continue

                    if not line:
                        break

                    if text_flag and '</post>' in line:
                        text_flag = False

                    elif text_flag:
                        cleaned_text = re.sub(r'[^A-Za-z]', '', line).lower()

                        writer.write(cleaned_text)
                        result_size += len(cleaned_text)

                    elif not text_flag and '<post>' in line:
                        text_flag = True

    print(f'Result size: {result_size}')


# -------------------------------------------------Collate utils--------------------------------------------------------

def generate_subsequent_mask(size: int) -> Tensor:
    return torch.triu(torch.ones((size, size), dtype=torch.float), diagonal=1) == 1


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


# ---------------------------------------------Columns preparation utils-----------------------------------------------


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
