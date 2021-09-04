import asyncio
import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple, Callable, Awaitable

import numpy as np
import celery
import requests
import torch
import torch.nn.functional as F
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from torch import Tensor
from torch.optim import Optimizer

import config
from data import Collate
from model import ZReader


# -------------------------------------Train data cleaning & preparation utils------------------------------------------


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

    with open(f'{Path(config.DATA_DIR, "raw_data", "blogs")}.clean', mode='w') as writer:
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


# --------------------------------------------------Trainer utils-------------------------------------------------------


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


def get_model(token_size: int,
              pe_max_len: int,
              num_layers: int,
              d_model: int,
              n_heads: int,
              d_ff: int,
              dropout: float,
              device: torch.device = torch.device('cpu'),
              weights: Optional[Path] = None
              ) -> ZReader:
    model = ZReader(token_size, pe_max_len, num_layers, d_model, n_heads, d_ff, dropout).to(device)

    if weights and weights.exists() and weights.is_file():
        model.load_parameters(weights, device=device)

    return model


def load_artifacts(model_artifacts: Path) -> Dict[str, Union[str, int, float]]:
    # TODO load mlflow artifacts
    return json.load(model_artifacts.open())


def save_parameters(data: Dict, filepath: Path, sort=False) -> None:
    with open(filepath, 'w') as f:
        json.dumps(data, indent=2, fp=f, sort_keys=sort)


# ------------------------------------------------Visualization utils---------------------------------------------------

def visualize_columns(grid: Union[Tensor, List[str]], delimiter: str = '', as_rows=False) -> Union[str, List[str]]:
    columns = list()
    max_depth = 0
    rows = list()

    if isinstance(grid, Tensor):
        for c in range(grid.size(0)):
            columns.append([Collate.num_to_alphabet[pos] for pos in range(grid.size(1)) if grid[c, pos]])
            max_depth = len(columns[c]) if len(columns[c]) > max_depth else max_depth
    else:
        columns = [list(column) for column in grid]
        max_depth = max([len(column) for column in grid])

    for d in range(max_depth):

        row = str()
        for c in range(len(columns)):
            row += f'{delimiter}{columns[c][d]}' if d < len(columns[c]) else f'{delimiter} '

        rows.append(f'{row}{delimiter}')

    return rows if as_rows else '\n'.join(rows)


def visualize_target(tgt: Tensor, delimiter: str = '') -> str:
    tgt = [Collate.num_to_alphabet[ch_id] for ch_id in tgt.tolist()]

    return f'{delimiter}{delimiter.join(tgt)}{delimiter}'


# ---------------------------------Plain inference data cleaning & preparation utils------------------------------------


def create_noisy_columns(data: str, min_noise: int, max_noise: int) -> List[str]:
    np.random.seed(None)
    columns = list()

    data = re.sub(r'[^A-Za-z]', '', data).lower()

    for symbol in data:
        noise_size = np.random.randint(low=min_noise, high=max_noise, size=1)[0]
        noise_indexes = np.random.randint(low=0, high=len(Collate.num_to_alphabet), size=noise_size)

        columns.append(f"{symbol}{''.join([Collate.num_to_alphabet[s_id] for s_id in noise_indexes])}")

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
    tensor = torch.zeros((len(columns), len(Collate.alphabet_to_num)), dtype=torch.float, device=device)

    for col in range(len(columns)):
        for symbol in columns[col]:
            tensor[col, Collate.alphabet_to_num[symbol]] = 1

    return tensor


def files_columns_to_tensors(files_columns: List[List[str]],
                             device: torch.device = torch.device('cpu')
                             ) -> List[Tensor]:
    return [columns_to_tensor(columns, device) for columns in files_columns]


# ----------------------------------------------Synchronous Beam Search-------------------------------------------------


def beam_step(candidates: List[Tuple[Tensor, float]],
              encoded_src: Tensor,
              z_reader: ZReader,
              width: int, device: torch.device
              ) -> List[Tuple[Tensor, float]]:
    step_candidates = list()

    for tgt_inp, score in candidates:
        prediction = z_reader.predict(tgt_inp, encoded_src, tgt_attn_mask=None, tgt_pad_mask=None, src_pad_mask=None)
        probabilities = F.log_softmax(prediction[:, -1], dim=-1).squeeze()  # first is batch

        values, indices = probabilities.topk(k=width, dim=-1)
        for prob, pos in zip(values, indices):
            new_token = torch.zeros(1, 1, z_reader.token_size, device=device)
            new_token[0, 0, pos] = 1

            step_candidates.append((torch.cat([tgt_inp, new_token], dim=1), score + float(prob)))

    return sorted(step_candidates, key=lambda candidate: -candidate[1])[:width]


def celery_task_loop(task: celery.Task
                     ) -> Callable[[Tensor, ZReader, int, torch.device], List[Tuple[Tensor, float]]]:
    def inner_loop(encoded_src: Tensor,
                   z_reader: ZReader,
                   width: int,
                   device: torch.device
                   ) -> List[Tuple[Tensor, float]]:
        candidates = [(torch.zeros(1, 1, z_reader.token_size, device=device), 0)]

        for progress in range(encoded_src.size(0)):
            candidates = beam_step(candidates, encoded_src, z_reader, width, device)
            task.update_state(meta={'progress': progress + 1})

        return candidates

    return inner_loop


def cli_interactive_loop(label: str = 'Processing'
                         ) -> Callable[[Tensor, ZReader, int, torch.device], List[Tuple[Tensor, float]]]:
    def inner_loop(encoded_src: Tensor,
                   z_reader: ZReader,
                   width: int,
                   device: torch.device
                   ) -> List[Tuple[Tensor, float]]:
        candidates = [(torch.zeros(1, 1, z_reader.token_size, device=device), 0)]

        with Progress(
                TextColumn('{task.description}', style='bright_blue'),
                BarColumn(complete_style='bright_blue'),
                TextColumn('{task.percentage:>3.0f}%', style='bright_blue'),
                TextColumn('Remaining', style='bright_blue'),
                TimeRemainingColumn(),
                TextColumn('Elapsed', style='bright_blue'),
                TimeElapsedColumn(),
                transient=True,
        ) as progress:
            beam_progress = progress.add_task(label, total=encoded_src.size(0))

            for _ in range(encoded_src.size(0)):
                candidates = beam_step(candidates, encoded_src, z_reader, width, device)
                progress.update(beam_progress, advance=1)

        return candidates

    return inner_loop


def standard_loop(encoded_src: Tensor,
                  z_reader: ZReader,
                  width: int,
                  device: torch.device
                  ) -> List[Tuple[Tensor, float]]:
    candidates = [(torch.zeros(1, 1, z_reader.token_size, device=device), 0)]

    for _ in range(encoded_src.size(0)):
        candidates = beam_step(candidates, encoded_src, z_reader, width, device)

    return candidates


def beam_search(src: Tensor,
                z_reader: ZReader,
                width: int,
                device: torch.device,
                beam_loop: Callable[[Tensor, ZReader, int, torch.device], List[Tuple[Tensor, float]]] = standard_loop
                ) -> List[Tuple[Tensor, float]]:
    src = src.unsqueeze(dim=0)

    encoded_src = z_reader.encode(src, src_pad_mask=None)

    candidates = beam_loop(encoded_src, z_reader, width, device)

    return [(torch.argmax(tgt.squeeze(), dim=-1)[1:], prob) for tgt, prob in candidates]  # first token is empty_token


# ---------------------------------------------Asynchronous Beam Search-------------------------------------------------


async def async_beam_step(candidates: List[Tuple[Tensor, float]],
                          encoded_src: Tensor,
                          z_reader: ZReader,
                          width: int, device: torch.device
                          ) -> List[Tuple[Tensor, float]]:
    async def candidate_step(tgt_inp: Tensor, score: float) -> None:
        prediction = z_reader.predict(tgt_inp, encoded_src, tgt_attn_mask=None, tgt_pad_mask=None, src_pad_mask=None)
        probabilities = F.log_softmax(prediction[:, -1], dim=-1).squeeze()  # first is batch

        values, indices = probabilities.topk(k=width, dim=-1)
        for prob, pos in zip(values, indices):
            new_token = torch.zeros(1, 1, z_reader.token_size, device=device)
            new_token[0, 0, pos] = 1

            step_candidates.append((torch.cat([tgt_inp, new_token], dim=1), score + float(prob)))

    step_candidates = list()

    for candidate_tgt_inp, candidate_score in candidates:
        await candidate_step(candidate_tgt_inp, candidate_score)

    return sorted(step_candidates, key=lambda candidate: -candidate[1])[:width]


def api_interactive_loop(queue: asyncio.Queue,
                         delimiter: str = ''
                         ) -> Callable[[Tensor, ZReader, int, torch.device], Awaitable]:
    async def async_inner_loop(encoded_src: Tensor,
                               z_reader: ZReader,
                               width: int,
                               device: torch.device
                               ) -> None:
        candidates = [(torch.zeros(1, 1, z_reader.token_size, device=device), 0)]

        for _ in range(encoded_src.size(0)):
            candidates = await async_beam_step(candidates, encoded_src, z_reader, width, device)

            chains = [(torch.argmax(tgt.squeeze(), dim=-1)[1:], prob) for tgt, prob in candidates]
            chains = [(visualize_target(tgt, delimiter), prob) for tgt, prob in chains]

            await queue.put(chains)

        await queue.put(None)

    return async_inner_loop


async def standard_async_loop(encoded_src: Tensor,
                              z_reader: ZReader,
                              width: int,
                              device: torch.device
                              ) -> List[Tuple[Tensor, float]]:
    candidates = [(torch.zeros(1, 1, z_reader.token_size, device=device), 0)]

    for _ in range(encoded_src.size(0)):
        candidates = await async_beam_step(candidates, encoded_src, z_reader, width, device)

    return candidates


async def async_beam_search(src: Tensor,
                            z_reader: ZReader,
                            width: int,
                            device: torch.device,
                            beam_loop: Callable[[Tensor, ZReader, int, torch.device],
                                                Awaitable[Optional[List[Tuple[Tensor, float]]]]] = standard_async_loop
                            ) -> Optional[List[Tuple[Tensor, float]]]:
    src = src.unsqueeze(dim=0)

    encoded_src = z_reader.encode(src, src_pad_mask=None)

    candidates = await beam_loop(encoded_src, z_reader, width, device)

    return [(torch.argmax(tgt.squeeze(), dim=-1)[1:], prob) for tgt, prob in candidates] if candidates else None


# ---------------------------------------------------CLI utils----------------------------------------------------------


def get_real_direct_link(sharing_link: str) -> str:
    pk_request = requests.get(
        f'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={sharing_link}')

    return pk_request.json().get('href')  # Returns None if the link cannot be 'converted'


def extract_filename(direct_link: str) -> Optional[str]:
    for chunk in direct_link.strip().split('&'):
        if chunk.startswith('filename='):
            return chunk.split('=')[1]

    return None


def get_files_columns(inference_path: Path,
                      separator: str,
                      noisy: bool,
                      min_noise: int,
                      max_noise: int,
                      n_to_show: int,
                      ) -> Tuple[List[Path], List[List[str]]]:
    if inference_path.is_file():
        files = [inference_path, ]
    else:
        files = [file for file in inference_path.iterdir()]

    if noisy:
        files_columns = read_files_columns(files, separator, n_to_show)
    else:
        files_columns = create_files_noisy_columns(files, min_noise, max_noise, n_to_show)

    return files, files_columns


if __name__ == '__main__':
    # clean_wiki_text('path/to/wikitext_raw_file')
    # clean_wiki_qa(join(config.data_path, 'WikiQA-train.txt'))
    # clean_blogs(join(config.data_path, 'blogs'))

    pass
