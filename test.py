import argparse
import math
import re
from os import listdir
from os.path import join, basename
from time import time

import numpy as np
import torch

import config
from data import Collate
from utils import get_model, visualize_columns, visualize_target, beam_search


def create_noise_columns(files: list, min_noise: int, max_noise: int, num_to_alphabet: dict,
                         n_to_show: int = 0) -> list:
    np.random.seed(None)
    files_columns = list()

    for file in files:
        with open(file) as f:
            data = re.sub(r'[^A-Za-z]', '', f.read()).lower()

        columns = list()

        for symbol in data:
            noise_size = np.random.randint(low=min_noise, high=max_noise, size=1)[0]
            noise_indexes = np.random.randint(low=0, high=len(num_to_alphabet), size=noise_size)

            columns.append(symbol + ''.join([num_to_alphabet[s_id] for s_id in noise_indexes]))

        if n_to_show:
            columns = columns[:n_to_show]

        files_columns.append(columns)

    return files_columns


def read_columns(files: list, separator: str, n_to_show: int = 0) -> list:
    files_columns = list()

    for file in files:
        with open(file) as f:
            data = re.sub(separator, ' ', f.read())

        cleaned_data = re.sub(r'[^A-Za-z ]', '', data).lower()
        columns = cleaned_data.split(' ')

        if n_to_show:
            columns = columns[:n_to_show]

        files_columns.append(columns)

    return files_columns


def columns_to_tensors(files_columns: list, alphabet_to_num: dict, device: torch.device = torch.device('cpu')) -> list:
    files_src = list()

    for columns in files_columns:
        src = torch.zeros((len(columns), len(alphabet_to_num)), dtype=torch.float, device=device)

        for col in range(len(columns)):
            for symbol in columns[col]:
                src[col, alphabet_to_num[symbol]] = 1

        files_src.append(src)

    return files_src


def test(params: argparse.Namespace) -> None:
    device = torch.device(f'cuda:{params.gpu_id}' if params.cuda and torch.cuda.is_available() else 'cpu')

    z_reader = get_model(params.token_size, params.pe_max_len, params.num_layers, params.d_model, params.n_heads,
                         params.d_ff, params.dropout, device)

    z_reader.load_parameters(params.weights, device=device)
    z_reader.eval()

    if params.console_width > 0:
        n_columns_to_show = math.ceil(params.console_width / max(2 * len(params.delimiter), 1)) - \
                            1 * len(params.delimiter)
    else:
        n_columns_to_show = 0

    if params.file_path:
        files = [params.file_path, ]
    else:
        files = [join(params.tests_folder, f) for f in listdir(params.tests_folder)]

    if params.noisy:
        files_columns = create_noise_columns(files, params.min_noise, params.max_noise, Collate.num_to_alphabet,
                                             n_columns_to_show)
    else:
        files_columns = read_columns(files, params.separator, n_columns_to_show)

    files_src = columns_to_tensors(files_columns, Collate.alphabet_to_num, device)

    for file, src in zip(files, files_src):
        start_time = time()

        chains = beam_search(src, z_reader, params.beam_weights, device)

        src_scale = src.size(0) * max(2 * len(params.delimiter), 1) + 1 * len(params.delimiter)
        printing_scale = params.console_width if 0 < params.console_width < src_scale else src_scale

        print(basename(file))
        print('-' * printing_scale)
        print(visualize_columns(src, delimiter=params.delimiter), end='')
        for tgt, _ in chains:
            print('-' * printing_scale)
            print(visualize_target(torch.argmax(tgt.squeeze(), dim=-1)[1:],  # first token is empty_token
                                   delimiter=params.delimiter), end='')

        print(f'Elapsed: {time() - start_time:>7.3f}s\n')


def get_params() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--file_path', action='store', dest='file_path', default=None,
                        help='')

    parser.add_argument('-tf', '--tests_folder', action='store', dest='tests_folder', default=config.tests_folder,
                        help='')

    parser.add_argument('-noisy', '--noisy', action='store_true', dest='noisy',
                        help='')

    parser.add_argument('-min_noise', '--min_noise', action='store', type=int, dest='min_noise', default=2,
                        help='')

    parser.add_argument('-max_noise', '--max_noise', action='store', type=int, dest='max_noise', default=6,
                        help='')

    parser.add_argument('-s', '--separator', action='store', dest='separator', default=' ',
                        help='')

    parser.add_argument('-delimiter', '--delimiter', action='store', dest='delimiter', default='',
                        help='')

    parser.add_argument('-cw', '--console_width', action='store', dest='console_width', type=int, default=0,
                        help='')

    # --------------------------------------------MODEL PARAMETERS------------------------------------------------------
    parser.add_argument('-token_size', '--token_size', action='store', dest='token_size', type=int,
                        default=len(Collate.num_to_alphabet), help='')

    parser.add_argument('-pe_max_len', '--pe_max_len', action='store', dest='pe_max_len', type=int, default=1000,
                        help='')

    parser.add_argument('-num_layers', '--num_layers', action='store', dest='num_layers', type=int, default=6,
                        help='')

    parser.add_argument('-d_model', '--d_model', action='store', dest='d_model', type=int, default=512,
                        help='')

    parser.add_argument('-n_heads', '--n_heads', action='store', dest='n_heads', type=int, default=8,
                        help='')

    parser.add_argument('-d_ff', '--d_ff', action='store', dest='d_ff', type=int, default=2048,
                        help='')

    parser.add_argument('-dropout', '--dropout', action='store', dest='dropout', type=float, default=0.1,
                        help='')

    parser.add_argument('-cuda', '--cuda', action='store_true', dest='cuda',
                        help='')

    parser.add_argument('-multi_gpu', '--multi_gpu', action='store_true', dest='multi_gpu',
                        help='')

    parser.add_argument('-gpu_id', '--gpu_id', action='store', dest='gpu_id', type=int, default=0,
                        help='')

    parser.add_argument('-w', '--weights', action='store', dest='weights',
                        default=join(config.weights_path, '0522_2147_336000'), help='')

    parser.add_argument('-bw', '--beam_weights', action='store', dest='beam_weights', type=int, default=5,
                        help='')

    return parser.parse_args()


if __name__ == '__main__':
    test(get_params())
