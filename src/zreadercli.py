import math
from pathlib import Path
from time import time

import requests
import torch
import typer
from typer import Argument, Option

import utils
from data import Collate

app = typer.Typer()


@app.command()
def download_weights(sharing_link: str, save_dir: str = './') -> None:
    """
        Download model weights from Yandex disk

        Notes
        -----
        sharing_link: str
            Sharing link to the model weights on Yandex disk

        save_dir: str
            Path where to save the downloaded weights

    """

    def get_real_direct_link():
        pk_request = requests.get(
            f'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={sharing_link}')

        return pk_request.json().get('href')  # Returns None if the link cannot be 'converted'

    def extract_filename():
        for chunk in direct_link.strip().split('&'):
            if chunk.startswith('filename='):
                return chunk.split('=')[1]

        return None

    if not (direct_link := get_real_direct_link()):
        typer.secho(f'Failed to download "{sharing_link}"', fg=typer.colors.BRIGHT_RED, bold=True)

    filename = extract_filename() or 'downloaded_weights'  # Try to recover the filename from the link

    with Path(save_dir, filename).open(mode='wb') as fw:
        response = requests.get(direct_link, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            fw.write(response.content)
        else:
            stream_iterator = response.iter_content(chunk_size=4096)
            length = int(total_length) // 4096
            label = typer.style('Downloading', fg=typer.colors.BLUE, bold=True)

            with typer.progressbar(stream_iterator, length=length, label=label) as stream:
                for data in stream:
                    fw.write(data)

    typer.secho(f'Downloaded "{filename}" to "{Path(save_dir, filename).absolute()}"', fg=typer.colors.GREEN, bold=True)


@app.command()
def zread(model_artifacts: str = Argument(..., help='Path to model artifacts json file'),
          weights_path: str = Argument(..., help='Path to model weights'),
          inference_path: str = Argument(..., help='Path to file or dir for inference'),
          cuda: bool = Option(True, help='CUDA enabled'),
          gpu_id: int = Option(0, help='GPU id'),
          separator: str = Option(' ', help='Columns separator in the input files'),
          noisy: bool = Option(False, help='Input files are noisy texts'),
          min_noise: int = Option(3, help='Min noise parameter. Minimum value is alphabet size'),
          max_noise: int = Option(5, help='Max noise parameter. Maximum value is alphabet size'),
          beam_width: int = Option(1, help='Width for beam search algorithm. Maximum value is alphabet size'),
          console_width: int = Option(0, help='Console width for visualization. Zero value means for no restrictions'),
          delimiter: str = Option('', help='Delimiter for columns visualization')) -> None:
    artifacts = utils.load_artifacts(Path(model_artifacts))

    inference_path = Path(inference_path)

    if inference_path.is_file():
        files = [inference_path, ]
    elif inference_path.is_dir():
        files = [file for file in inference_path.iterdir()]
    else:
        typer.secho(f'Files for inference needed to be specified', fg=typer.colors.BRIGHT_RED, bold=True)
        return

    device = torch.device(f'cuda:{gpu_id}' if cuda and torch.cuda.is_available() else 'cpu')

    z_reader = utils.get_model(artifacts['token_size'], artifacts['pe_max_len'], artifacts['num_layers'],
                               artifacts['d_model'], artifacts['n_heads'], artifacts['d_ff'], artifacts['dropout'],
                               device, weights=Path(weights_path))
    z_reader.eval()

    if console_width > 0:
        n_columns_to_show = math.ceil(console_width / max(2 * len(delimiter), 1)) - 1 * len(delimiter)
    else:
        n_columns_to_show = 0

    if noisy:
        files_columns = utils.read_files_columns(files, separator, n_columns_to_show)
    else:
        files_columns = utils.create_files_noisy_columns(files, min_noise, max_noise, Collate.num_to_alphabet,
                                                         n_columns_to_show)

    files_src = utils.files_columns_to_tensors(files_columns, Collate.alphabet_to_num, device)

    for file, src in zip(files, files_src):
        start_time = time()

        chains = utils.beam_search(src, z_reader, beam_width, device)

        src_scale = src.size(0) * max(2 * len(delimiter), 1) + 1 * len(delimiter)
        printing_scale = console_width if 0 < console_width < src_scale else src_scale

        print(file.name)
        print('-' * printing_scale)
        print(utils.visualize_columns(src, delimiter=delimiter), end='')
        for tgt, _ in chains:
            print('-' * printing_scale)
            print(utils.visualize_target(torch.argmax(tgt.squeeze(), dim=-1)[1:],  # first token is empty_token
                                         delimiter=delimiter))

        print(f'Elapsed: {time() - start_time:>7.3f}s\n')


if __name__ == '__main__':
    app()
