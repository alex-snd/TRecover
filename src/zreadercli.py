import math
import os
from pathlib import Path
from time import time
from typing import Optional
from zipfile import ZipFile

import requests
import torch
import typer
from typer import Typer, Argument, Option

import utils

cli = Typer(name='ZreaderAPI')


def download_from_disk(sharing_link: str, save_dir: str) -> Optional[Path]:
    """
        Download file from Yandex disk

        Notes
        -----
        sharing_link: str
            Sharing link to the file on Yandex disk

        save_dir: str
            Path where to store downloaded file


        Returns
        -------
        filepath: Optional[Path]
            Path to the downloaded file. None if failed to download

    """

    if not (direct_link := utils.get_real_direct_link(sharing_link)):
        typer.secho(f'Failed to download "{sharing_link}"', fg=typer.colors.BRIGHT_RED, bold=True)
        return None

    filename = utils.extract_filename(direct_link) or 'downloaded_data'  # Try to recover the filename from the link
    filepath = Path(save_dir, filename)

    with filepath.open(mode='wb') as fw:
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

    return filepath


@cli.command()
def download_data(sharing_link: str = Argument(..., help='Sharing link to the train data on Yandex disk'),
                  save_dir: str = Option('./', help='Path where to store downloaded data')
                  ) -> None:
    """
        Download train data from Yandex disk

        Notes
        -----
        sharing_link: str
            Sharing link to the train data on Yandex disk

        save_dir: str
            Path where to store downloaded data

    """

    if filepath := download_from_disk(sharing_link, save_dir):
        with ZipFile(filepath) as zf:
            zf.extractall(path=Path(save_dir, filepath.stem))

        os.remove(filepath)


@cli.command()
def download_weights(sharing_link: str = Argument(..., help='Sharing link to the model weights on Yandex disk'),
                     save_dir: str = Option('./', help='Path where to save downloaded weights')
                     ) -> None:
    """
        Download model weights from Yandex disk

        Notes
        -----
        sharing_link: str
            Sharing link to the model weights on Yandex disk

        save_dir: str
            Path where to save downloaded weights

    """

    download_from_disk(sharing_link, save_dir)


@cli.command()
def zread(inference_path: str = Argument(..., help='Path to file or dir for inference'),
          model_artifacts: str = Argument(..., help='Path to model artifacts json file'),
          weights_path: str = Argument(..., help='Path to model weights'),
          cuda: bool = Option(True, help='CUDA enabled'),
          gpu_id: int = Option(0, help='GPU id'),
          separator: str = Option(' ', help='Columns separator in the input files'),
          noisy: bool = Option(False, help='Input files are noisy texts'),
          min_noise: int = Option(3, help='Min noise parameter. Minimum value is alphabet size'),
          max_noise: int = Option(5, help='Max noise parameter. Maximum value is alphabet size'),
          beam_width: int = Option(1, help='Width for beam search algorithm. Maximum value is alphabet size'),
          console_width: int = Option(0, help='Console width for visualization. Zero value means for no restrictions'),
          delimiter: str = Option('', help='Delimiter for columns visualization')
          ) -> None:
    inference_path = Path(inference_path)

    if not noisy and min_noise >= max_noise:
        typer.secho('Maximum noise range must be grater than minimum noise range',
                    fg=typer.colors.BRIGHT_RED, bold=True)
        return

    if not any([inference_path.is_file(), inference_path.is_dir()]):
        typer.secho('Files for inference needed to be specified', fg=typer.colors.BRIGHT_RED, bold=True)
        return

    artifacts = utils.load_artifacts(Path(model_artifacts))
    device = torch.device(f'cuda:{gpu_id}' if cuda and torch.cuda.is_available() else 'cpu')
    z_reader = utils.get_model(artifacts['token_size'], artifacts['pe_max_len'], artifacts['num_layers'],
                               artifacts['d_model'], artifacts['n_heads'], artifacts['d_ff'], artifacts['dropout'],
                               device, weights=Path(weights_path))
    z_reader.eval()

    files, files_columns = utils.get_files_columns(inference_path, separator, noisy, min_noise, max_noise,
                                                   console_width, delimiter)
    files_src = utils.files_columns_to_tensors(files_columns, device)

    for file_id, (file, src) in enumerate(zip(files, files_src), start=1):
        start_time = time()

        loop_label = typer.style(f'[{file_id}/{len(files_src)}] Processing {file.name}',
                                 fg=typer.colors.BLUE, bold=True)
        chains = utils.beam_search(src, z_reader, beam_width, device,
                                   beam_loop=utils.cli_interactive_loop(label=loop_label))

        src_scale = src.size(0) * max(2 * len(delimiter), 1) + 1 * len(delimiter)
        printing_scale = console_width if 0 < console_width < src_scale else src_scale

        print('-' * printing_scale)
        print(utils.visualize_columns(src, delimiter=delimiter), end='')
        for tgt, _ in chains:
            print('-' * printing_scale)
            print(utils.visualize_target(tgt, delimiter=delimiter))

        typer.secho(f'Elapsed: {time() - start_time:>7.3f}s\n', fg=typer.colors.BLUE, bold=True)


if __name__ == '__main__':
    cli()
