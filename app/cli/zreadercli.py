import os
from pathlib import Path
from time import time
from zipfile import ZipFile

import torch
from rich.console import Group
from rich.panel import Panel
from rich.progress import Progress, TextColumn
from rich.text import Text
from typer import Typer, Argument, Option

import config
from src.utils.beam_search import beam_search, cli_interactive_loop
from src.utils.cli import download_from_disk, get_files_columns
from src.utils.data import files_columns_to_tensors
from src.utils.model import get_model, load_artifacts
from src.utils.visualization import visualize_columns, visualize_target

cli = Typer(name='Zreader-cli')


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
          n_to_show: int = Option(0, help='Number of columns to visualize. Zero value means for no restrictions'),
          delimiter: str = Option('', help='Delimiter for columns visualization')
          ) -> None:
    inference_path = Path(inference_path)
    artifacts = load_artifacts(Path(model_artifacts))

    if not noisy and min_noise >= max_noise:
        config.project_logger.error('[red]Maximum noise range must be grater than minimum noise range')
        return

    if not any([inference_path.is_file(), inference_path.is_dir()]):
        config.project_logger.error('[red]Files for inference needed to be specified')
        return

    if artifacts['pe_max_len'] < n_to_show:
        config.project_logger.error(f'[red]Parameter n_to_show={n_to_show} must be less than {artifacts["pe_max_len"]}')
        return
    elif n_to_show == 0:
        n_to_show = artifacts['pe_max_len']

    device = torch.device(f'cuda:{gpu_id}' if cuda and torch.cuda.is_available() else 'cpu')

    with Progress(TextColumn('{task.description}', style='bright_blue'), transient=True) as progress:
        progress.add_task('Model loading...')
        z_reader = get_model(artifacts['token_size'], artifacts['pe_max_len'], artifacts['num_layers'],
                             artifacts['d_model'], artifacts['n_heads'], artifacts['d_ff'], artifacts['dropout'],
                             device, weights=Path(weights_path))
    z_reader.eval()

    config.project_console.print()

    files, files_columns = get_files_columns(inference_path, separator, noisy, min_noise, max_noise, n_to_show)
    files_src = files_columns_to_tensors(files_columns, device)

    for file_id, (file, src) in enumerate(zip(files, files_src), start=1):
        start_time = time()

        loop_label = f'{file_id}/{len(files_src)} Processing {file.name}'
        chains = beam_search(src, z_reader, beam_width, device,
                             beam_loop=cli_interactive_loop(label=loop_label))
        chains = [Text(visualize_target(chain, delimiter=delimiter), style='cyan', justify='center',
                       overflow='ellipsis', end='\n\n') for (chain, _) in chains]

        columns = visualize_columns(src, delimiter=delimiter, as_rows=True)
        columns = (Text(row, style='bright_blue', overflow='ellipsis', no_wrap=True) for row in columns)

        panel_group = Group(
            Text('Columns', style='magenta', justify='center'),
            *columns,
            Text('Predicted', style='magenta', justify='center'),
            *chains
        )

        config.project_console.print(
            Panel(panel_group, title=file.name, border_style='magenta'),
            justify='center'
        )

        config.project_console.print(f'\nElapsed: {time() - start_time:>7.3f} s\n', style='bright_blue')


if __name__ == '__main__':
    try:
        cli()
    except Exception as e:
        config.project_logger.error(e)
        config.project_console.print_exception(show_locals=True)
        config.error_console.print_exception(show_locals=True)
