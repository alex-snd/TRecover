from pathlib import Path
from time import time

import torch
from rich.console import Group
from rich.panel import Panel
from rich.progress import Progress, TextColumn
from rich.text import Text
from typer import Typer, Argument, Option

from config import var, log
from zreader.utils.beam_search import beam_search, cli_interactive_loop
from zreader.utils.cli import download_archive_from_disk, get_files_columns
from zreader.utils.data import files_columns_to_tensors
from zreader.utils.model import get_model, load_params
from zreader.utils.visualization import visualize_columns, visualize_target

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

    download_archive_from_disk(sharing_link, save_dir)


@cli.command()
def download_artifacts(sharing_link: str = Argument(..., help='Sharing link to the model artifacts on Yandex disk'),
                       save_dir: str = Option('./', help='Path where to save downloaded artifacts')
                       ) -> None:
    """
        Download model artifacts from Yandex disk

        Notes
        -----
        sharing_link: str
            Sharing link to the model artifacts on Yandex disk

        save_dir: str
            Path where to save downloaded artifacts

    """

    download_archive_from_disk(sharing_link, save_dir)


@cli.command()
def zread(inference_path: str = Argument(..., help='Path to file or dir for inference'),
          model_params: str = Option(var.INFERENCE_PARAMS_PATH, help='Path to model params json file'),
          weights_path: str = Option(var.INFERENCE_WEIGHTS_PATH, help='Path to model weights'),
          cuda: bool = Option(var.CUDA, help='CUDA enabled'),
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
    params = load_params(Path(model_params))

    if not noisy and min_noise >= max_noise:
        log.project_logger.error('[red]Maximum noise range must be grater than minimum noise range')
        return

    if not any([inference_path.is_file(), inference_path.is_dir()]):
        log.project_logger.error('[red]Files for inference needed to be specified')
        return

    if params.pe_max_len < n_to_show:
        log.project_logger.error(f'[red]Parameter n_to_show={n_to_show} must be less than {params.pe_max_len}')
        return
    elif n_to_show == 0:
        n_to_show = params.pe_max_len

    device = torch.device(f'cuda:{gpu_id}' if cuda and torch.cuda.is_available() else 'cpu')

    with Progress(TextColumn('{task.description}', style='bright_blue'),
                  transient=True,
                  console=log.project_console
                  ) as progress:
        progress.add_task('Model loading...')
        z_reader = get_model(params.token_size, params.pe_max_len, params.num_layers,
                             params.d_model, params.n_heads, params.d_ff, params.dropout,
                             device, weights=Path(weights_path), silently=True)
    z_reader.eval()

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

        log.project_console.print(
            Panel(panel_group, title=file.name, border_style='magenta'),
            justify='center'
        )

        log.project_console.print(f'\nElapsed: {time() - start_time:>7.3f} s\n', style='bright_blue')


if __name__ == '__main__':
    try:
        cli()
    except Exception as e:
        log.project_logger.error(e)
        log.project_console.print_exception(show_locals=True)
        log.error_console.print_exception(show_locals=True)
