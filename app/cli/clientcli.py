import json
from http import HTTPStatus
from pathlib import Path
from time import time, sleep

import requests
from rich.console import Group
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.text import Text
from typer import Typer, Argument, Option

import config
from src.utils.cli import get_files_columns
from src.utils.visualization import visualize_columns

cli = Typer(name='Client-cli')


@cli.command()
def params(url: str = Option(config.FASTAPI_URL, help='API url'),
           param: str = Option(None, help='Param name to receive')):
    if param:
        response = requests.get(url=f'{url}/params/{param}')
    else:
        response = requests.get(url=f'{url}/params')

    config.project_console.print(json.dumps(response.json(), indent=4))


@cli.command()
def zread(inference_path: str = Argument(..., help='Path to file or dir for inference'),
          url: str = Option(config.FASTAPI_URL, help='API url'),
          separator: str = Option(' ', help='Columns separator in the input files'),
          noisy: bool = Option(False, help='Input files are noisy texts'),
          min_noise: int = Option(3, help='Min noise parameter. Minimum value is alphabet size'),
          max_noise: int = Option(5, help='Max noise parameter. Maximum value is alphabet size'),
          beam_width: int = Option(1, help='Width for beam search algorithm. Maximum value is alphabet size'),
          n_to_show: int = Option(0, help='Number of columns to visualize. Zero value means for no restrictions'),
          delimiter: str = Option('', help='Delimiter for columns visualization')) -> None:
    inference_path = Path(inference_path).absolute()

    if not noisy and min_noise >= max_noise:
        config.project_logger.error('[red]Maximum noise range must be grater than minimum noise range')
        return

    if not any([inference_path.is_file(), inference_path.is_dir()]):
        config.project_logger.error('[red]Files for inference needed to be specified')
        return

    files, files_columns = get_files_columns(inference_path, separator, noisy, min_noise, max_noise, n_to_show)
    payload = {
        'data': None,
        'beam_width': beam_width,
        'delimiter': delimiter
    }

    for file_id, (file, file_columns) in enumerate(zip(files, files_columns), start=1):
        start_time = time()

        file_columns = [''.join(set(c)) for c in file_columns]

        payload['data'] = file_columns
        task_info = requests.post(url=f'{url}/zread', json=payload)
        task_info = task_info.json()

        if not task_info['task_id']:
            config.project_logger.error(f'{file_id}/{len(files_columns)} [red]Failed {file.name}:\n'
                                        f'{task_info["message"]}')
            continue

        task_status = requests.get(url=f'{url}/status/{task_info["task_id"]}')
        task_status = task_status.json()

        label = f'{file_id}/{len(files_columns)} Processing {file.name}'

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
            request_progress = progress.add_task(label, total=len(file_columns))

            while task_status['status_code'] == HTTPStatus.PROCESSING:
                task_status = requests.get(url=f'{url}/status/{task_info["task_id"]}')
                task_status = task_status.json()

                progress.update(request_progress, completed=task_status['progress'])

                sleep(0.5)

        requests.delete(url=f'{url}/status/{task_info["task_id"]}')

        if task_status['status_code'] != HTTPStatus.OK:
            config.project_logger.error(f'{file_id}/{len(files_columns)} [red]Failed {file.name}:\n'
                                        f'{task_status["message"]}')
            continue

        columns = visualize_columns(file_columns, delimiter=delimiter, as_rows=True)
        columns = (Text(row, style='bright_blue', overflow='ellipsis', no_wrap=True) for row in columns)

        chains = [Text(chain, style='cyan', justify='center', overflow='ellipsis', end='\n\n')
                  for (chain, _) in task_status['chains']]

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
