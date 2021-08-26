import json
from pathlib import Path
from time import time

import requests
import typer
from typer import Typer, Argument, Option

import utils

cli = Typer(name='ZreaderAPI')


@cli.command()
def params(host: str = Option('http://127.0.0.1', help='API host'),
           port: int = Option(5001, help='API port'),
           param: str = Option(None, help='Param name to receive')):
    if param:
        response = requests.get(url=f'{host}:{port}/params/{param}')
    else:
        response = requests.get(url=f'{host}:{port}/params')

    print(json.dumps(response.json(), indent=4))  # TODO use rich.print


@cli.command()
def zread(inference_path: str = Argument(..., help='Path to file or dir for inference'),
          host: str = Option('http://127.0.0.1', help='API host'),
          port: int = Option(5001, help='API port'),
          separator: str = Option(' ', help='Columns separator in the input files'),
          noisy: bool = Option(False, help='Input files are noisy texts'),
          min_noise: int = Option(3, help='Min noise parameter. Minimum value is alphabet size'),
          max_noise: int = Option(5, help='Max noise parameter. Maximum value is alphabet size'),
          beam_width: int = Option(1, help='Width for beam search algorithm. Maximum value is alphabet size'),
          console_width: int = Option(0, help='Console width for visualization. Zero value means for no restrictions'),
          delimiter: str = Option('', help='Delimiter for columns visualization')) -> None:
    inference_path = Path(inference_path).absolute()

    if not noisy and min_noise >= max_noise:
        typer.secho('Maximum noise range must be grater than minimum noise range',
                    fg=typer.colors.BRIGHT_RED, bold=True)
        return

    if not any([inference_path.is_file(), inference_path.is_dir()]):
        typer.secho('Files for inference needed to be specified', fg=typer.colors.BRIGHT_RED, bold=True)
        return

    files, files_columns = utils.get_files_columns(inference_path, separator, noisy, min_noise, max_noise,
                                                   console_width, delimiter)
    payload = {
        'data': None,
        'beam_width': beam_width,
        'delimiter': delimiter
    }

    for file_id, (file, file_columns) in enumerate(zip(files, files_columns), start=1):
        start_time = time()

        payload['data'] = file_columns
        response = requests.post(url=f'{host}:{port}/zread', json=payload)
        job_data = response.json()

        chains = None
        label = typer.style(f'[{file_id}/{len(files_columns)}] Processing {file.name}', fg=typer.colors.BLUE, bold=True)

        with typer.progressbar(length=job_data['size'], label=label, show_pos=True, show_eta=False) as progress:
            for _ in range(job_data['size']):
                response = requests.get(url=f'{host}:{port}/status/{job_data["identifier"]}')
                status_data = response.json()

                chains = status_data['chains']

                progress.update(1)

        requests.get(url=f'{host}:{port}/status/{job_data["identifier"]}')  # last request for job removing

        src_scale = len(file_columns) * max(2 * len(delimiter), 1) + 1 * len(delimiter)
        printing_scale = console_width if 0 < console_width < src_scale else src_scale

        print('-' * printing_scale)
        print(utils.visualize_columns(file_columns, delimiter))
        print('-' * printing_scale, end='\n\n')
        for chain, _ in chains:
            print(chain, end='\n\n')

        typer.secho(f'Elapsed: {time() - start_time:>7.3f}s\n', fg=typer.colors.BLUE, bold=True)


if __name__ == '__main__':
    cli()
