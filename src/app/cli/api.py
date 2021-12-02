from typer import Typer, Option, Argument, Context

from config import var, log

cli = Typer(name='API-cli', add_completion=False)


@cli.callback(invoke_without_command=True)
def api_state_verification(ctx: Context) -> None:
    if var.API_PID.exists():
        if ctx.invoked_subcommand in ('start', None):
            log.project_console.print(':rocket: The API service is already started', style='bright_blue')
            ctx.exit(0)

    elif ctx.invoked_subcommand is None:
        api_start(host=var.FASTAPI_HOST, port=var.FASTAPI_PORT, loglevel=var.LogLevel.info,
                  concurrency=var.FASTAPI_WORKERS, attach=False)

    elif ctx.invoked_subcommand not in ('start', 'params', 'zread'):
        log.project_console.print('The API service is not started', style='yellow')
        ctx.exit(1)


@cli.command(name='params')
def api_params(url: str = Option(var.FASTAPI_URL, help='API url'),
               param: str = Option(None, help='Param name to receive')
               ) -> None:
    import json
    import requests

    if param:
        response = requests.get(url=f'{url}/params/{param}')
    else:
        response = requests.get(url=f'{url}/params')

    log.project_console.print(json.dumps(response.json(), indent=4))


@cli.command(name='zread')
def api_zread(inference_path: str = Argument(..., help='Path to file or dir for inference'),
              url: str = Option(var.FASTAPI_URL, help='API url'),
              separator: str = Option(' ', help='Columns separator in the input files'),
              noisy: bool = Option(False, help='Input files are noisy texts'),
              min_noise: int = Option(3, help='Min noise parameter. Minimum value is alphabet size'),
              max_noise: int = Option(5, help='Max noise parameter. Maximum value is alphabet size'),
              beam_width: int = Option(1, help='Width for beam search algorithm. Maximum value is alphabet size'),
              n_to_show: int = Option(0, help='Number of columns to visualize. Zero value means for no restrictions'),
              delimiter: str = Option('', help='Delimiter for columns visualization')
              ) -> None:
    import requests
    from http import HTTPStatus
    from time import time, sleep
    from pathlib import Path

    from rich.console import Group
    from rich.panel import Panel
    from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
    from rich.text import Text

    from zreader.utils.cli import get_files_columns
    from zreader.utils.visualization import visualize_columns

    inference_path = Path(inference_path).absolute()

    if not noisy and min_noise >= max_noise:
        log.project_logger.error('[red]Maximum noise range must be grater than minimum noise range')
        return

    if not any([inference_path.is_file(), inference_path.is_dir()]):
        log.project_logger.error('[red]Files for inference needed to be specified')
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
            log.project_logger.error(f'{file_id}/{len(files_columns)} [red]Failed {file.name}:\n'
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
            log.project_logger.error(f'{file_id}/{len(files_columns)} [red]Failed {file.name}:\n'
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

        log.project_console.print(
            Panel(panel_group, title=file.name, border_style='magenta'),
            justify='center'
        )

        log.project_console.print(f'\nElapsed: {time() - start_time:>7.3f} s\n', style='bright_blue')


@cli.command(name='start')
def api_start(host: str = Option(var.FASTAPI_HOST, '--host', '-h', help='Bind socket to this host.'),
              port: int = Option(var.FASTAPI_PORT, '--port', '-p', help='Bind socket to this port.'),
              loglevel: var.LogLevel = Option(var.LogLevel.info, '--loglevel', '-l', help='Logging level.'),
              concurrency: int = Option(var.FASTAPI_WORKERS, '-c', help='The number of worker processes.'),
              attach: bool = Option(False, '--attach', '-a', is_flag=True,
                                    help='Attach output and error streams')
              ) -> None:
    from subprocess import Popen, CREATE_NO_WINDOW, STDOUT

    argv = [
        'uvicorn', 'app.api.zreaderapi:api',
        '--host', host,
        '--port', str(port),
        '--workers', str(concurrency),
        '--log-level', loglevel
    ]

    process = Popen(argv, creationflags=CREATE_NO_WINDOW, stdout=log.API_LOG.open(mode='w'), stderr=STDOUT,
                    universal_newlines=True)

    with var.API_PID.open('w') as f:
        f.write(str(process.pid))

    log.project_console.print('The API service is started', style='bright_blue')

    if attach:
        api_attach(live=False)


@cli.command(name='stop')
def api_stop() -> None:
    from zreader.utils.cli import stop_service

    stop_service(name='API', pidfile=var.API_PID)


@cli.command(name='status')
def api_status() -> None:
    from zreader.utils.cli import check_service

    check_service(name='API', pidfile=var.API_PID)


@cli.command(name='attach')
def api_attach(live: bool = Option(False, '--live', '-l', is_flag=True,
                                   help='Stream only fresh log records')
               ) -> None:
    from zreader.utils.cli import stream

    with log.project_console.screen():
        for record in stream(logfile=log.API_LOG, live=live):
            log.project_console.print(record.strip())


if __name__ == '__main__':
    try:
        cli()
    except Exception as e:
        log.project_logger.error(e)
        log.project_console.print_exception(show_locals=True)
        log.error_console.print_exception(show_locals=True)
