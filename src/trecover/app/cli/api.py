from typer import Typer, Option, Argument, Context

from trecover.config import var

cli = Typer(name='API-cli', add_completion=False, help='Manage API service')


@cli.callback(invoke_without_command=True)
def api_state_verification(ctx: Context) -> None:
    """
    Perform cli commands verification (state checking).

    Parameters
    ----------
    ctx : Context
        Typer (Click like) special internal object that holds state relevant
        for the script execution at every single level.

    """

    from trecover.config import log

    if var.API_PID.exists():
        if ctx.invoked_subcommand in ('start', None):
            log.project_console.print(':rocket: The API service is already started', style='bright_blue')
            ctx.exit(0)

    elif ctx.invoked_subcommand is None:
        api_start(host=var.FASTAPI_HOST, port=var.FASTAPI_PORT, loglevel=var.LogLevel.info,
                  concurrency=var.FASTAPI_WORKERS, attach=False, no_daemon=False)

    elif ctx.invoked_subcommand not in ('start', 'params', 'recover'):
        log.project_console.print('The API service is not started', style='yellow')
        ctx.exit(1)


@cli.command(name='config', help='Receive configuration or specific parameter of the model used for inference')
def api_config(url: str = Option(var.FASTAPI_URL, help='API url'),
               param: str = Option(None, help='Param name to receive')
               ) -> None:
    """
    Receive configuration or specific parameter of the model used for inference.

    Parameters
    ----------
    url : str, default=ENV(FASTAPI_URL) or 'http://localhost:8001'
        API url.
    param : str, default=None
        Param name whose value to receive. Receive all configuration values if None.

    """

    import json
    import requests

    from trecover.config import log

    if param:
        response = requests.get(url=f'{url}/config/{param}')
    else:
        response = requests.get(url=f'{url}/config')

    log.project_console.print(json.dumps(response.json(), indent=4))


@cli.command(name='recover', help='Send keyless reading API request')
def api_recover(data_path: str = Argument(..., help='Path to file or dir for data'),
                url: str = Option(var.FASTAPI_URL, help='API url'),
                separator: str = Option(' ', help='Columns separator in the input files'),
                noisy: bool = Option(False, help='Input files are noisy texts'),
                min_noise: int = Option(3, help='Min noise parameter. Minimum value is alphabet size'),
                max_noise: int = Option(5, help='Max noise parameter. Maximum value is alphabet size'),
                beam_width: int = Option(1, help='Width for beam search algorithm. Maximum value is alphabet size'),
                n_to_show: int = Option(0, help='Number of columns to visualize. Zero value means for no restrictions'),
                delimiter: str = Option('', help='Delimiter for columns visualization')
                ) -> None:
    """
    Send keyless reading API request.

    Parameters
    ----------
    data_path : Path
        Path to file or dir for data.
    url : str
        API url.
    separator : str, default=' '
        Columns separator in the input files.
    noisy : bool, default=False
        Indicates that input files are noisy texts.
    min_noise : int, default=3
        Min noise size per column. Minimum value is zero.
    max_noise : int, default=5
        Max noise size per column. Maximum value is alphabet size.
    beam_width : int, default=5
        Width for beam search algorithm. Maximum value is alphabet size.
    n_to_show : int, default=0
        Number of columns to visualize. Zero value means for no restriction's.
    delimiter : str, default=''
        Delimiter for columns visualization.

    Examples
    --------
    >>> trecover api recover examples/example_1.txt
    ╭──────────────────────────────────────────────────── example_1.txt ───────────────────────────────────────────────╮
    │                                                        Columns                                                   │
    │ ajocmbfeafodadbddciafqnahdfeihhkieeaacacafkdchddakhecmmlibfinaehbcbdiicejkeahnfemaeaadbkagacbdmahbibacfddfbbbca… │
    │ enpenkhgglrifflheioentrmjenkjnrmlhphdddeihliekeeeolflonpmctjolgkdeljjmljmmjiisjknjghgeelhkbddlpjjekrkdkilgiocii… │
    │ gsxtoplqkrtknksinktipwvnlnqqrstotoqspoejtsnoiuoflpohvtovqeutunjojlmksonosskpvxporrltnfgoprdemstnshnssgnronjreqj… │
    │ xvzwttqtxvxuoptowuxnxyzrwrrtwtyqwqvutrwrxvtxxwurrtqlwuqzvnwvxossmmpnutosuxlswyuvtttvqulrqzrrwuxtyqouwiuupwsxnrm… │
    │  y y yz zy  y w zy uz  yys   u tzs   x u         wx     wy w tuvpuwu  x yyowyz  z  wxyu     xyy   v yr    t yvw… │
    │                                                       Predicted                                                  │
    │ enpeoplearoundthecountrywereintothestreetstickedatheconvictionsspewditnessesinpentlandboardeddytheirwindowsbyra… │
    │                                                                                                                  │
    ╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
    Elapsed:   4.716 s

    Notes
    -----
    A larger "beam_width" parameter value can improve keyless reading, but it will also take longer to compute.

    """

    import requests
    from http import HTTPStatus
    from time import time, sleep
    from pathlib import Path

    from rich.console import Group
    from rich.panel import Panel
    from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
    from rich.text import Text

    from trecover.config import log
    from trecover.utils.cli import get_files_columns
    from trecover.utils.visualization import visualize_columns

    data_path = Path(data_path).absolute()

    if not noisy and min_noise >= max_noise:
        log.project_logger.error('[red]Maximum noise range must be grater than minimum noise range')
        return

    if not any([data_path.is_file(), data_path.is_dir()]):
        log.project_logger.error('[red]Files for inference needed to be specified')
        return

    files, files_columns = get_files_columns(data_path, separator, noisy, min_noise, max_noise, n_to_show)
    payload = {
        'columns': None,
        'beam_width': beam_width,
        'delimiter': delimiter
    }

    for file_id, (file, file_columns) in enumerate(zip(files, files_columns), start=1):
        start_time = time()

        file_columns = [''.join(set(c)) for c in file_columns]

        payload['columns'] = file_columns
        task_info = requests.post(url=f'{url}/recover', json=payload)
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

        requests.delete(url=f'{url}/{task_info["task_id"]}')

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


@cli.command(name='start', help='Start service')
def api_start(host: str = Option(var.FASTAPI_HOST, '--host', '-h', help='Bind socket to this host.'),
              port: int = Option(var.FASTAPI_PORT, '--port', '-p', help='Bind socket to this port.'),
              loglevel: var.LogLevel = Option(var.LogLevel.info, '--loglevel', '-l', help='Logging level.'),
              concurrency: int = Option(var.FASTAPI_WORKERS, '-c', help='The number of worker processes.'),
              attach: bool = Option(False, '--attach', '-a', is_flag=True,
                                    help='Attach output and error streams'),
              no_daemon: bool = Option(False, '--no-daemon', is_flag=True, help='Do not run as a daemon process')
              ) -> None:
    """
    Start API service.

    Parameters
    ----------
    host : str, default=ENV(FASTAPI_HOST) or 'localhost'
        Bind socket to this host.
    port : int, default=ENV(FASTAPI_PORT) or 8001
        Bind socket to this port.
    loglevel : {'debug', 'info', 'warning', 'error', 'critical'}, default='info'
        Level of logging.
    concurrency : int, default=ENV(FASTAPI_WORKERS) or 1
        The number of worker processes.
    attach : bool, default=False
        Attach output and error streams.
    no_daemon : bool, default=False
        Do not run as a daemon process.

    """

    from subprocess import run

    from trecover.config import log
    from trecover.utils.cli import start_service

    argv = [
        'uvicorn', 'trecover.app.api.trecoverapi:api',
        '--host', host,
        '--port', str(port),
        '--workers', str(concurrency),
        '--log-level', loglevel
    ]

    if no_daemon:
        run(argv)
    else:
        start_service(argv, name='API', logfile=log.API_LOG, pidfile=var.API_PID)

        if attach:
            api_attach(live=False)


@cli.command(name='stop', help='Stop service')
def api_stop() -> None:
    """ Stop API service. """

    from trecover.config import log
    from trecover.utils.cli import stop_service

    stop_service(name='API', pidfile=var.API_PID, logfile=log.API_LOG)


@cli.command(name='status', help='Display service status')
def api_status() -> None:
    """ Display API service status. """

    from trecover.utils.cli import check_service

    check_service(name='API', pidfile=var.API_PID)


@cli.command(name='attach', help='Attach local output stream to a service')
def api_attach(live: bool = Option(False, '--live', '-l', is_flag=True,
                                   help='Stream only fresh log records')
               ) -> None:
    """
    Attach local output stream to a running API service.

    Parameters
    ----------
    live : bool, Default=False
        Stream only fresh log records

    """

    from trecover.config import log
    from trecover.utils.cli import stream

    with log.project_console.screen():
        for record in stream(('API', log.API_LOG), live=live):
            log.project_console.print(record)

    log.project_console.clear()


if __name__ == '__main__':
    cli()
