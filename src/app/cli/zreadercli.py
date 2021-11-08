from enum import Enum
from pathlib import Path

from typer import Typer, Argument, Option, Context, BadParameter

from config import var, log


class PoolType(str, Enum):
    prefork = 'prefork'
    eventlet = 'eventlet'
    gevent = 'gevent'
    processes = 'processes'
    solo = 'solo'


class LogLevel(str, Enum):
    debug = 'debug'
    info = 'info'
    warning = 'warning'
    error = 'error'
    critical = 'critical'


# TODO help info
cli = Typer(name='Zreader-cli', add_completion=False)
download = Typer(name='Download-cli', add_completion=False)
train = Typer(name='Train-cli', add_completion=False)
dashboard = Typer(name='Dashboard-cli', add_completion=False)
api = Typer(name='API-cli', add_completion=False)
worker = Typer(name='Worker-cli', add_completion=False)
broker = Typer(name='Broker-cli', add_completion=False)
backend = Typer(name='Backend-cli', add_completion=False)

cli.add_typer(download, name='download')
cli.add_typer(train, name='train')
cli.add_typer(dashboard, name='dashboard')
cli.add_typer(api, name='api')
cli.add_typer(worker, name='worker')
cli.add_typer(broker, name='broker')
cli.add_typer(backend, name='backend')


# -------------------------------------------------Download commands----------------------------------------------------


@download.command(name='data', help='Download train data from Yandex disk')
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

    from zreader.utils.cli import download_archive_from_disk

    download_archive_from_disk(sharing_link, save_dir)


@download.command(name='artifacts', help='Download model artifacts from Yandex disk')
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

    from zreader.utils.cli import download_archive_from_disk

    download_archive_from_disk(sharing_link, save_dir)


# --------------------------------------------------Train commands------------------------------------------------------


# ---------------------------------------------Local client commands-------------------------------------------------


@cli.command()
def zread(inference_path: Path = Argument(..., help='Path to file or dir for inference', exists=True),
          model_params: Path = Option(var.INFERENCE_PARAMS_PATH, help='Path to model params json file', exists=True),
          weights_path: Path = Option(var.INFERENCE_WEIGHTS_PATH, help='Path to model weights', exists=True),
          cuda: bool = Option(var.CUDA, help='CUDA enabled'),
          gpu_id: int = Option(0, help='GPU id'),
          separator: str = Option(' ', help='Columns separator in the input files'),
          noisy: bool = Option(False, help='Input files are noisy texts'),
          min_noise: int = Option(3, help='Min noise parameter. Minimum value is zero'),
          max_noise: int = Option(5, help='Max noise parameter. Maximum value is alphabet size'),
          beam_width: int = Option(1, help='Width for beam search algorithm. Maximum value is alphabet size'),
          n_to_show: int = Option(0, help='Number of columns to visualize. Zero value means for no restrictions'),
          delimiter: str = Option('', help='Delimiter for columns visualization')
          ) -> None:
    from time import time

    import torch
    from rich.console import Group
    from rich.panel import Panel
    from rich.progress import Progress, TextColumn
    from rich.text import Text

    from zreader.utils.beam_search import beam_search, cli_interactive_loop
    from zreader.utils.cli import get_files_columns
    from zreader.utils.data import files_columns_to_tensors
    from zreader.utils.model import get_model, load_params
    from zreader.utils.visualization import visualize_columns, visualize_target

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


@cli.callback(invoke_without_command=True)
def cli_state_verification(ctx: Context,
                           file: str = Option('zreader-compose.toml', '--file', '-f',
                                              help='Zreader configuration file'),

                           ) -> None:
    from zreader.utils.cli import parse_config
    from zreader.utils.docker import is_docker_running

    if ctx.invoked_subcommand is None:
        log.project_console.print(ctx.get_help(), markup=False)
        ctx.exit(0)

    if ctx.invoked_subcommand == 'up':
        if not is_docker_running():
            log.project_console.print('Docker engine is not running', style='red')
            ctx.exit(1)

        ctx.params['conf'] = parse_config(Path(file))


@cli.command()
def up(ctx: Context) -> None:
    from zreader.utils.cli import check_service

    conf = ctx.parent.params['conf']

    if var.DASHBOARD_PID.exists():
        check_service(name='dashboard', pidfile=var.DASHBOARD_PID)
    else:
        print(conf)


# ------------------------------------------------API client commands---------------------------------------------------


@api.command(name='params')
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


@api.command(name='zread')
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


# ---------------------------------------------Dashboard service commands-----------------------------------------------


@dashboard.callback(invoke_without_command=True)
def dashboard_state_verification(ctx: Context) -> None:
    if var.DASHBOARD_PID.exists():
        if ctx.invoked_subcommand in ('start', None):
            log.project_console.print(':rocket: The dashboard service is already started', style='bright_blue')
            ctx.exit(0)

    elif ctx.invoked_subcommand is None:
        dashboard_start(host=var.STREAMLIT_HOST, port=var.STREAMLIT_PORT, loglevel=LogLevel.info, attach=False)

    elif ctx.invoked_subcommand != 'start':
        log.project_console.print('The dashboard service is not started', style='yellow')
        ctx.exit(1)


@dashboard.command(name='start')
def dashboard_start(host: str = Option(var.STREAMLIT_HOST, '--host', '-h', help='Bind socket to this host.'),
                    port: int = Option(var.STREAMLIT_PORT, '--port', '-p', help='Bind socket to this port.'),
                    loglevel: LogLevel = Option(LogLevel.info, '--loglevel', '-l', help='Logging level.'),
                    attach: bool = Option(False, '--attach', '-a', is_flag=True,
                                          help='Attach output and error streams')
                    ) -> None:
    from app.api import dashboard
    from subprocess import Popen, STDOUT, CREATE_NO_WINDOW

    argv = ['streamlit',
            'run', dashboard.__file__,
            '--server.address', host,
            '--server.port', str(port),
            '--logger.level', loglevel,
            '--global.suppressDeprecationWarnings', 'True',
            '--theme.backgroundColor', '#E7EAD9',
            '--theme.secondaryBackgroundColor', '#DFE3D0',
            '--theme.primaryColor', '#FF8068',
            '--theme.textColor', '#157D96'
            ]

    process = Popen(argv, creationflags=CREATE_NO_WINDOW, stdout=log.DASHBOARD_LOG.open(mode='w'), stderr=STDOUT,
                    universal_newlines=True)

    with var.DASHBOARD_PID.open('w') as f:
        f.write(str(process.pid))

    log.project_console.print('The dashboard service is started', style='bright_blue')

    if attach:
        dashboard_attach(live=False)


@dashboard.command(name='stop')
def dashboard_stop() -> None:
    from zreader.utils.cli import stop_service

    stop_service(name='dashboard', pidfile=var.DASHBOARD_PID)


@dashboard.command(name='status')
def dashboard_status() -> None:
    from zreader.utils.cli import check_service

    check_service(name='dashboard', pidfile=var.DASHBOARD_PID)


@dashboard.command(name='attach')
def dashboard_attach(live: bool = Option(False, '--live', '-l', is_flag=True,
                                         help='Stream only fresh log records')
                     ) -> None:
    from zreader.utils.cli import stream

    with log.project_console.screen():
        for record in stream(logfile=log.DASHBOARD_LOG, live=live):
            log.project_console.print(record.strip())


# -----------------------------------------------API service commands---------------------------------------------------


@api.callback(invoke_without_command=True)
def api_state_verification(ctx: Context) -> None:
    if var.API_PID.exists():
        if ctx.invoked_subcommand in ('start', None):
            log.project_console.print(':rocket: The API service is already started', style='bright_blue')
            ctx.exit(0)

    elif ctx.invoked_subcommand is None:
        api_start(host=var.FASTAPI_HOST, port=var.FASTAPI_PORT, concurrency=var.FASTAPI_WORKERS,
                  loglevel=LogLevel.info, attach=False)

    elif ctx.invoked_subcommand != 'start':
        log.project_console.print('The API service is not started', style='yellow')
        ctx.exit(1)


@api.command(name='start')
def api_start(host: str = Option(var.FASTAPI_HOST, '--host', '-h', help='Bind socket to this host.'),
              port: int = Option(var.FASTAPI_PORT, '--port', '-p', help='Bind socket to this port.'),
              concurrency: int = Option(var.FASTAPI_WORKERS, '-c', help='The number of worker processes.'),
              loglevel: LogLevel = Option(LogLevel.info, '--loglevel', '-l', help='Logging level.'),
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
        dashboard_attach(live=False)


@api.command(name='stop')
def api_stop() -> None:
    from zreader.utils.cli import stop_service

    stop_service(name='API', pidfile=var.API_PID)


@api.command(name='status')
def api_status() -> None:
    from zreader.utils.cli import check_service

    check_service(name='API', pidfile=var.API_PID)


@api.command(name='attach')
def api_attach(live: bool = Option(False, '--live', '-l', is_flag=True,
                                   help='Stream only fresh log records')
               ) -> None:
    from zreader.utils.cli import stream

    with log.project_console.screen():
        for record in stream(logfile=log.API_LOG, live=live):
            log.project_console.print(record.strip())


# ----------------------------------------------Worker service commands-------------------------------------------------


@worker.callback(invoke_without_command=True)
def worker_state_verification(ctx: Context) -> None:
    if var.WORKER_PID.exists():
        if ctx.invoked_subcommand in ('start', None):
            log.project_console.print(':rocket: The worker service is already started', style='bright_blue')
            ctx.exit(0)

    elif ctx.invoked_subcommand is None:
        worker_start(name='ZReaderWorker', concurrency=var.CELERY_WORKERS, pool=PoolType.solo,
                     broker_url=var.CELERY_BROKER, backend_url=var.CELERY_BACKEND, loglevel=LogLevel.info,
                     attach=False)

    elif ctx.invoked_subcommand != 'start':
        log.project_console.print('The worker service is not started', style='yellow')
        ctx.exit(1)


@worker.command(name='start')
def worker_start(name: str = Option('ZReaderWorker', '--name', '-n', help='Set custom worker name.'),
                 concurrency: int = Option(var.CELERY_WORKERS, '-c', help='The number of worker processes/threads.'),
                 pool: PoolType = Option(PoolType.solo, '--pool', '-p', help='Worker processes/threads pool type.'),
                 loglevel: LogLevel = Option(LogLevel.info, '--loglevel', '-l', help='Logging level.'),
                 broker_url: str = Option(var.CELERY_BROKER, '--broker', help='Broker url.'),
                 backend_url: str = Option(var.CELERY_BACKEND, '--backend', help='Backend url.'),
                 attach: bool = Option(False, '--attach', '-a', is_flag=True,
                                       help='Attach output and error streams')
                 ) -> None:
    import platform
    from subprocess import Popen, CREATE_NO_WINDOW, STDOUT

    if platform.system() == 'Windows' and pool != PoolType.solo:
        raise BadParameter("Windows platform only supports 'solo' pool")

    argv = [
        'celery',
        '--broker', broker_url,
        '--result-backend', backend_url,
        '--app', 'src.app.api.backend.celeryapp', 'worker',
        '--hostname', name,
        '--concurrency', str(concurrency),
        '--pool', pool,
        '--loglevel', loglevel
    ]

    process = Popen(argv, creationflags=CREATE_NO_WINDOW, stdout=log.WORKER_LOG.open(mode='w'), stderr=STDOUT,
                    universal_newlines=True)

    with var.WORKER_PID.open('w') as f:
        f.write(str(process.pid))

    log.project_console.print('The worker service is started', style='bright_blue')

    if attach:
        dashboard_attach(live=False)


@worker.command(name='stop')
def worker_stop() -> None:
    from zreader.utils.cli import stop_service

    stop_service(name='worker', pidfile=var.WORKER_PID)


@worker.command(name='status')
def worker_status() -> None:
    from zreader.utils.cli import check_service

    check_service(name='worker', pidfile=var.WORKER_PID)


@worker.command(name='attach')
def worker_attach(live: bool = Option(False, '--live', '-l', is_flag=True,
                                      help='Stream only fresh log records')
                  ) -> None:
    from zreader.utils.cli import stream

    with log.project_console.screen():
        for record in stream(logfile=log.WORKER_LOG, live=live):
            log.project_console.print(record.strip())


# ----------------------------------------------Broker service commands-------------------------------------------------

@broker.callback(invoke_without_command=True)
def broker_state_verification(ctx: Context) -> None:
    from zreader.utils.docker import is_docker_running, get_container

    if not is_docker_running():
        log.project_console.print('Docker engine is not running', style='red')
        ctx.exit(1)

    elif (container := get_container(var.BROKER_ID)) and container.status == 'running':
        if ctx.invoked_subcommand in ('start', None):
            log.project_console.print(':rocket: The broker service is already started', style='bright_blue')
            ctx.exit(0)

    elif ctx.invoked_subcommand is None:
        broker_start(port=var.BROKER_PORT, ui_port=var.BROKER_UI_PORT, attach=False, auto_remove=False)

    elif ctx.invoked_subcommand not in ('start', 'prune'):
        log.project_console.print('The broker service is not started', style='yellow')
        ctx.exit(1)


@broker.command(name='start')
def broker_start(port: int = Option(var.BROKER_PORT, '--port', '-p',
                                    help='Bind socket to this port.'),
                 ui_port: int = Option(var.BROKER_UI_PORT, '--port', '-p',
                                       help='Bind UI socket to this port.'),
                 attach: bool = Option(False, '--attach', '-a', is_flag=True,
                                       help='Attach local standard input, output, and error streams'),
                 auto_remove: bool = Option(False, '--rm', is_flag=True,
                                            help='Remove docker container after service exit')
                 ) -> None:
    from zreader.utils.docker import client, get_container, get_image, pull_image
    from rich.prompt import Confirm

    if not (image := get_image(var.BROKER_IMAGE)):
        with log.project_console.screen(hide_cursor=False):
            if not Confirm.ask(f"The broker image '{var.BROKER_IMAGE}' is needed to be pulled.\nContinue?",
                               default=True):
                return

        image = pull_image(var.BROKER_IMAGE)

    if container := get_container(var.BROKER_ID):
        container.start()

        log.project_console.print(f'The broker service is started', style='bright_blue')

    else:
        client.containers.run(image=image.id,
                              name=var.BROKER_ID,
                              auto_remove=auto_remove,
                              detach=True,
                              stdin_open=True,
                              stdout=True,
                              tty=True,
                              stop_signal='SIGTERM',
                              ports={5672: port, 15672: ui_port})

        log.project_console.print(f'The broker service is launched', style='bright_blue')

    if attach:
        broker_attach()


@broker.command(name='stop')
def broker_stop(prune: bool = Option(False, '--prune', '-p', is_flag=True,
                                     help='Prune broker.'),
                v: bool = Option(False, '--volume', '-v', is_flag=True,
                                 help='Remove the volumes associated with the container')
                ) -> None:
    from zreader.utils.docker import get_container

    get_container(var.BROKER_ID).stop()

    log.project_console.print('The broker service is stopped', style='bright_blue')

    if prune:
        broker_prune(force=False, v=v)


@broker.command(name='prune')
def broker_prune(force: bool = Option(False, '--force', '-f', is_flag=True,
                                      help='Force the removal of a running container'),
                 v: bool = Option(False, '--volume', '-v', is_flag=True,
                                  help='Remove the volumes associated with the container')
                 ) -> None:
    from zreader.utils.docker import get_container

    container = get_container(var.BROKER_ID)

    if container.status == 'running' and not force:
        log.project_console.print('You need to stop broker service before pruning or use --force flag', style='yellow')
    else:
        container.remove(v=v, force=force)
        log.project_console.print('The broker service is pruned', style='bright_blue')


@broker.command(name='status')
def broker_status() -> None:
    from zreader.utils.docker import get_container

    log.project_console.print(f'Broker status: {get_container(var.BROKER_ID).status}', style='bright_blue')


@broker.command(name='attach')
def broker_attach() -> None:
    from zreader.utils.docker import get_container

    with log.project_console.screen(hide_cursor=True):
        for line in get_container(var.BROKER_ID).attach(stream=True, logs=True):
            log.project_console.print(line.decode().strip())


# ----------------------------------------------Backend service commands------------------------------------------------


@backend.callback(invoke_without_command=True)
def backend_state_verification(ctx: Context) -> None:
    from zreader.utils.docker import is_docker_running, get_container

    if not is_docker_running():
        log.project_console.print('Docker engine is not running', style='red')
        ctx.exit(1)

    elif (container := get_container(var.BACKEND_ID)) and container.status == 'running':
        if ctx.invoked_subcommand in ('start', None):
            log.project_console.print(':rocket: The backend service is already started', style='bright_blue')
            ctx.exit(0)

    elif ctx.invoked_subcommand is None:
        backend_start(port=var.BACKEND_PORT, attach=False, auto_remove=False)

    elif ctx.invoked_subcommand not in ('start', 'prune'):
        log.project_console.print('The backend service is not started', style='yellow')
        ctx.exit(1)


@backend.command(name='start')
def backend_start(port: int = Option(var.BACKEND_PORT, '--port', '-p',
                                     help='Bind socket to this port.'),
                  attach: bool = Option(False, '--attach', '-a', is_flag=True,
                                        help='Attach local standard input, output, and error streams'),
                  auto_remove: bool = Option(False, '--rm', is_flag=True,
                                             help='Remove docker container after service exit')
                  ) -> None:
    from zreader.utils.docker import client, get_container, get_image, pull_image
    from rich.prompt import Confirm

    if not (image := get_image(var.BACKEND_IMAGE)):
        with log.project_console.screen(hide_cursor=False):
            if not Confirm.ask(f"The backend image '{var.BACKEND_IMAGE}' is needed to be pulled.\nContinue?",
                               default=True):
                return

        image = pull_image(var.BACKEND_IMAGE)

    if container := get_container(var.BACKEND_ID):
        container.start()

        log.project_console.print(f'The backend service is started', style='bright_blue')

    else:
        client.containers.run(image=image.id,
                              name=var.BACKEND_ID,
                              auto_remove=auto_remove,
                              detach=True,
                              stdin_open=True,
                              stdout=True,
                              tty=True,
                              stop_signal='SIGTERM',
                              ports={6379: port})

        log.project_console.print(f'The backend service is launched', style='bright_blue')

    if attach:
        backend_attach()


@backend.command(name='stop')
def backend_stop(prune: bool = Option(False, '--prune', '-p', is_flag=True,
                                      help='Prune backend.'),
                 v: bool = Option(False, '--volume', '-v', is_flag=True,
                                  help='Remove the volumes associated with the container')
                 ) -> None:
    from zreader.utils.docker import get_container

    get_container(var.BACKEND_ID).stop()

    log.project_console.print('The backend service is stopped', style='bright_blue')

    if prune:
        backend_prune(force=False, v=v)


@backend.command(name='prune')
def backend_prune(force: bool = Option(False, '--force', '-f', is_flag=True,
                                       help='Force the removal of a running container'),
                  v: bool = Option(False, '--volume', '-v', is_flag=True,
                                   help='Remove the volumes associated with the container')
                  ) -> None:
    from zreader.utils.docker import get_container

    container = get_container(var.BACKEND_ID)

    if container.status == 'running' and not force:
        log.project_console.print('You need to stop backend service before pruning or use --force flag', style='yellow')
    else:
        container.remove(v=v, force=force)
        log.project_console.print('Backend service is pruned', style='bright_blue')


@backend.command(name='status')
def backend_status() -> None:
    from zreader.utils.docker import get_container

    status = container.status if (container := get_container(var.BACKEND_ID)) else '[yellow]is not started'

    log.project_console.print(f'Backend status: {status}', style='bright_blue')


@backend.command(name='attach')
def backend_attach() -> None:
    from zreader.utils.docker import get_container

    with log.project_console.screen(hide_cursor=True):
        for line in get_container(var.BACKEND_ID).attach(stream=True, logs=True):
            log.project_console.print(line.decode().strip())


if __name__ == '__main__':
    try:
        cli()
    except Exception as e:
        log.project_logger.error(e)
        log.project_console.print_exception(show_locals=True)
        log.error_console.print_exception(show_locals=True)
