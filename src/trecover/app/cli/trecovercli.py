from pathlib import Path

import typer
from typer import Typer, Argument, Option, Context

from trecover.app.cli import download, train, mlflow, dashboard, api, worker, broker, backend
from trecover.config import var

cli = Typer(name='TRecover-cli', add_completion=False)

cli.add_typer(download.cli, name='download')
cli.add_typer(train.cli, name='train')
cli.add_typer(mlflow.cli, name='mlflow')
cli.add_typer(dashboard.cli, name='dashboard')
cli.add_typer(api.cli, name='api')
cli.add_typer(worker.cli, name='worker')
cli.add_typer(broker.cli, name='broker')
cli.add_typer(backend.cli, name='backend')


@cli.command(help='Perform keyless reading')
def recover(data_path: Path = Argument(..., help='Path to file or dir for data', exists=True),
            model_params: Path = Option(var.INFERENCE_PARAMS_PATH, help='Path to model params json file', exists=True),
            weights_path: Path = Option(var.INFERENCE_WEIGHTS_PATH, help='Path to model weights', exists=True),
            cuda: bool = Option(var.CUDA, envvar='CUDA', help='CUDA enabled'),
            gpu_id: int = Option(0, help='GPU id'),
            separator: str = Option(' ', help='Columns separator in the input files'),
            noisy: bool = Option(False, help='Input files are noisy texts'),
            min_noise: int = Option(3, help='Min noise parameter. Minimum value is zero'),
            max_noise: int = Option(5, help='Max noise parameter. Maximum value is alphabet size'),
            beam_width: int = Option(5, help='Width for beam search algorithm. Maximum value is alphabet size'),
            n_to_show: int = Option(0, help="Number of columns to visualize. Zero value means for no restriction's"),
            delimiter: str = Option('', help='Delimiter for columns visualization')
            ) -> None:
    """
    Perform keyless reading locally.

    Parameters
    ----------
    data_path : Path
        Path to file or dir for data.
    model_params : Path
        Path to model params json file.
    weights_path : Path
        Path to model weights.
    cuda : bool
        CUDA enabled.
    gpu_id : int, default=0
        GPU id on which perform computations.
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
    >>> trecover recover examples/example_1.txt
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

    from time import time

    import torch
    from rich.console import Group
    from rich.panel import Panel
    from rich.progress import Progress, TextColumn
    from rich.text import Text

    from trecover.config import log
    from trecover.utils.beam_search import beam_search, cli_interactive_loop
    from trecover.utils.cli import get_files_columns
    from trecover.utils.transform import files_columns_to_tensors
    from trecover.utils.model import get_model, load_params
    from trecover.utils.transform import tensor_to_columns, tensor_to_target
    from trecover.utils.visualization import visualize_columns, visualize_target

    data_path = Path(data_path)
    params = load_params(Path(model_params))

    if not noisy and min_noise >= max_noise:
        log.project_logger.error('[red]Maximum noise range must be grater than minimum noise range')
        return

    if not any([data_path.is_file(), data_path.is_dir()]):
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
        model = get_model(params.token_size, params.pe_max_len, params.num_layers,
                          params.d_model, params.n_heads, params.d_ff, params.dropout,
                          device, weights=Path(weights_path), silently=True)
    model.eval()

    files, files_columns = get_files_columns(data_path, separator, noisy, min_noise, max_noise, n_to_show)
    files_src = files_columns_to_tensors(files_columns, device)

    for file_id, (file, src) in enumerate(zip(files, files_src), start=1):
        start_time = time()

        loop_label = f'{file_id}/{len(files_src)} Processing {file.name}'
        chains = beam_search(src, model, beam_width, device,
                             beam_loop=cli_interactive_loop(label=loop_label))
        chains = [Text(visualize_target(tensor_to_target(chain), delimiter=delimiter), style='cyan', justify='center',
                       overflow='ellipsis', end='\n\n') for (chain, _) in chains]

        columns = visualize_columns(tensor_to_columns(src), delimiter=delimiter, as_rows=True)
        columns = (Text(row, style='bright_blue', overflow='ellipsis', no_wrap=True, justify='left') for row in columns)

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


@cli.callback(invoke_without_command=True, help='')
def cli_state_verification(ctx: Context,
                           config_file: Path = Option(var.BASE_DIR / 'trecover-compose.toml', '--file', '-f',
                                                      file_okay=True,
                                                      help='Path to TRecover configuration file for "up" command'),
                           ) -> None:
    """
    Perform cli commands verification (state checking) and config file parsing.

    Parameters
    ----------
    ctx : Context
        Typer (Click like) special internal object that holds state relevant
        for the script execution at every single level.
    config_file : Path, default=var.BASE_DIR / 'trecover-compose.toml'
        Path to TRecover configuration file for "up" command.
    """

    if ctx.invoked_subcommand == 'init':
        return
    elif not var.BASE_INIT.exists():
        typer.echo(typer.style('You need to initialize the project environment.\n'
                               'For more information use: trecover init --help',
                               fg=typer.colors.RED))
        ctx.exit(1)

    from trecover.config import log

    if ctx.invoked_subcommand is None:
        log.project_console.print(ctx.get_help(), markup=False)
        ctx.exit(0)

    if ctx.invoked_subcommand in ('up', 'down', 'status', 'attach'):
        from trecover.utils.docker import is_docker_running
        from trecover.utils.cli import parse_config

        if not is_docker_running():
            log.project_console.print('Docker engine is not running', style='red')
            ctx.exit(1)

        if not config_file.exists():
            log.project_console.print('Defined TRecover configuration file does not exist', style='red')
            ctx.exit(1)

        ctx.params['conf'] = parse_config(config_file)


@cli.command(help="Initialize project's environment")
def init(base: Path = Option(Path().absolute(), '--base', '-b', help="Path to the project's base directory"),
         relocate: bool = Option(False, '--relocate', '-r', is_flag=True, help='Relocate an existing environment')
         ) -> None:
    """
    Initialize project's environment.

    Parameters
    ----------
    base : Path, default='./'
        Path to the project's base directory.
    relocate : bool, default=False
        Relocate an existing environment

    """

    from shutil import move, Error

    def rebase(src: Path, dst: Path) -> None:
        try:
            move(str(src), str(dst))
        except (PermissionError, Error):
            typer.echo(typer.style(f'Failed to relocate: {src}', fg=typer.colors.YELLOW))
        else:
            typer.echo(typer.style(f'Relocated: {src}', fg=typer.colors.BRIGHT_BLUE))

    base.mkdir(parents=True, exist_ok=True)

    with var.BASE_INIT.open(mode='w') as f:
        f.write(base.as_posix())

    typer.echo(typer.style("Project's environment is initialized.", fg=typer.colors.BRIGHT_BLUE))

    if relocate:
        if var.LOGS_DIR.exists():
            rebase(var.LOGS_DIR, base)
        if var.INFERENCE_DIR.exists():
            rebase(var.INFERENCE_DIR, base)
        if var.DATA_DIR.exists():
            rebase(var.DATA_DIR, base)
        if var.EXPERIMENTS_DIR.exists():
            rebase(var.EXPERIMENTS_DIR, base)


@cli.command(help='Start services')
def up(ctx: Context,
       attach_stream: bool = Option(False, '--attach', '-a', is_flag=True,
                                    help='Attach output and error streams')
       ) -> None:
    """
    Start services: Dashboard, API, Worker, Broker, Backend.

    Command uses trecover-compose.toml config file specified by --file and
    attaches output and error streams if --attach flag is set.

    Parameters
    ----------
    ctx : Context
        Typer (Click like) special internal object that holds state relevant
        for the script execution at every single level.
    attach_stream : bool, default=False
        Attach output and error streams.

    Config Variables
    ----------------
    dashboard host : str, default=ENV(STREAMLIT_HOST) or 'localhost'
        The address where the server will listen for client and browser connections.
        Use this if you want to bind the server to a specific address. If set, the server
        will only be accessible from this address, and not from any aliases (like localhost).

    dashboard port : int, default=ENV(STREAMLIT_PORT) or 8000
        The port where the server will listen for browser connections.

    dashboard loglevel : {'debug', 'info', 'warning', 'error', 'critical'}, default='info'
        Level of logging.

    api host : str, default=ENV(FASTAPI_HOST) or 'localhost'
        Bind socket to this host.

    api port : int, default=ENV(FASTAPI_PORT) or 8001
        Bind socket to this port.

    api loglevel : {'debug', 'info', 'warning', 'error', 'critical'}, default='info'
        Level of logging.

    api concurrency : int, default=ENV(FASTAPI_WORKERS) or 1
        The number of worker processes.

    worker name : str, default='TRecoverWorker'
        Custom worker hostname.

    worker pool : str, {'prefork', 'eventlet', 'gevent', 'processes', 'solo'}, default='solo'
        Worker processes/threads pool type.

    worker loglevel : {'debug', 'info', 'warning', 'error', 'critical'}, default='info'
        Level of logging.

    worker concurrency : int, default=ENV(CELERY_WORKERS) or 1
        The number of worker processes.

    worker broker_url : str, default=ENV(CELERY_BROKER) or 'pyamqp://guest@localhost'
        Broker url.

    worker backend_url : str, default=ENV(CELERY_BACKEND) or 'redis://localhost'
        Backend url.

    broker port : int, default=ENV(BROKER_PORT) or 5672
        Bind broker socket to this port.

    broker ui_port : int, default=ENV(BROKER_UI_PORT) or 15672
        Bind UI socket to this port.

    broker auto_remove : bool, default=False
        Remove broker docker container after service exit.

    backend port : int, default=ENV(BACKEND_PORT) or 6379
        Bind backend socket to this port.

    backend auto_remove : bool, default=False
        Remove backend docker container after service exit.

    Examples
    --------
    # trecover-compose.toml

        [dashboard]
        host = "localhost"
        port = 8000
        loglevel = 'info'

        [api]
        host = "localhost"
        port = 8001
        loglevel = 'info'
        concurrency = 1

        [worker]
        name = "TRecoverWorker"
        pool = "solo"
        loglevel = "info"
        concurrency = 1
        broker_url = "pyamqp://guest@localhost:5672"
        backend_url = "redis://localhost:6379"

        [broker]
        port = 5672
        ui_port = 15672
        auto_remove = false

        [backend]
        port = 6379
        auto_remove = false

    """

    from trecover.config import log
    from trecover.utils.docker import get_container
    from trecover.utils.cli import check_service

    conf = ctx.parent.params['conf']

    if var.DASHBOARD_PID.exists():
        check_service(name='dashboard', pidfile=var.DASHBOARD_PID)
    else:
        dashboard.dashboard_start(host=conf.dashboard.host,
                                  port=conf.dashboard.port,
                                  loglevel=conf.dashboard.loglevel,
                                  attach=False,
                                  no_daemon=False)

    if var.API_PID.exists():
        check_service(name='API', pidfile=var.API_PID)
    else:
        api.api_start(host=conf.api.host,
                      port=conf.api.port,
                      loglevel=conf.api.loglevel,
                      concurrency=conf.api.concurrency,
                      attach=False,
                      no_daemon=False)

    if var.WORKER_PID.exists():
        check_service(name='worker', pidfile=var.WORKER_PID)
    else:
        worker.worker_start(name=conf.worker.name,
                            pool=conf.worker.pool,
                            loglevel=conf.worker.loglevel,
                            concurrency=conf.worker.concurrency,
                            broker_url=conf.worker.broker_url,
                            backend_url=conf.worker.backend_url,
                            attach=False,
                            no_daemon=False)

    if (container := get_container(var.BROKER_ID)) and container.status == 'running':
        log.project_console.print(':rocket: The broker status: running', style='bright_blue')
    else:
        broker.broker_start(port=conf.broker.port,
                            ui_port=conf.broker.ui_port,
                            auto_remove=conf.broker.auto_remove,
                            attach=False)

    if (container := get_container(var.BACKEND_ID)) and container.status == 'running':
        log.project_console.print(':rocket: The backend status: running', style='bright_blue')
    else:
        backend.backend_start(port=conf.backend.port,
                              auto_remove=conf.backend.auto_remove,
                              attach=False)

    if attach_stream:
        try:
            attach(live=False)
        finally:
            down(prune=False, v=False)


@cli.command(help='Stop services')
def down(prune: bool = Option(False, '--prune', '-p', is_flag=True,
                              help='Prune all docker containers after exit.'),
         v: bool = Option(False, '--volume', '-v', is_flag=True,
                          help='Remove the volumes associated with the all docker containers and the log files.')
         ) -> None:
    """
    Stop services: Dashboard, API, Worker, Broker, Backend.

    Parameters
    ----------
    prune : bool, default=False
        Prune all docker containers after exit.
    v : bool, default=False
        Remove the volumes associated with the all docker containers and the log files.

    """

    from trecover.config import log
    from trecover.utils.docker import get_container
    from trecover.utils.cli import stop_service

    if var.DASHBOARD_PID.exists():
        stop_service(name='dashboard', pidfile=var.DASHBOARD_PID, logfile=log.DASHBOARD_LOG)

    if var.API_PID.exists():
        stop_service(name='API', pidfile=var.API_PID, logfile=log.API_LOG)

    if var.WORKER_PID.exists():
        stop_service(name='worker', pidfile=var.WORKER_PID, logfile=log.WORKER_LOG)

    if (container := get_container(var.BROKER_ID)) and container.status == 'running':
        container.stop()

        log.project_console.print('The broker service is stopped', style='bright_blue')

        if prune:
            broker.broker_prune(force=False, v=v)

    if (container := get_container(var.BACKEND_ID)) and container.status == 'running':
        container.stop()

        log.project_console.print('The backend service is stopped', style='bright_blue')

        if prune:
            backend.backend_prune(force=False, v=v)


@cli.command(name='status', help='Display services status')
def status() -> None:
    """ Display services status """

    from trecover.utils.cli import check_service
    from trecover.app.cli.broker import broker_status
    from trecover.app.cli.backend import backend_status

    check_service(name='dashboard', pidfile=var.DASHBOARD_PID)
    check_service(name='API', pidfile=var.API_PID)
    check_service(name='worker', pidfile=var.WORKER_PID)
    broker_status()
    backend_status()


@cli.command(name='attach', help='Attach local output stream to a running services')
def attach(live: bool = Option(False, '--live', '-l', is_flag=True,
                               help='Stream only fresh log records')
           ) -> None:
    """
    Attach local output stream to a running services.

    Parameters
    ----------
    live : bool, Default=False
        Stream only fresh log records

    """

    from trecover.config import log
    from trecover.utils.cli import stream

    with log.project_console.screen():
        for record in stream(('dashboard', log.DASHBOARD_LOG),
                             ('API', log.API_LOG),
                             ('worker', log.WORKER_LOG),
                             live=live):
            log.project_console.print(record)

    log.project_console.clear()


if __name__ == '__main__':
    cli()
