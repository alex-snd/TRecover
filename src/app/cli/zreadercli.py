from pathlib import Path

from typer import Typer, Argument, Option, Context

from app.cli import download, train, mlflow, dashboard, api, worker, broker, backend
from config import var, log

# TODO help info
cli = Typer(name='Zreader-cli', add_completion=False)

cli.add_typer(download.cli, name='download')
cli.add_typer(train.cli, name='train')
cli.add_typer(mlflow.cli, name='mlflow')
cli.add_typer(dashboard.cli, name='dashboard')
cli.add_typer(api.cli, name='api')
cli.add_typer(worker.cli, name='worker')
cli.add_typer(broker.cli, name='broker')
cli.add_typer(backend.cli, name='backend')


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
    from zreader.utils.transform import files_columns_to_tensors
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
    if ctx.invoked_subcommand is None:
        log.project_console.print(ctx.get_help(), markup=False)
        ctx.exit(0)

    if ctx.invoked_subcommand in ('up', 'down'):
        from zreader.utils.docker import is_docker_running
        from zreader.utils.cli import parse_config

        if not is_docker_running():
            log.project_console.print('Docker engine is not running', style='red')
            ctx.exit(1)

        if not (config_file := Path(file)).exists():
            log.project_console.print('Defined Zreader configuration file does not exist', style='red')
            ctx.exit(1)

        ctx.params['conf'] = parse_config(config_file)


@cli.command()
def up(ctx: Context) -> None:
    from zreader.utils.docker import get_container
    from zreader.utils.cli import check_service

    conf = ctx.parent.params['conf']

    if var.DASHBOARD_PID.exists():
        check_service(name='dashboard', pidfile=var.DASHBOARD_PID)
    else:
        dashboard.dashboard_start(host=conf.dashboard.host,
                                  port=conf.dashboard.port,
                                  loglevel=conf.dashboard.loglevel,
                                  attach=False)

    if var.API_PID.exists():
        check_service(name='API', pidfile=var.API_PID)
    else:
        api.api_start(host=conf.api.host,
                      port=conf.api.port,
                      loglevel=conf.api.loglevel,
                      concurrency=conf.api.concurrency,
                      attach=False)

    if var.WORKER_PID.exists():
        check_service(name='worker', pidfile=var.WORKER_PID)
    else:
        worker.worker_start(name=conf.worker.name,
                            pool=conf.worker.pool,
                            loglevel=conf.worker.loglevel,
                            concurrency=conf.worker.concurrency,
                            broker_url=conf.worker.broker_url,
                            backend_url=conf.worker.backend_url,
                            attach=False)

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


@cli.command()
def down(prune: bool = Option(False, '--prune', '-p', is_flag=True,
                              help='Prune broker.'),
         v: bool = Option(False, '--volume', '-v', is_flag=True,
                          help='Remove the volumes associated with the container')
         ) -> None:
    from zreader.utils.docker import get_container
    from zreader.utils.cli import stop_service

    if var.DASHBOARD_PID.exists():
        stop_service(name='dashboard', pidfile=var.DASHBOARD_PID)

    if var.API_PID.exists():
        stop_service(name='API', pidfile=var.API_PID)

    if var.WORKER_PID.exists():
        stop_service(name='worker', pidfile=var.WORKER_PID)

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


@cli.command(name='status')
def status() -> None:
    from zreader.utils.cli import check_service
    from app.cli.broker import broker_status
    from app.cli.backend import backend_status

    check_service(name='dashboard', pidfile=var.DASHBOARD_PID)
    check_service(name='API', pidfile=var.API_PID)
    check_service(name='worker', pidfile=var.WORKER_PID)
    broker_status()
    backend_status()


if __name__ == '__main__':
    try:
        cli()
    except Exception as e:
        log.project_logger.error(e)
        log.project_console.print_exception(show_locals=True)
        log.error_console.print_exception(show_locals=True)
