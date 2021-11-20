from typer import Typer, Option, Context, BadParameter

from config import var, log

cli = Typer(name='Mlflow-cli', add_completion=False)


@cli.callback(invoke_without_command=True)
def mlflow_state_verification(ctx: Context) -> None:
    if var.MLFLOW_PID.exists():
        if ctx.invoked_subcommand in ('start', None):
            log.project_console.print(':rocket: The mlflow service is already started', style='bright_blue')
            ctx.exit(0)

    elif ctx.invoked_subcommand is None:
        mlflow_start(host=var.MLFLOW_HOST, port=var.MLFLOW_PORT, concurrency=var.MLFLOW_WORKERS,
                     registry=var.MLFLOW_REGISTRY_DIR, backend_uri=var.MLFLOW_BACKEND, only_ui=False, attach=False)

    elif ctx.invoked_subcommand != 'start':
        log.project_console.print('The mlflow service is not started', style='yellow')
        ctx.exit(1)


@cli.command(name='start')
def mlflow_start(host: str = Option(var.MLFLOW_HOST, '--host', '-h', help='Bind socket to this host.'),
                 port: int = Option(var.MLFLOW_PORT, '--port', '-p', help='Bind socket to this port.'),
                 concurrency: int = Option(var.MLFLOW_WORKERS, '-c',
                                           help='The number of mlflow server workers.'),
                 registry: str = Option(var.MLFLOW_REGISTRY_DIR, '--registry', '-r',
                                        help='Path to local directory to store artifacts.'),
                 backend_uri: str = Option(var.MLFLOW_BACKEND, '--backend', help='Backend uri.'),
                 only_ui: bool = Option(False, '--only-ui', is_flag=True, help='Launch only the MLflow tracking UI'),
                 attach: bool = Option(False, '--attach', '-a', is_flag=True, help='Attach output and error streams')
                 ) -> None:
    import platform
    from subprocess import Popen, CREATE_NO_WINDOW, STDOUT

    if (is_windows := platform.system() == 'Windows') and concurrency != 1:
        raise BadParameter("Windows platform does not support concurrency option")

    command = 'ui' if only_ui else 'server'

    argv = [
        'mlflow', command,
        '--host', host,
        '--port', str(port),
        '--default-artifact-root', str(registry),
        '--backend-store-uri', str(backend_uri)
    ]

    if not only_ui and not is_windows:
        argv.extend(['--workers', str(concurrency)])

    process = Popen(argv, creationflags=CREATE_NO_WINDOW, stdout=log.MLFLOW_LOG.open(mode='w'), stderr=STDOUT,
                    universal_newlines=True)

    with var.MLFLOW_PID.open('w') as f:
        f.write(str(process.pid))

    log.project_console.print('The mlflow service is started', style='bright_blue')

    if attach:
        mlflow_attach(live=False)


@cli.command(name='stop')
def mlflow_stop() -> None:
    from zreader.utils.cli import stop_service

    stop_service(name='mlflow', pidfile=var.MLFLOW_PID)


@cli.command(name='status')
def mlflow_status() -> None:
    from zreader.utils.cli import check_service

    check_service(name='mlflow', pidfile=var.MLFLOW_PID)


@cli.command(name='attach')
def mlflow_attach(live: bool = Option(False, '--live', '-l', is_flag=True,
                                      help='Stream only fresh log records')
                  ) -> None:
    from zreader.utils.cli import stream

    with log.project_console.screen():
        for record in stream(logfile=log.MLFLOW_LOG, live=live):
            log.project_console.print(record.strip())


if __name__ == '__main__':
    try:
        cli()
    except Exception as e:
        log.project_logger.error(e)
        log.project_console.print_exception(show_locals=True)
        log.error_console.print_exception(show_locals=True)
