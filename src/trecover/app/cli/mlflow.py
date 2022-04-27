from typing import Optional

from typer import Typer, Option, Context, BadParameter

from trecover.config import var

cli = Typer(name='Mlflow-cli', add_completion=False, help='Manage Mlflow service')


@cli.callback(invoke_without_command=True)
def mlflow_state_verification(ctx: Context) -> None:
    """
    Perform cli commands verification (state checking).

    Parameters
    ----------
    ctx : Context
        Typer (Click like) special internal object that holds state relevant
        for the script execution at every single level.

   """

    from trecover.config import log, exp_var

    if var.MLFLOW_PID.exists():
        if ctx.invoked_subcommand in ('start', None):
            log.project_console.print(':rocket: The mlflow service is already started', style='bright_blue')
            ctx.exit(0)

    elif ctx.invoked_subcommand is None:
        mlflow_start(host=var.MLFLOW_HOST, port=var.MLFLOW_PORT, concurrency=var.MLFLOW_WORKERS,
                     registry=exp_var.MLFLOW_REGISTRY_DIR.as_uri(), backend_uri=exp_var.MLFLOW_BACKEND,
                     only_ui=False, attach=False)

    elif ctx.invoked_subcommand != 'start':
        log.project_console.print('The mlflow service is not started', style='yellow')
        ctx.exit(1)


@cli.command(name='start', help='Start service')
def mlflow_start(host: str = Option(var.MLFLOW_HOST, '--host', '-h', help='Bind socket to this host.'),
                 port: int = Option(var.MLFLOW_PORT, '--port', '-p', help='Bind socket to this port.'),
                 concurrency: int = Option(var.MLFLOW_WORKERS, '-c',
                                           help='The number of mlflow server workers.'),
                 registry: Optional[str] = Option(None, '--registry', '-r',
                                                  help='Path to local directory to store artifacts.'),
                 backend_uri: Optional[str] = Option(None, '--backend', help='Backend uri.'),
                 only_ui: bool = Option(False, '--only-ui', is_flag=True, help='Launch only the Mlflow tracking UI'),
                 attach: bool = Option(False, '--attach', '-a', is_flag=True, help='Attach output and error streams')
                 ) -> None:
    """
    Start dashboard service.

    Parameters
    ----------
    host : str, default=ENV(MLFLOW_HOST) or 'localhost'
        The address where the server will listen for client and browser connections.
        Use this if you want to bind the server to a specific address. If set, the server
        will only be accessible from this address, and not from any aliases (like localhost).
    port : int, default=ENV(MLFLOW_PORT) or 8002
        The port where the server will listen for browser connections.
    concurrency : int, default=ENV(MLFLOW_WORKERS) or 1
        The number of mlflow server workers.
    registry : str, default=ENV(MLFLOW_REGISTRY_DIR) or 'file:///<BASE_DIR>/experiments/mlflow_registry'
        URI to which to persist experiment and run data. Acceptable URIs are SQLAlchemy-compatible
        database connection strings (e.g. 'sqlite:///path/to/file.db') or
        local filesystem URIs (e.g. 'file:///absolute/path/to/directory').
        By default, data will be logged to the ./mlruns directory.
    backend_uri : str, default=ENV(MLFLOW_BACKEND) or 'sqlite:///<BASE_DIR>/experiments/mlflow_registry/mlflow.db'
        Local or S3 URI to store artifacts, for new experiments. Note that this
        flag does not impact already-created experiments.
    only_ui : bool, default=False
        Launch only the Mlflow tracking UI.
    attach : bool, default=False
        Attach output and error streams.

    Raises
    ------
    typer.BadParameter:
        If concurrency option is not equal to one for windows platform.

    """

    import platform

    from trecover.config import log, exp_var
    from trecover.utils.cli import start_service

    if (is_windows := platform.system() == 'Windows') and concurrency != 1:
        raise BadParameter("Windows platform does not support concurrency option")

    command = 'ui' if only_ui else 'server'
    registry = registry or str(exp_var.MLFLOW_REGISTRY_DIR.as_uri())
    backend_uri = backend_uri or str(exp_var.MLFLOW_BACKEND)

    argv = [
        'mlflow', command,
        '--host', host,
        '--port', str(port),
        '--default-artifact-root', backend_uri,
        '--backend-store-uri', registry
    ]

    if not only_ui and not is_windows:
        argv.extend(['--workers', str(concurrency)])

    start_service(argv, name='mlflow', logfile=log.MLFLOW_LOG, pidfile=var.MLFLOW_PID)

    if attach:
        mlflow_attach(live=False)


@cli.command(name='stop', help='Stop service')
def mlflow_stop() -> None:
    """ Stop dashboard service. """

    from trecover.config import log
    from trecover.utils.cli import stop_service

    stop_service(name='mlflow', pidfile=var.MLFLOW_PID, logfile=log.MLFLOW_LOG)


@cli.command(name='status', help='Display service status')
def mlflow_status() -> None:
    """ Display dashboard service status. """

    from trecover.utils.cli import check_service

    check_service(name='mlflow', pidfile=var.MLFLOW_PID)


@cli.command(name='attach', help='Attach local output stream to a service')
def mlflow_attach(live: bool = Option(False, '--live', '-l', is_flag=True,
                                      help='Stream only fresh log records')
                  ) -> None:
    """
    Attach local output stream to a running dashboard service.

    Parameters
    ----------
    live : bool, Default=False
        Stream only fresh log records.

    """

    from trecover.config import log
    from trecover.utils.cli import stream

    with log.project_console.screen():
        for record in stream(('mlflow', log.MLFLOW_LOG), live=live):
            log.project_console.print(record)

    log.project_console.clear()


if __name__ == '__main__':
    cli()
