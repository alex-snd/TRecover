from typer import Typer, Option, Context, BadParameter

from trecover.config import var

cli = Typer(name='Worker-cli', add_completion=False, help='Manage Worker service')


@cli.callback(invoke_without_command=True)
def worker_state_verification(ctx: Context) -> None:
    """
    Perform cli commands verification (state checking).

    Parameters
    ----------
    ctx : Context
        Typer (Click like) special internal object that holds state relevant
        for the script execution at every single level.

   """

    from trecover.config import log

    if var.WORKER_PID.exists():
        if ctx.invoked_subcommand in ('start', None):
            log.project_console.print(':rocket: The worker service is already started', style='bright_blue')
            ctx.exit(0)

    elif ctx.invoked_subcommand is None:
        worker_start(name='TRecoverWorker', pool=var.PoolType.solo, loglevel=var.LogLevel.info,
                     concurrency=var.CELERY_WORKERS, broker_url=var.CELERY_BROKER,
                     backend_url=var.CELERY_BACKEND, attach=False, no_daemon=False)

    elif ctx.invoked_subcommand != 'start':
        log.project_console.print('The worker service is not started', style='yellow')
        ctx.exit(1)


@cli.command(name='start', help='Start service')
def worker_start(name: str = Option('TRecoverWorker', '--name', '-n', help='Set custom worker hostname.'),
                 pool: var.PoolType = Option(var.PoolType.solo, '--pool', '-p',
                                             help='Worker processes/threads pool type.'),
                 loglevel: var.LogLevel = Option(var.LogLevel.info, '--loglevel', '-l', help='Logging level.'),
                 concurrency: int = Option(var.CELERY_WORKERS, '-c', help='The number of worker processes.'),
                 broker_url: str = Option(var.CELERY_BROKER, '--broker', help='Broker url.'),
                 backend_url: str = Option(var.CELERY_BACKEND, '--backend', help='Backend url.'),
                 attach: bool = Option(False, '--attach', '-a', is_flag=True, help='Attach output and error streams'),
                 no_daemon: bool = Option(False, '--no-daemon', is_flag=True, help='Do not run as a daemon process')
                 ) -> None:
    """
    Start API service.

    Parameters
    ----------
    name : str, default='TRecoverWorker'
        Custom worker hostname.
    pool : str, {'prefork', 'eventlet', 'gevent', 'processes', 'solo'}, default='solo'
        Worker processes/threads pool type.
    loglevel : {'debug', 'info', 'warning', 'error', 'critical'}, default='info'
        Level of logging.
    concurrency : int, default=ENV(CELERY_WORKERS) or 1
        The number of worker processes.
    broker_url : str, default=ENV(CELERY_BROKER) or 'pyamqp://guest@localhost'
        Broker url.
    backend_url : str, default=ENV(CELERY_BACKEND) or 'redis://localhost'
        Backend url.
    attach : bool, default=False
        Attach output and error streams.
    no_daemon : bool, default=False
        Do not run as a daemon process.

    Raises
    ------
    typer.BadParameter:
        If non-solo pool type is selected for windows platform.

    """

    import platform
    from subprocess import run

    from trecover.config import log
    from trecover.utils.cli import start_service

    if platform.system() == 'Windows' and pool != var.PoolType.solo:
        raise BadParameter("Windows platform only supports 'solo' pool")

    argv = [
        'celery',
        '--broker', broker_url,
        '--result-backend', backend_url,
        '--app', 'trecover.app.api.backend.celeryapp', 'worker',
        '--hostname', name,
        '--concurrency', str(concurrency),
        '--pool', pool,
        '--loglevel', loglevel
    ]

    if no_daemon:
        run(argv)
    else:
        start_service(argv, name='worker', logfile=log.WORKER_LOG, pidfile=var.WORKER_PID)

        if attach:
            worker_attach(live=False)


@cli.command(name='stop', help='Stop service')
def worker_stop() -> None:
    """ Stop worker service. """

    from trecover.config import log
    from trecover.utils.cli import stop_service

    stop_service(name='worker', pidfile=var.WORKER_PID, logfile=log.WORKER_LOG)


@cli.command(name='status', help='Display service status')
def worker_status() -> None:
    """ Display worker service status. """

    from trecover.utils.cli import check_service

    check_service(name='worker', pidfile=var.WORKER_PID)


@cli.command(name='attach', help='Attach local output stream to a service')
def worker_attach(live: bool = Option(False, '--live', '-l', is_flag=True,
                                      help='Stream only fresh log records')
                  ) -> None:
    """
    Attach local output stream to a running worker service.

    Parameters
    ----------
    live : bool, Default=False
        Stream only fresh log records.

    """

    from trecover.config import log
    from trecover.utils.cli import stream

    with log.project_console.screen():
        for record in stream(('worker', log.WORKER_LOG), live=live):
            log.project_console.print(record)

    log.project_console.clear()


if __name__ == '__main__':
    cli()
