from typer import Typer, Option, Context, BadParameter

from config import var, log

cli = Typer(name='Worker-cli', add_completion=False)


@cli.callback(invoke_without_command=True)
def worker_state_verification(ctx: Context) -> None:
    if var.WORKER_PID.exists():
        if ctx.invoked_subcommand in ('start', None):
            log.project_console.print(':rocket: The worker service is already started', style='bright_blue')
            ctx.exit(0)

    elif ctx.invoked_subcommand is None:
        worker_start(name='ZReaderWorker', pool=var.PoolType.solo, loglevel=var.LogLevel.info,
                     concurrency=var.CELERY_WORKERS, broker_url=var.CELERY_BROKER,
                     backend_url=var.CELERY_BACKEND, attach=False)

    elif ctx.invoked_subcommand != 'start':
        log.project_console.print('The worker service is not started', style='yellow')
        ctx.exit(1)


@cli.command(name='start')
def worker_start(name: str = Option('ZReaderWorker', '--name', '-n', help='Set custom worker name.'),
                 pool: var.PoolType = Option(var.PoolType.solo, '--pool', '-p',
                                             help='Worker processes/threads pool type.'),
                 loglevel: var.LogLevel = Option(var.LogLevel.info, '--loglevel', '-l', help='Logging level.'),
                 concurrency: int = Option(var.CELERY_WORKERS, '-c', help='The number of worker processes/threads.'),
                 broker_url: str = Option(var.CELERY_BROKER, '--broker', help='Broker url.'),
                 backend_url: str = Option(var.CELERY_BACKEND, '--backend', help='Backend url.'),
                 attach: bool = Option(False, '--attach', '-a', is_flag=True, help='Attach output and error streams')
                 ) -> None:
    import platform
    from zreader.utils.cli import start_service

    if platform.system() == 'Windows' and pool != var.PoolType.solo:
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

    start_service(argv, name='worker', logfile=log.WORKER_LOG, pidfile=var.WORKER_PID)

    if attach:
        worker_attach(live=False)


@cli.command(name='stop')
def worker_stop() -> None:
    from zreader.utils.cli import stop_service

    stop_service(name='worker', pidfile=var.WORKER_PID)


@cli.command(name='status')
def worker_status() -> None:
    from zreader.utils.cli import check_service

    check_service(name='worker', pidfile=var.WORKER_PID)


@cli.command(name='attach')
def worker_attach(live: bool = Option(False, '--live', '-l', is_flag=True,
                                      help='Stream only fresh log records')
                  ) -> None:
    from zreader.utils.cli import stream

    with log.project_console.screen():
        for record in stream(('worker', log.WORKER_LOG), live=live):
            log.project_console.print(record)

    log.project_console.clear()


if __name__ == '__main__':
    try:
        cli()
    except Exception as e:
        log.project_logger.error(e)
        log.project_console.print_exception(show_locals=True)
        log.error_console.print_exception(show_locals=True)
