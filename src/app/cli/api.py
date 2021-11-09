from typer import Typer, Option, Context

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

    elif ctx.invoked_subcommand != 'start':
        log.project_console.print('The API service is not started', style='yellow')
        ctx.exit(1)


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
