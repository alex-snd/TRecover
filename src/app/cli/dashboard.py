from typer import Typer, Option, Context

from config import var, log

cli = Typer(name='Dashboard-cli', add_completion=False)


@cli.callback(invoke_without_command=True)
def dashboard_state_verification(ctx: Context) -> None:
    if var.DASHBOARD_PID.exists():
        if ctx.invoked_subcommand in ('start', None):
            log.project_console.print(':rocket: The dashboard service is already started', style='bright_blue')
            ctx.exit(0)

    elif ctx.invoked_subcommand is None:
        dashboard_start(host=var.STREAMLIT_HOST, port=var.STREAMLIT_PORT, loglevel=var.LogLevel.info, attach=False)

    elif ctx.invoked_subcommand != 'start':
        log.project_console.print('The dashboard service is not started', style='yellow')
        ctx.exit(1)


@cli.command(name='start')
def dashboard_start(host: str = Option(var.STREAMLIT_HOST, '--host', '-h', help='Bind socket to this host.'),
                    port: int = Option(var.STREAMLIT_PORT, '--port', '-p', help='Bind socket to this port.'),
                    loglevel: var.LogLevel = Option(var.LogLevel.info, '--loglevel', '-l', help='Logging level.'),
                    attach: bool = Option(False, '--attach', '-a', is_flag=True,
                                          help='Attach output and error streams')
                    ) -> None:
    from app import dashboard
    from zreader.utils.cli import start_service

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

    start_service(argv, name='dashboard', logfile=log.DASHBOARD_LOG, pidfile=var.DASHBOARD_PID)

    if attach:
        dashboard_attach(live=False)


@cli.command(name='stop')
def dashboard_stop() -> None:
    from zreader.utils.cli import stop_service

    stop_service(name='dashboard', pidfile=var.DASHBOARD_PID)


@cli.command(name='status')
def dashboard_status() -> None:
    from zreader.utils.cli import check_service

    check_service(name='dashboard', pidfile=var.DASHBOARD_PID)


@cli.command(name='attach')
def dashboard_attach(live: bool = Option(False, '--live', '-l', is_flag=True,
                                         help='Stream only fresh log records')
                     ) -> None:
    from zreader.utils.cli import stream

    with log.project_console.screen():
        for record in stream(('dashboard', log.DASHBOARD_LOG), live=live):
            log.project_console.print(record)

    log.project_console.clear()


if __name__ == '__main__':
    try:
        cli()
    except Exception as e:
        log.project_logger.error(e)
        log.project_console.print_exception(show_locals=True)
        log.error_console.print_exception(show_locals=True)
