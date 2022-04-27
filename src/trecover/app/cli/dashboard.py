from typer import Typer, Option, Context

from trecover.config import var, log

cli = Typer(name='Dashboard-cli', add_completion=False, help='Manage Dashboard service')


@cli.callback(invoke_without_command=True)
def dashboard_state_verification(ctx: Context) -> None:
    """
    Perform cli commands verification (state checking).

    Parameters
    ----------
    ctx : Context
        Typer (Click like) special internal object that holds state relevant
        for the script execution at every single level.

    """

    from trecover.config import log

    if var.DASHBOARD_PID.exists():
        if ctx.invoked_subcommand in ('start', None):
            log.project_console.print(':rocket: The dashboard service is already started', style='bright_blue')
            ctx.exit(0)

    elif ctx.invoked_subcommand is None:
        dashboard_start(host=var.STREAMLIT_HOST, port=var.STREAMLIT_PORT, loglevel=var.LogLevel.info, attach=False,
                        no_daemon=False)

    elif ctx.invoked_subcommand != 'start':
        log.project_console.print('The dashboard service is not started', style='yellow')
        ctx.exit(1)


@cli.command(name='start', help='Start service')
def dashboard_start(host: str = Option(var.STREAMLIT_HOST, '--host', '-h', help='Bind socket to this host.'),
                    port: int = Option(var.STREAMLIT_PORT, '--port', '-p', help='Bind socket to this port.'),
                    loglevel: var.LogLevel = Option(var.LogLevel.info, '--loglevel', '-l', help='Logging level.'),
                    attach: bool = Option(False, '--attach', '-a', is_flag=True,
                                          help='Attach output and error streams'),
                    no_daemon: bool = Option(False, '--no-daemon', is_flag=True, help='Do not run as a daemon process')
                    ) -> None:
    """
    Start dashboard service.

    Parameters
    ----------
    host : str, default=ENV(STREAMLIT_HOST) or 'localhost'
        The address where the server will listen for client and browser connections.
        Use this if you want to bind the server to a specific address. If set, the server
        will only be accessible from this address, and not from any aliases (like localhost).
    port : int, default=ENV(STREAMLIT_PORT) or 8000
        The port where the server will listen for browser connections.
    loglevel : {'debug', 'info', 'warning', 'error', 'critical'}, default='info'
        Level of logging.
    attach : bool, default=False
        Attach output and error streams.
    no_daemon : bool, default=False
        Do not run as a daemon process.

    """

    from subprocess import run

    from trecover.config import log
    from trecover.app import dashboard
    from trecover.utils.cli import start_service

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

    if no_daemon:
        run(argv)
    else:
        start_service(argv, name='dashboard', logfile=log.DASHBOARD_LOG, pidfile=var.DASHBOARD_PID)

        if attach:
            dashboard_attach(live=False)


@cli.command(name='stop', help='Stop service')
def dashboard_stop() -> None:
    """ Stop dashboard service. """

    from trecover.utils.cli import stop_service

    stop_service(name='dashboard', pidfile=var.DASHBOARD_PID, logfile=log.DASHBOARD_LOG)


@cli.command(name='status', help='Display service status')
def dashboard_status() -> None:
    """ Display dashboard service status. """

    from trecover.utils.cli import check_service

    check_service(name='dashboard', pidfile=var.DASHBOARD_PID)


@cli.command(name='attach', help='Attach local output stream to a service')
def dashboard_attach(live: bool = Option(False, '--live', '-l', is_flag=True,
                                         help='Stream only fresh log records')
                     ) -> None:
    """
    Attach local output stream to a running dashboard service.

    Parameters
    ----------
    live : bool, Default=False
        Stream only fresh log records

    """

    from trecover.config import log
    from trecover.utils.cli import stream

    with log.project_console.screen():
        for record in stream(('dashboard', log.DASHBOARD_LOG), live=live):
            log.project_console.print(record)

    log.project_console.clear()


if __name__ == '__main__':
    cli()
