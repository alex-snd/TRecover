from typer import Typer, Context, Option

cli = Typer(name='Train-cli', add_completion=False, help='Manage training')


@cli.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
             add_help_option=False,
             help='Start local training')
def local(ctx: Context,
          show_help: bool = Option(False, '--help', '-h', is_flag=True, help='Show help message and exit.')) -> None:
    """
    Start local training.

    Parameters
    ----------
    ctx : Context
        Typer (Click like) special internal object that holds state relevant
        for the script execution at every single level.
    show_help : bool, default=False
        Show local train options.

    """

    from trecover.config import log
    from trecover.train.local import train, get_local_parser

    if show_help:
        get_local_parser().print_help()
    else:
        try:
            train(cli_args=ctx.args)
        except Exception as e:
            log.project_logger.error(e)
            log.project_console.print_exception()
            log.error_console.print_exception(show_locals=True)


@cli.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
             add_help_option=False,
             help='Start collaborative training')
def collab(ctx: Context,
           show_help: bool = Option(False, '--help', '-h', is_flag=True, help='Show help message and exit.')
           ) -> None:
    """
    Start collaborative training.

    Parameters
    ----------
    ctx : Context
        Typer (Click like) special internal object that holds state relevant
        for the script execution at every single level.
    show_help : bool, default=False
        Show remote train options.

    """

    from trecover.train.collab import train
    from trecover.train.collab.arguments import get_train_parser

    if show_help:
        get_train_parser().print_help()
    else:
        train(cli_args=ctx.args)


@cli.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
             add_help_option=False,
             help='Start collaborative training monitor')
def collab_monitor(ctx: Context,
                   show_help: bool = Option(False, '--help', '-h', is_flag=True, help='Show help message and exit.')
                   ) -> None:
    """
    Start collaborative training monitor.

    Parameters
    ----------
    ctx : Context
        Typer (Click like) special internal object that holds state relevant
        for the script execution at every single level.
    show_help : bool, default=False
        Show remote train options.

    """
    from trecover.train.collab import monitor
    from trecover.train.collab.arguments import get_monitor_parser

    if show_help:
        get_monitor_parser().print_help()
    else:
        monitor(cli_args=ctx.args)


if __name__ == '__main__':
    cli()
