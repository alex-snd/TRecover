from typer import Typer, Context, Option

cli = Typer(name='Collab-cli', add_completion=True, help='Manage collaborative training')


@cli.command(context_settings={'allow_extra_args': True, 'ignore_unknown_options': True},
             add_help_option=False,
             help='Start collaborative training monitor')
def monitor(ctx: Context,
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


@cli.command(context_settings={'allow_extra_args': True, 'ignore_unknown_options': True},
             add_help_option=False,
             help='Visualize collaborative training progress')
def visualize(ctx: Context,
              show_help: bool = Option(False, '--help', '-h', is_flag=True, help='Show help message and exit.')
              ) -> None:
    """
    Visualize collaborative training progress.

    Parameters
    ----------
    ctx : Context
        Typer (Click like) special internal object that holds state relevant
        for the script execution at every single level.
    show_help : bool, default=False
        Show remote train options.

    """

    from trecover.train.collab import visualize
    from trecover.train.collab.arguments import get_visualization_parser

    if show_help:
        get_visualization_parser().print_help()
    else:
        visualize(cli_args=ctx.args)


@cli.command(context_settings={'allow_extra_args': True, 'ignore_unknown_options': True},
             add_help_option=False,
             help='Tune batch size for this machine')
def tune(ctx: Context,
         show_help: bool = Option(False, '--help', '-h', is_flag=True, help='Show help message and exit.')
         ) -> None:
    """
    Tune batch size for this machine.

    Parameters
    ----------
    ctx : Context
        Typer (Click like) special internal object that holds state relevant
        for the script execution at every single level.
    show_help : bool, default=False
        Show tune options.

    """

    from trecover.train.collab import tune
    from trecover.train.collab.arguments import get_train_parser

    if show_help:
        get_train_parser().print_help()
    else:
        tune(cli_args=ctx.args)


@cli.command(context_settings={'allow_extra_args': True, 'ignore_unknown_options': True},
             add_help_option=False,
             help='Start collaborative training')
def train(ctx: Context,
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
        Show collab train options.

    """

    from trecover.train.collab import train
    from trecover.train.collab.arguments import get_train_parser

    if show_help:
        get_train_parser().print_help()
    else:
        train(cli_args=ctx.args)


@cli.command(context_settings={'allow_extra_args': True, 'ignore_unknown_options': True},
             add_help_option=False,
             help='Start auxiliary peer (for cpu-only workers)')
def aux(ctx: Context,
        show_help: bool = Option(False, '--help', '-h', is_flag=True, help='Show help message and exit.')
        ) -> None:
    """
    Start auxiliary peers for gradient averaging (for cpu-only workers).

    Parameters
    ----------
    ctx : Context
        Typer (Click like) special internal object that holds state relevant
        for the script execution at every single level.
    show_help : bool, default=False
        Show collab train options.

    """

    from trecover.train.collab import auxiliary
    from trecover.train.collab.arguments import get_train_parser

    if show_help:
        get_train_parser().print_help()
    else:
        auxiliary(cli_args=ctx.args)


if __name__ == '__main__':
    cli()
