from typer import Typer, Context, Option

from config import log

cli = Typer(name='Train-cli', add_completion=False)


@cli.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, add_help_option=False)
def local(ctx: Context, show_help: bool = Option(False, '--help', '-h', is_flag=True,
                                                 help='Show help message and exit.')):
    from zreader.train.local import train, get_parser

    if show_help:
        get_parser().print_help()
    else:
        try:
            train(args=ctx.args)
        except Exception as ee:
            log.project_logger.error(ee)
            log.project_console.print_exception()
            log.error_console.print_exception(show_locals=True)


@cli.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, add_help_option=False)
def remote(ctx: Context, show_help: bool = Option(False, '--help', '-h', is_flag=True,
                                                  help='Show help message and exit.')):
    pass


if __name__ == '__main__':
    cli()
