from typer import Typer, Context, Option

from config import log

cli = Typer(name='Train-cli', add_completion=False)


@cli.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, add_help_option=False)
def local(ctx: Context, show_help: bool = Option(False, '--help', '-h', is_flag=True,
                                                 help='Show help message and exit.')):
    from zreader.train.base import get_parser, get_experiment_params, train

    if show_help:
        get_parser().print_help()
        ctx.exit(0)

    train(params=get_experiment_params(ctx.args))


@cli.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, add_help_option=False)
def wandb(ctx: Context, show_help: bool = Option(False, '--help', '-h', is_flag=True,
                                                 help='Show help message and exit.')):
    from zreader.train.wandb_train import get_parser, get_experiment_params, train

    if show_help:
        get_parser().print_help()
        ctx.exit(0)

    train(params=get_experiment_params(ctx.args))


if __name__ == '__main__':
    try:
        cli()
    except Exception as e:
        log.project_logger.error(e)
        log.project_console.print_exception(show_locals=True)
        log.error_console.print_exception(show_locals=True)
