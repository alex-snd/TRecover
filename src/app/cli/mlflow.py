from typer import Typer

from config import log

cli = Typer(name='Train-cli', add_completion=False)

if __name__ == '__main__':
    try:
        cli()
    except Exception as e:
        log.project_logger.error(e)
        log.project_console.print_exception(show_locals=True)
        log.error_console.print_exception(show_locals=True)
