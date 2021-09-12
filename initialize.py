import sysconfig
from pathlib import Path

from typer import Typer, Option

import config

cli = Typer(name='Zreader-initialization')


@cli.command()
def initialize(deactivate: bool = Option(False, '-d', help='Deactivate flag')) -> None:
    pth_file = Path(sysconfig.get_paths()['purelib'], 'zreader.pth')

    if deactivate:
        if pth_file.exists():
            pth_file.unlink()
            config.project_console.print('ZReader project was deactivated', style='bright_blue')

        else:
            config.project_console.print('ZReader project cannot be deactivated', style='bright_red')

    else:
        with pth_file.open(mode='w') as f:
            f.write(str(Path(__file__).parent.absolute()))

        config.project_console.print('ZReader project was initialized', style='bright_blue')


if __name__ == '__main__':
    try:
        cli()
    except Exception as e:
        config.project_logger.error(e)
        config.project_console.print_exception(show_locals=True)
        config.error_console.print_exception(show_locals=True)
