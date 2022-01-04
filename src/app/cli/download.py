from typer import Typer, Argument, Option

from config import log

cli = Typer(name='Download-cli', add_completion=False)


@cli.command(name='data', help='Download train data from Yandex disk')
def download_data(sharing_link: str = Argument(..., help='Sharing link to the train data on Yandex disk'),
                  save_dir: str = Option('./', help='Path where to store downloaded data')
                  ) -> None:
    """
        Download train data from Yandex disk

        Parameters
        -----
        sharing_link: str
            Sharing link to the train data on Yandex disk

        save_dir: str
            Path where to store downloaded data

    """

    from zreader.utils.cli import download_archive_from_disk

    download_archive_from_disk(sharing_link, save_dir)


@cli.command(name='artifacts', help='Download model artifacts from Yandex disk')
def download_artifacts(sharing_link: str = Argument(..., help='Sharing link to the model artifacts on Yandex disk'),
                       save_dir: str = Option('./', help='Path where to save downloaded artifacts')
                       ) -> None:
    """
        Download model artifacts from Yandex disk

        Parameters
        -----
        sharing_link: str
            Sharing link to the model artifacts on Yandex disk

        save_dir: str
            Path where to save downloaded artifacts

    """

    from zreader.utils.cli import download_archive_from_disk

    download_archive_from_disk(sharing_link, save_dir)


if __name__ == '__main__':
    try:
        cli()
    except Exception as e:
        log.project_logger.error(e)
        log.project_console.print_exception(show_locals=True)
        log.error_console.print_exception(show_locals=True)
