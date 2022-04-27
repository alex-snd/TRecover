from pathlib import Path

from typer import Typer, Option

from trecover.config import var, log

cli = Typer(name='Download-cli', add_completion=False, help='Download train data or pre-trained model')


@cli.command(name='data', help='Download train data')
def download_data(link: str = Option(var.TRAIN_DATA_URL, help='Link to the train data on Yandex disk or GitHub'),
                  save_dir: Path = Option(var.DATA_DIR, help='Path where to store downloaded data'),
                  yandex_disk: bool = Option(False, is_flag=True, help='If the link is to Yandex disk')
                  ) -> None:
    """
    Download train data from Yandex disk or GitHub.

    Parameters
    ----------
    link : str, default=var.TRAIN_DATA_URL
        Sharing link to the train data on Yandex disk or GitHub.
    save_dir : Path, default=var.DATA_DIR
        Path where to store downloaded data.
    yandex_disk : bool, default=False
        If the link is to Yandex disk.

    """

    from trecover.utils.cli import download_archive

    download_archive(link=link, save_dir=save_dir, yandex_disk=yandex_disk)


@cli.command(name='artifacts', help='Download model artifacts by specified version or archive_link')
def download_artifacts(version: str = Option('latest', help="Artifacts' version"),
                       archive_link: str = Option(None, help='Link to the artifacts archive on Yandex disk or GitHub'),
                       save_dir: Path = Option(var.INFERENCE_DIR, help='Path where to save downloaded artifacts'),
                       yandex_disk: bool = Option(False, is_flag=True, help='If the archive_link is to Yandex disk'),
                       show: bool = Option(False, is_flag=True, help="Print available artifacts' versions")
                       ) -> None:
    """
    Download model artifacts by specified version or archive_link to Yandex disk or GitHub.

    Parameters
    ----------
    version : str, default='latest'
        Artifacts' version.
    archive_link : str, default=None
        Sharing link to the model artifacts archive on Yandex disk or GitHub.
    save_dir : Path, default=var.INFERENCE_DIR
        Path where to save downloaded artifacts.
    yandex_disk : bool, default=False
        If the link is to Yandex disk.
    show : bool, default=False
        Print available artifacts' versions.

    """

    from rich.prompt import Confirm
    from trecover.utils.cli import download_archive, download_from_github

    if show:
        log.project_console.print(var.CHECKPOINT_URLS.keys())

    elif archive_link:
        download_archive(link=archive_link, save_dir=save_dir, yandex_disk=yandex_disk)

    elif version in var.CHECKPOINT_URLS:
        download_from_github(direct_link=var.CHECKPOINT_URLS[version]['model'], save_dir=save_dir)
        download_from_github(direct_link=var.CHECKPOINT_URLS[version]['config'], save_dir=save_dir)

    elif Confirm.ask(prompt='[bright_blue]Specified version was not found. Continue downloading the latest version?',
                     default=True,
                     console=log.project_console):
        download_from_github(direct_link=var.CHECKPOINT_URLS['latest']['model'], save_dir=save_dir)
        download_from_github(direct_link=var.CHECKPOINT_URLS['latest']['config'], save_dir=save_dir)


if __name__ == '__main__':
    cli()
