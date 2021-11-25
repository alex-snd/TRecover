import os
import time
from argparse import Namespace
from pathlib import Path
from typing import Optional, List, Tuple, Generator
from zipfile import ZipFile

import psutil
import requests
import toml
from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn

from config import var, log
from zreader.utils.data import read_files_columns, create_files_noisy_columns


def get_real_direct_link(sharing_link: str) -> str:
    pk_request = requests.get(
        f'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={sharing_link}')

    return pk_request.json().get('href')  # Returns None if the link cannot be 'converted'


def extract_filename(direct_link: str) -> Optional[str]:
    for chunk in direct_link.strip().split('&'):
        if chunk.startswith('filename='):
            return chunk.split('=')[1]

    return None


def download_from_disk(sharing_link: str, save_dir: str) -> Optional[Path]:
    """
        Download file from Yandex disk

        Notes
        -----
        sharing_link: str
            Sharing link to the file on Yandex disk

        save_dir: str
            Path where to store downloaded file


        Returns
        -------
        filepath: Optional[Path]
            Path to the downloaded file. None if failed to download

    """

    if not (direct_link := get_real_direct_link(sharing_link)):
        log.project_logger.error(f'[red]Failed to download data from [/][bright_blue] {sharing_link}')
        return None

    filename = extract_filename(direct_link) or 'downloaded_data'  # Try to recover the filename from the link
    filepath = Path(save_dir, filename)

    with filepath.open(mode='wb') as fw:
        response = requests.get(direct_link, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            fw.write(response.content)
        else:
            data_stream = response.iter_content(chunk_size=4096)

            with Progress(
                    TextColumn('{task.description}', style='bright_blue'),
                    BarColumn(complete_style='bright_blue'),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    transient=True,
                    console=log.project_console
            ) as progress:
                download_progress = progress.add_task('Downloading', total=int(total_length))

                for data in data_stream:
                    fw.write(data)
                    progress.update(download_progress, advance=4096)

    log.project_console.print(f'Downloaded "{filename}" to {filepath.absolute()}', style='green')

    return filepath


def download_archive_from_disk(sharing_link: str, save_dir: str) -> None:
    """
        Download archive file from Yandex disk and extract it to save_dir

        Notes
        -----
        sharing_link: str
            Sharing link to the archive file on Yandex disk

        save_dir: str
            Path where to store extracted data

    """

    if filepath := download_from_disk(sharing_link, save_dir):
        with ZipFile(filepath) as zf:
            zf.extractall(path=Path(save_dir, filepath.stem))

        os.remove(filepath)

    log.project_console.print(f'Archive extracted to {Path(save_dir, filepath.stem).absolute()}', style='green')


def get_files_columns(inference_path: Path,
                      separator: str,
                      noisy: bool,
                      min_noise: int,
                      max_noise: int,
                      n_to_show: int,
                      ) -> Tuple[List[Path], List[List[str]]]:
    if inference_path.is_file():
        files = [inference_path, ]
    else:
        files = [file for file in inference_path.iterdir()]

    if noisy:
        files_columns = read_files_columns(files, separator, n_to_show)
    else:
        files_columns = create_files_noisy_columns(files, min_noise, max_noise, n_to_show)

    return files, files_columns


def parse_config(file: Path) -> Namespace:
    conf = var.DEFAULT_CONFIG
    parsed_conf = toml.load(file)

    for service, params in parsed_conf.items():
        for variable, value in params.items():
            conf[service][variable] = value

    for service, params in conf.items():
        conf[service] = Namespace(**params)

    return Namespace(**conf)


def stop_service(name: str, pidfile: Path) -> None:
    try:
        with pidfile.open() as f:
            pid = int(f.read())

        service = psutil.Process(pid=pid)

        for child_proc in service.children(recursive=True):
            child_proc.kill()

        service.kill()

    except ValueError:
        log.project_console.print(f'The {name} service could not be stopped correctly'
                                  ' because its PID file is corrupted', style='red')
    except (OSError, psutil.NoSuchProcess):
        log.project_console.print(f'The {name} service could not be stopped correctly'
                                  ' because it probably failed earlier', style='red')
    else:
        log.project_console.print(f'The {name} service is stopped', style='bright_blue')

    finally:
        if pidfile.exists():
            os.remove(pidfile)


def check_service(name: str, pidfile: Path) -> None:
    try:
        with pidfile.open() as f:
            pid = int(f.read())

        if psutil.pid_exists(pid):
            log.project_console.print(f':rocket: The {name} status: running', style='bright_blue')
        else:
            log.project_console.print(f'The {name} status: dead', style='red')

    except ValueError:
        log.project_console.print(f'The {name} service could not be checked correctly'
                                  ' because its PID file is corrupted', style='red')


def stream(logfile: Path, live: bool = False, period: float = 0.1) -> Generator[str, None, None]:
    with logfile.open() as log_stream:
        if live:
            log_stream.seek(0, 2)

        while True:
            if record := log_stream.readline():
                yield record
            else:
                time.sleep(period)
