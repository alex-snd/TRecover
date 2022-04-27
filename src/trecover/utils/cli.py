import os
import platform
import time
from argparse import Namespace
from pathlib import Path
from subprocess import Popen, STDOUT
from typing import Optional, List, Tuple, Generator, Union
from zipfile import ZipFile

import psutil
import requests
import toml
from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn

from trecover.config import var, log


def download(direct_link: str, filepath: Path) -> None:
    """
    Download file.

    Parameters
    ----------
    direct_link : str
        Sharing link to the file on GutHub.
    filepath : Path
        Path to the downloaded file.

   """

    filepath.parent.mkdir(parents=True, exist_ok=True)

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


def get_real_direct_link(sharing_link: str) -> str:
    """
    Get a direct download link.

    Parameters
    ----------
    sharing_link : str
        Sharing link to the file on Yandex disk.

    Returns
    -------
    str:
        Direct link if it converts, otherwise None.

    """

    pk_request = requests.get(
        f'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={sharing_link}')

    return pk_request.json().get('href')


def extract_filename(direct_link: str) -> Optional[str]:
    """ Get filename of downloaded data """

    for chunk in direct_link.strip().split('&'):
        if chunk.startswith('filename='):
            return chunk.split('=')[1]

    return None


def download_from_disk(sharing_link: str, save_dir: Path) -> Optional[Path]:
    """
    Download file from Yandex disk.

    Parameters
    ----------
    sharing_link : str
        Sharing link to the file on Yandex disk.
    save_dir : Path
        Path where to store downloaded file.

    Returns
    -------
    filepath : Optional[Path]
        Path to the file if download was successful, otherwise None.

    """

    if not (direct_link := get_real_direct_link(sharing_link)):
        log.project_logger.error(f'[red]Failed to download data from [/][bright_blue] {sharing_link}')
        return None

    filename = extract_filename(direct_link) or 'downloaded_data'  # Try to recover the filename from the link
    filepath = save_dir / filename

    download(direct_link=direct_link, filepath=filepath)

    log.project_console.print(f'Downloaded "{filename}" to {filepath.absolute()}', style='green')

    return filepath


def download_from_github(direct_link: str, save_dir: Path) -> Path:
    """
    Download file from GutHub assets.

    Parameters
    ----------
    direct_link : str
        Sharing link to the file on GutHub.
    save_dir : Path
        Path where to store downloaded file.

    Returns
    -------
    filepath : Path
        Path to the downloaded file.

   """

    filename = direct_link.split('/')[-1]
    filepath = save_dir / filename

    download(direct_link=direct_link, filepath=filepath)

    log.project_console.print(f'Downloaded "{filename}" to {filepath.absolute()}', style='green')

    return filepath


def download_archive(link: str, save_dir: Path, yandex_disk: bool = False) -> None:
    """
    Download archive file and extract it to save_dir.

    Parameters
    ----------
    link : str
        Sharing link to the archive file on Yandex disk or GitHub assets.
    save_dir : Path
        Path where to store extracted data
    yandex_disk : bool, default=False
        If the link is to Yandex disk.

    """

    filepath = download_from_disk(link, save_dir) if yandex_disk else download_from_github(link, save_dir)

    if filepath:
        with ZipFile(filepath) as zf:
            zf.extractall(path=save_dir)

        os.remove(filepath)

        log.project_console.print(f'Archive extracted to {save_dir.absolute()}', style='green')


def get_files_columns(inference_path: Path,
                      separator: str,
                      noisy: bool,
                      min_noise: int,
                      max_noise: int,
                      n_to_show: int,
                      ) -> Tuple[List[Path], List[List[str]]]:
    """
    Get columns for keyless reading from plain data contained in the files with defined noise range.

    Parameters
    ----------
    inference_path : Path
        Paths to folder with files that contain data to read or create noised columns for keyless reading.
    separator : str
        Separator to split the data into columns.
    noisy : bool
        Indicates that the data in the files is already noisy and contains columns for keyless reading.
    min_noise : int
        Minimum noise range value.
    max_noise : int
        Maximum noise range value.
    n_to_show : int
        Maximum number of columns. Zero means no restrictions.

    Returns
    -------
    (files, files_columns) : Tuple[List[Path], List[List[str]]]
        List of paths and batch of columns for keyless reading.

    """

    from trecover.utils.inference import read_files_columns, create_files_noisy_columns

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
    """ Parse configuration file for 'trecover up' command. """

    conf = var.DEFAULT_CONFIG
    parsed_conf = toml.load(file)

    for service, params in parsed_conf.items():
        for variable, value in params.items():
            conf[service][variable] = value

    for service, params in conf.items():
        conf[service] = Namespace(**params)

    return Namespace(**conf)


def start_service(argv: List[str], name: str, logfile: Path, pidfile: Path) -> None:
    """
    Start service as a new process with given pid and log files.

    Parameters
    ----------
    argv : List[str]
        New process command.
    name : str
        Service name.
    logfile : Path
        Service logfile path.
    pidfile : Path
        Service pidfile path.

    """

    if platform.system() == 'Windows':
        from subprocess import CREATE_NO_WINDOW

        process = Popen(argv, creationflags=CREATE_NO_WINDOW, stdout=logfile.open(mode='w+'), stderr=STDOUT,
                        universal_newlines=True, start_new_session=True)
    else:
        process = Popen(argv, stdout=logfile.open(mode='w+'), stderr=STDOUT, universal_newlines=True,
                        start_new_session=True)

    with pidfile.open('w') as f:
        f.write(str(process.pid))

    log.project_console.print(f'The {name} service is started', style='bright_blue')


def stop_service(name: str, pidfile: Path, logfile: Path) -> None:
    """
    Send an interrupt signal to the process with given pid.

    Parameters
    ----------
    name : str
        Service name.
    pidfile : Path
        Service pidfile path.
    logfile : Path
        Service logfile path.

    """

    try:
        with pidfile.open() as f:
            pid = int(f.read())

        service = psutil.Process(pid=pid)

        for child_proc in service.children(recursive=True):
            child_proc.kill()

        if platform.system() != 'Windows':
            service.kill()

        service.wait()

        if logfile.exists():
            os.remove(logfile)

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
    """
    Display status of the process with given pid.

    Parameters
    ----------
    name : str
        Service name.
    pidfile : Path
        Service pidfile path.

    """

    try:
        with pidfile.open() as f:
            pid = int(f.read())

        if psutil.pid_exists(pid):
            log.project_console.print(f':rocket: The {name} status: running', style='bright_blue')
        else:
            log.project_console.print(f'The {name} status: dead', style='red')

    except FileNotFoundError:
        log.project_console.print(f'The {name} service is not started', style='yellow')

    except ValueError:
        log.project_console.print(f'The {name} service could not be checked correctly'
                                  ' because its PID file is corrupted', style='red')


def stream(*services: Union[Tuple[str, Path], Tuple[Tuple[str, Path]]],
           live: bool = False,
           period: float = 0.1
           ) -> Optional[Generator[str, None, None]]:
    """
    Get a generator that yields the services' stdout streams.

    Parameters
    ----------
    *services : Union[Tuple[str, Path], Tuple[Tuple[str, Path]]]
        Sequence of services' names and logfile's paths.
    live : bool, default=False
        Yield only new services' logs.
    period : float, default=0.1
        Generator's delay.

    Returns
    -------
    Optional[Generator[str, None, None]]:
        Generator that yields the services' stdout streams or None if services are stopped.

    """

    names = list()
    streams = list()
    alignment = 0

    try:
        for (name, log_file) in services:
            if not log_file.exists():
                continue

            names.append(name)

            service_stream = log_file.open()

            if live:
                service_stream.seek(0, 2)

            streams.append(service_stream)

            if len(name) > alignment:
                alignment = len(name)

        if not (n_services := len(names)):
            return None

        while True:
            for i in range(n_services):
                if record := streams[i].read().strip():
                    color = var.COLORS[i % len(var.COLORS)]

                    for record_line in record.split('\n'):
                        yield f'[{color}]{names[i]: <{alignment}} |[/{color}] {record_line}'

            time.sleep(period)

    finally:
        for service_stream in streams:
            service_stream.close()
