from pathlib import Path
from typing import Optional, List, Tuple

import requests
from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn

from config import log
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

    log.project_logger.info(f'[green]Downloaded "{filename}" to {Path(save_dir, filename).absolute()}')

    return filepath


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
