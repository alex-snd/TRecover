from typing import List, Optional

import docker
from docker.models.containers import Container
from docker.models.images import Image
from docker.models.volumes import Volume
from rich.live import Live

from config import log

client = docker.from_env()


def is_docker_running() -> bool:
    try:
        client.ping()
    except docker.errors.APIError:
        return False

    return True


def get_images_list() -> List[str]:
    image_names = list()

    for image in client.images.list():
        image_names.extend(image.tags)

    return image_names


def get_containers_list() -> List[str]:
    return [container.name for container in client.containers.list(all=True)]


def get_container(id_or_name: str) -> Optional[Container]:
    try:
        return client.containers.get(id_or_name)
    except docker.errors.NotFound:
        return None


def get_volume(id_or_name: str) -> Optional[Volume]:
    try:
        return client.volumes.get(id_or_name)
    except docker.errors.NotFound:
        return None


def get_image(name: str) -> Optional[Image]:
    try:
        return client.images.get(name)
    except docker.errors.NotFound:
        return None


def pull_image(name: str) -> Image:
    with Live(console=log.project_console) as screen:
        screen.update('[bright_blue]Waiting')
        for state in client.api.pull(name, stream=True, decode=True):
            try:
                screen.update(f'[bright_blue]{state["status"]}[/] {state["progress"]}')
            except KeyError:
                screen.update(f'[bright_blue]{state["status"]}')

    return get_image(name)
