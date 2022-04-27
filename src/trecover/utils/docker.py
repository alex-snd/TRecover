from typing import List, Optional

import docker
from docker import DockerClient
from docker.models.containers import Container
from docker.models.images import Image
from docker.models.volumes import Volume
from rich.live import Live

from trecover.config import log


def get_client() -> DockerClient:
    """ Get the docker client """

    return docker.from_env()


def is_docker_running() -> bool:
    """ Check if docker is running """
    try:
        get_client().ping()
    except (docker.errors.APIError, docker.errors.DockerException):
        return False

    return True


def get_images_list() -> List[str]:
    """ Get list of docker images """
    image_names = list()

    for image in get_client().images.list():
        image_names.extend(image.tags)

    return image_names


def get_containers_list() -> List[str]:
    """ Get list of docker containers """

    return [container.name for container in get_client().containers.list(all=True)]


def get_container(id_or_name: str) -> Optional[Container]:
    """
    Get the docker container with given name or id.

    Parameters
    ----------
    id_or_name : str
        Docker container id or name.

    Returns
    -------
    Optional[Container]:
        Container instance if it exists, otherwise None.

    """

    try:
        return get_client().containers.get(id_or_name)
    except docker.errors.NotFound:
        return None


def get_volume(id_or_name: str) -> Optional[Volume]:
    """
    Get the docker volume with given name or id.

    Parameters
    ----------
    id_or_name : str
        Volume id or name.

    Returns
    -------
    Optional[Volume]:
        Volume instance if it exists, otherwise None.

    """

    try:
        return get_client().volumes.get(id_or_name)
    except docker.errors.NotFound:
        return None


def get_image(name: str) -> Optional[Image]:
    """
    Get the docker image with given name.

    Parameters
    ----------
    name : str
        Docker image name.

    Returns
    -------
    Optional[Image]:
        Image instance if it exists, otherwise None.

    """

    try:
        return get_client().images.get(name)
    except docker.errors.NotFound:
        return None


def pull_image(name: str) -> Image:
    """
    Pull (download) the docker image with given name.

    Parameters
    ----------
    name : str
        Docker image name.

    Returns
    -------
    Image:
        Image instance of the pulled docker image.

    """

    with Live(console=log.project_console) as screen:
        screen.update('[bright_blue]Waiting')
        for state in get_client().api.pull(name, stream=True, decode=True):
            try:
                screen.update(f'[bright_blue]{state["status"]}[/] {state["progress"]}')
            except KeyError:
                screen.update(f'[bright_blue]{state["status"]}')

    return get_image(name)
