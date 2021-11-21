from typer import Typer, Option, Context

from config import var, log

cli = Typer(name='Backend-cli', add_completion=False)


@cli.callback(invoke_without_command=True)
def backend_state_verification(ctx: Context) -> None:
    from zreader.utils.docker import is_docker_running, get_container

    if not is_docker_running():
        log.project_console.print('Docker engine is not running', style='red')
        ctx.exit(1)

    elif container := get_container(var.BACKEND_ID):
        if container.status == 'running' and ctx.invoked_subcommand in ('start', None):
            log.project_console.print(':rocket: The backend service is already started', style='bright_blue')
            ctx.exit(0)

    elif ctx.invoked_subcommand is None:
        backend_start(port=var.BACKEND_PORT, auto_remove=False, attach=False)

    elif ctx.invoked_subcommand != 'start':
        log.project_console.print('The backend service is not started', style='yellow')
        ctx.exit(1)


@cli.command(name='start')
def backend_start(port: int = Option(var.BACKEND_PORT, '--port', '-p',
                                     help='Bind socket to this port.'),
                  auto_remove: bool = Option(False, '--rm', is_flag=True,
                                             help='Remove docker container after service exit'),
                  attach: bool = Option(False, '--attach', '-a', is_flag=True,
                                        help='Attach local standard input, output, and error streams')

                  ) -> None:
    from zreader.utils.docker import get_client, get_container, get_image, pull_image
    from rich.prompt import Confirm

    if not (image := get_image(var.BACKEND_IMAGE)):
        with log.project_console.screen(hide_cursor=False):
            if not Confirm.ask(f"The backend image '{var.BACKEND_IMAGE}' is needed to be pulled.\nContinue?",
                               default=True):
                return

        image = pull_image(var.BACKEND_IMAGE)

    if container := get_container(var.BACKEND_ID):
        container.start()

        log.project_console.print(f'The backend service is started', style='bright_blue')

    else:
        get_client().containers.run(image=image.id,
                                    name=var.BACKEND_ID,
                                    auto_remove=auto_remove,
                                    detach=True,
                                    stdin_open=True,
                                    stdout=True,
                                    tty=True,
                                    stop_signal='SIGTERM',
                                    ports={6379: port},
                                    volumes=[f'{var.BACKEND_VOLUME_ID}:/data'])

        log.project_console.print(f'The backend service is launched', style='bright_blue')

    if attach:
        backend_attach()


@cli.command(name='stop')
def backend_stop(prune: bool = Option(False, '--prune', '-p', is_flag=True,
                                      help='Prune backend.'),
                 v: bool = Option(False, '--volume', '-v', is_flag=True,
                                  help='Remove the volumes associated with the container')
                 ) -> None:
    from zreader.utils.docker import get_container

    get_container(var.BACKEND_ID).stop()

    log.project_console.print('The backend service is stopped', style='bright_blue')

    if prune:
        backend_prune(force=False, v=v)


@cli.command(name='prune')
def backend_prune(force: bool = Option(False, '--force', '-f', is_flag=True,
                                       help='Force the removal of a running container'),
                  v: bool = Option(False, '--volume', '-v', is_flag=True,
                                   help='Remove the volumes associated with the container')
                  ) -> None:
    from zreader.utils.docker import get_container, get_volume

    container = get_container(var.BACKEND_ID)

    if container.status == 'running' and not force:
        log.project_console.print('You need to stop backend service before pruning or use --force flag', style='yellow')
    else:
        container.remove(force=force)

        if v and (volume := get_volume(var.BACKEND_VOLUME_ID)):
            volume.remove(force=force)

        log.project_console.print('Backend service is pruned', style='bright_blue')


@cli.command(name='status')
def backend_status() -> None:
    from zreader.utils.docker import get_container

    status = container.status if (container := get_container(var.BACKEND_ID)) else '[yellow]is not started'

    log.project_console.print(f'Backend status: {status}', style='bright_blue')


@cli.command(name='attach')
def backend_attach() -> None:
    from zreader.utils.docker import get_container

    with log.project_console.screen(hide_cursor=True):
        for line in get_container(var.BACKEND_ID).attach(stream=True, logs=True):
            log.project_console.print(line.decode().strip())


if __name__ == '__main__':
    try:
        cli()
    except Exception as e:
        log.project_logger.error(e)
        log.project_console.print_exception(show_locals=True)
        log.error_console.print_exception(show_locals=True)
