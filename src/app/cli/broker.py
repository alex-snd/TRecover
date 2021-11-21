from typer import Typer, Option, Context

from config import var, log

cli = Typer(name='Broker-cli', add_completion=False)


@cli.callback(invoke_without_command=True)
def broker_state_verification(ctx: Context) -> None:
    from zreader.utils.docker import is_docker_running, get_container

    if not is_docker_running():
        log.project_console.print('Docker engine is not running', style='red')
        ctx.exit(1)

    elif container := get_container(var.BROKER_ID):
        if container.status == 'running' and ctx.invoked_subcommand in ('start', None):
            log.project_console.print(':rocket: The broker service is already started', style='bright_blue')
            ctx.exit(0)

    elif ctx.invoked_subcommand is None:
        broker_start(port=var.BROKER_PORT, ui_port=var.BROKER_UI_PORT, auto_remove=False, attach=False)

    elif ctx.invoked_subcommand != 'start':
        log.project_console.print('The broker service is not started', style='yellow')
        ctx.exit(1)


@cli.command(name='start')
def broker_start(port: int = Option(var.BROKER_PORT, '--port', '-p',
                                    help='Bind socket to this port.'),
                 ui_port: int = Option(var.BROKER_UI_PORT, '--port', '-p',
                                       help='Bind UI socket to this port.'),
                 auto_remove: bool = Option(False, '--rm', is_flag=True,
                                            help='Remove docker container after service exit'),
                 attach: bool = Option(False, '--attach', '-a', is_flag=True,
                                       help='Attach local standard input, output, and error streams')

                 ) -> None:
    from zreader.utils.docker import get_client, get_container, get_image, pull_image
    from rich.prompt import Confirm

    if not (image := get_image(var.BROKER_IMAGE)):
        with log.project_console.screen(hide_cursor=False):
            if not Confirm.ask(f"The broker image '{var.BROKER_IMAGE}' is needed to be pulled.\nContinue?",
                               default=True):
                return

        image = pull_image(var.BROKER_IMAGE)

    if container := get_container(var.BROKER_ID):
        container.start()

        log.project_console.print(f'The broker service is started', style='bright_blue')

    else:
        get_client().containers.run(image=image.id,
                                    name=var.BROKER_ID,
                                    auto_remove=auto_remove,
                                    detach=True,
                                    stdin_open=True,
                                    stdout=True,
                                    tty=True,
                                    stop_signal='SIGTERM',
                                    ports={5672: port, 15672: ui_port},
                                    volumes=[f'{var.BROKER_VOLUME_ID}:/data'])

        log.project_console.print(f'The broker service is launched', style='bright_blue')

    if attach:
        broker_attach()


@cli.command(name='stop')
def broker_stop(prune: bool = Option(False, '--prune', '-p', is_flag=True,
                                     help='Prune broker.'),
                v: bool = Option(False, '--volume', '-v', is_flag=True,
                                 help='Remove the volumes associated with the container')
                ) -> None:
    from zreader.utils.docker import get_container

    get_container(var.BROKER_ID).stop()

    log.project_console.print('The broker service is stopped', style='bright_blue')

    if prune:
        broker_prune(force=False, v=v)


@cli.command(name='prune')
def broker_prune(force: bool = Option(False, '--force', '-f', is_flag=True,
                                      help='Force the removal of a running container'),
                 v: bool = Option(False, '--volume', '-v', is_flag=True,
                                  help='Remove the volumes associated with the container')
                 ) -> None:
    from zreader.utils.docker import get_container, get_volume

    container = get_container(var.BROKER_ID)

    if container.status == 'running' and not force:
        log.project_console.print('You need to stop broker service before pruning or use --force flag', style='yellow')
    else:
        container.remove(force=force)

        if v and (volume := get_volume(var.BROKER_VOLUME_ID)):
            volume.remove(force=force)

        log.project_console.print('The broker service is pruned', style='bright_blue')


@cli.command(name='status')
def broker_status() -> None:
    from zreader.utils.docker import get_container

    log.project_console.print(f'Broker status: {get_container(var.BROKER_ID).status}', style='bright_blue')


@cli.command(name='attach')
def broker_attach() -> None:
    from zreader.utils.docker import get_container

    with log.project_console.screen(hide_cursor=True):
        for line in get_container(var.BROKER_ID).attach(stream=True, logs=True):
            log.project_console.print(line.decode().strip())


if __name__ == '__main__':
    try:
        cli()
    except Exception as e:
        log.project_logger.error(e)
        log.project_console.print_exception(show_locals=True)
        log.error_console.print_exception(show_locals=True)
