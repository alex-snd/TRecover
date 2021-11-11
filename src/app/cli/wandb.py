from typer import Typer, Option, Context

from config import var, log

cli = Typer(name='Wandb-cli', add_completion=False)


@cli.callback(invoke_without_command=True)
def wandb_state_verification(ctx: Context) -> None:
    from zreader.utils.docker import is_docker_running, get_container

    if not is_docker_running():
        log.project_console.print('Docker engine is not running', style='red')
        ctx.exit(1)

    elif (container := get_container(var.WANDB_ID)) and container.status == 'running':
        if ctx.invoked_subcommand in ('start', None):
            log.project_console.print(':rocket: The wandb service is already started', style='bright_blue')
            ctx.exit(0)

    elif ctx.invoked_subcommand is None:
        wandb_start(port=var.WANDB_PORT, auto_remove=False, attach=False)

    elif not container and ctx.invoked_subcommand != 'start':
        log.project_console.print('The wandb service is not started', style='yellow')
        ctx.exit(1)


@cli.command(name='start')
def wandb_start(port: int = Option(var.WANDB_PORT, '--port', '-p',
                                   help='Bind socket to this port.'),
                auto_remove: bool = Option(False, '--rm', is_flag=True,
                                           help='Remove docker container after service exit'),
                attach: bool = Option(False, '--attach', '-a', is_flag=True,
                                      help='Attach local standard input, output, and error streams')

                ) -> None:
    from zreader.utils.docker import client, get_container, get_image, pull_image
    from rich.prompt import Confirm

    if not (image := get_image(var.WANDB_IMAGE)):
        with log.project_console.screen(hide_cursor=False):
            if not Confirm.ask(f"The wandb image '{var.WANDB_IMAGE}' is needed to be pulled.\nContinue?",
                               default=True):
                return

        image = pull_image(var.WANDB_IMAGE)

    if container := get_container(var.WANDB_ID):
        container.start()

        log.project_console.print(f'The wandb service is started', style='bright_blue')

    else:
        client.containers.run(image=image.id,
                              name=var.WANDB_ID,
                              auto_remove=auto_remove,
                              detach=True,
                              stdin_open=True,
                              stdout=True,
                              tty=True,
                              stop_signal='SIGTERM',
                              ports={8080: port},
                              volumes={var.WANDB_REGISTRY_DIR: {'bind': '/vol', 'mode': 'rw'}},
                              environment={'LOGGING_ENABLED': True})

        log.project_console.print(f'The wandb service is launched', style='bright_blue')

    if attach:
        wandb_attach()


@cli.command(name='stop')
def wandb_stop(prune: bool = Option(False, '--prune', '-p', is_flag=True,
                                    help='Prune wandb.'),
               v: bool = Option(False, '--volume', '-v', is_flag=True,
                                help='Remove the volumes associated with the container')
               ) -> None:
    from zreader.utils.docker import get_container

    get_container(var.WANDB_ID).stop()

    log.project_console.print('The wandb service is stopped', style='bright_blue')

    if prune:
        wandb_prune(force=False, v=v)


@cli.command(name='prune')
def wandb_prune(force: bool = Option(False, '--force', '-f', is_flag=True,
                                     help='Force the removal of a running container'),
                v: bool = Option(False, '--volume', '-v', is_flag=True,
                                 help='Remove the volumes associated with the container')
                ) -> None:
    from zreader.utils.docker import get_container

    container = get_container(var.WANDB_ID)

    if container.status == 'running' and not force:
        log.project_console.print('You need to stop wandb service before pruning or use --force flag', style='yellow')
    else:
        container.remove(v=v, force=force)
        log.project_console.print('The wandb service is pruned', style='bright_blue')


@cli.command(name='status')
def wandb_status() -> None:
    from zreader.utils.docker import get_container

    log.project_console.print(f'Wandb status: {get_container(var.WANDB_ID).status}', style='bright_blue')


@cli.command(name='attach')
def wandb_attach() -> None:
    from zreader.utils.docker import get_container

    with log.project_console.screen(hide_cursor=True):
        for line in get_container(var.WANDB_ID).attach(stream=True, logs=True):
            log.project_console.print(line.decode().strip())


if __name__ == '__main__':
    try:
        cli()
    except Exception as e:
        log.project_logger.error(e)
        log.project_console.print_exception(show_locals=True)
        log.error_console.print_exception(show_locals=True)
