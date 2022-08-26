from typing import Optional

from rich.progress import Progress, TextColumn, SpinnerColumn

from trecover.config import log


class CommonStatus(object):
    def __init__(self, spinner_name: str = 'dots12', spinner_style='salmon1'):
        self.progress = Progress(
            SpinnerColumn(spinner_name=spinner_name, style=spinner_style, speed=0.5),
            TextColumn(text_format='{task.description}'),
            console=log.project_console,
            refresh_per_second=3,
            redirect_stdout=True,
            redirect_stderr=True
        )
        self.progress.start()

    def disable(self) -> None:
        self.progress.stop()


class Status(object):
    def __init__(self,
                 name: str,
                 name_style: str,
                 status_style: str,
                 common_status: Optional[CommonStatus] = None,
                 spinner_name: str = 'dots12',
                 spinner_style='salmon1'):
        self.name = name
        self.name_style = name_style
        self.status_style = status_style
        self.common_status = common_status or CommonStatus(spinner_name, spinner_style)
        self.id = self.common_status.progress.add_task(description=f'[{name_style}]{name}', visible=False)

    def enable(self) -> None:
        self.common_status.progress.update(
            self.id, description=f'[{self.name_style}]{self.name}: [{self.status_style}]Starting...', visible=True
        )
        self.common_status.progress.start_task(self.id)

    def disable(self) -> None:
        self.common_status.progress.stop_task(self.id)

    def update(self, status: str, style: Optional[str] = None) -> None:
        log.project_console.print(f'Update status to {status}', style=style, justify='left')
        if style:
            self.status_style = style

        self.common_status.progress.update(
            self.id, description=f'[{self.name_style}]{self.name}: [{self.status_style}]{status}'
        )
        log.project_console.print(f'End update status to {status}', style=style, justify='left')
