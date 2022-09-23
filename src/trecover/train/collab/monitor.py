import time
from collections import OrderedDict
from typing import Generator, List, Optional, Tuple, Dict

import hivemind
import numpy as np
import wandb
from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from wandb.util import generate_id

from trecover.config import log
from trecover.train.collab.dht import LocalMetrics, GlobalMetrics
from trecover.train.collab.optim import AuxiliaryOptimizer
from trecover.train.collab.status import CommonStatus, Status


class CollaborativeMonitor(object):
    def __init__(self,
                 dht: hivemind.DHT,
                 experiment_prefix: str,
                 delay_in_steps: int = 1,
                 delay_in_seconds: int = 180,
                 refresh_period: int = 10,
                 upload_state: bool = False,
                 wandb_key: Optional[str] = None,
                 wandb_project: Optional[str] = None,
                 wandb_id: Optional[str] = None,
                 wandb_registry: Optional[str] = None,
                 aux_opt: Optional[AuxiliaryOptimizer] = None,
                 common_status: Optional[CommonStatus] = None):
        self.dht = dht
        self.metrics_key = f'{experiment_prefix}_metrics'
        self.aux_opt = aux_opt
        self.wandb_report = wandb_key is not None
        self.steps_metrics = OrderedDict()
        self.delay_in_steps = delay_in_steps
        self.delay_in_seconds = delay_in_seconds
        self.last_yield_time = time.monotonic()
        self.last_yield_step = -1
        self.refresh_period = refresh_period
        self.upload_state = upload_state
        self.last_upload_time = time.monotonic()
        self.stopped = False
        self.status = Status(name='Monitor', name_style='dark_violet', status_style='bright_blue',
                             common_status=common_status)

        if self.wandb_report:
            wandb.login(key=wandb_key)

            if wandb_id is None:
                wandb_id = generate_id()

            wandb.init(
                project=wandb_project,
                name=wandb_id,
                id=wandb_id,
                dir=wandb_registry,
                resume='allow',
                anonymous='never',
                settings=wandb.Settings(start_method='thread')
            )

        self._peer_status()

    def stream(self) -> Generator[Tuple[int, GlobalMetrics], None, None]:
        self.status.update('Fetching metrics...')

        while not self.stopped or self.steps_metrics:
            if metrics_dict := self._fetch_metrics_dict():
                for peer, metrics in metrics_dict.items():
                    metrics = LocalMetrics.parse_obj(metrics.value)
                    if metrics.step > self.last_yield_step:
                        self.steps_metrics.setdefault(metrics.step, dict())
                        self.steps_metrics[metrics.step][peer] = metrics

            if self._is_time_to_yield:
                step, step_peers_metrics = self.steps_metrics.popitem(last=False)
                yield step, self._average_peers_metrics(step_peers_metrics.values())

                self.last_yield_time = time.monotonic()
                self.last_yield_step = step

            time.sleep(self.refresh_period)

    def start(self) -> None:
        try:
            self.status.enable()
            self._monitor_loop()

        except KeyboardInterrupt:
            self.status.update('Stopping...', style='yellow')
        finally:
            self.stopped = True

            if self.steps_metrics:
                self.status.update(f'Trying to report {len(self.steps_metrics)} delayed metrics...', style='yellow')
                self.delay_in_seconds = 0
                self.refresh_period = 0
                self._monitor_loop()

            self.status.update('Is stopped', style='yellow')
            self.status.disable()

    def _peer_status(self) -> None:
        if not self.wandb_report:
            log.project_console.print(
                'This peer does not report metrics to the W&B',
                style='yellow', justify='right'
            )
        if not self.upload_state:
            log.project_console.print(
                'This peer does not upload collab state to the W&B',
                style='yellow', justify='right'
            )
        elif not self.wandb_report:
            log.project_console.print(
                'Unable to upload collab state to the W&B since credentials did not specified'
                'Specify credentials with `--wandb-key` argument',
                style='red', justify='right'
            )
        elif self.aux_opt is None:
            log.project_console.print(
                'Unable to upload collab state to the W&B since AuxiliaryOptimizer is not initialized',
                style='red', justify='right'
            )
        elif self.aux_opt.args.backup_every_step is None:
            log.project_console.print(
                'Unable to upload collab state to the W&B since backup frequency is not specified.'
                'Specify frequency with `--backup-every-step` argument',
                style='red', justify='right'
            )
        elif self.aux_opt.args.backup_every_step <= 0:
            log.project_console.print(
                'Unable to upload collab state to the W&B since backup frequency is specified incorrectly.',
                style='red', justify='right'
            )

    def _fetch_metrics_dict(self) -> Optional[Dict]:
        if (
                not self.stopped
                and (metrics_entry := self.dht.get(self.metrics_key, latest=True))
                and (metrics_dict := metrics_entry.value)
        ):
            return metrics_dict

    @property
    def _is_time_to_yield(self) -> bool:
        if len(self.steps_metrics) == 0:
            return False
        if time.monotonic() - self.last_yield_time > self.delay_in_seconds:
            return True
        if len(self.steps_metrics) > self.delay_in_steps:
            return True

        return False

    @property
    def _is_time_to_upload(self) -> bool:
        return (
                self.wandb_report
                and self.aux_opt
                and self.aux_opt.state_path.exists()
                and self.aux_opt.state_path.stat().st_mtime > self.last_upload_time
        )

    @staticmethod
    def _average_peers_metrics(metrics: List[LocalMetrics]) -> GlobalMetrics:
        loss = sum([item.loss for item in metrics])
        accuracy = sum([item.accuracy for item in metrics])
        lr = np.median([item.lr for item in metrics])
        min_noise = min([item.min_noise for item in metrics])
        max_noise = max([item.max_noise for item in metrics])
        samples_accumulated = sum([item.samples_accumulated for item in metrics])
        samples_per_second = sum([item.samples_per_second for item in metrics])
        mini_steps = sum([item.mini_steps for item in metrics])

        if mini_steps:
            loss /= mini_steps
            accuracy /= mini_steps

        return GlobalMetrics(
            loss=loss,
            accuracy=accuracy,
            lr=lr,
            min_noise=min_noise,
            max_noise=max_noise,
            samples_per_second=samples_per_second,
            samples_accumulated=samples_accumulated,
            alive_peers=len(metrics)
        )

    def _monitor_loop(self) -> None:
        for step, metrics in self.stream():
            self._print_metrics(step, metrics)

            if self.wandb_report:
                wandb.log(metrics.dict(), step=step)

            self._upload_state()

    @staticmethod
    def _print_metrics(step: int, metrics: GlobalMetrics) -> None:
        panel_group = Group(
            Text(f'Global loss: {metrics.loss:.5f}', style='bright_blue', justify='left'),
            Text(f'Global accuracy: {metrics.accuracy:.5f}', style='bright_blue', justify='left'),
            Text(f'Learning rate: {metrics.lr}', style='bright_blue', justify='left'),
            Text(f'Min-max noise range: {f"{metrics.min_noise}-{metrics.max_noise}"}',
                 style='bright_blue', justify='left'),
            Text(f'Samples accumulated: {metrics.samples_accumulated:,}', style='bright_blue', justify='left'),
            Text(f'Performance: {metrics.samples_per_second:.2f} samples/sec', style='bright_blue', justify='left'),
            Text(f'Peers alive: {metrics.alive_peers}', style='bright_blue', justify='left')
        )

        log.project_console.print(
            Panel(panel_group, title=f'Global step {step}', title_align='left', border_style='magenta'),
            justify='full'
        )

    def _upload_state(self) -> None:
        if self._is_time_to_upload:
            wandb.save(str(self.aux_opt.state_path.absolute()),
                       base_path=str(self.aux_opt.state_path.parent),
                       policy='now')
            self.last_upload_time = time.monotonic()
