import time
from typing import Generator, List, Optional

import hivemind
import numpy as np
import wandb
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from trecover.config import log
from trecover.train.collab.dht import LocalMetrics, GlobalMetrics
from trecover.train.collab.optim import AuxiliaryOptimizer


class MetricsMonitor(object):
    def __init__(self,
                 dht: hivemind.DHT,
                 experiment_prefix: str,
                 refresh_period: int = 2,
                 upload_every_step: Optional[int] = None,
                 wandb_key: Optional[str] = None,
                 wandb_project: Optional[str] = None,
                 wandb_registry: Optional[str] = None,
                 aux_optimizer: Optional[AuxiliaryOptimizer] = None):
        self.dht = dht
        self.metrics_key = f'{experiment_prefix}_metrics'
        self.aux_optimizer = aux_optimizer
        self.wandb_report = wandb_key is not None
        self.current_step = -1
        self.refresh_period = refresh_period
        self.upload_every_step = upload_every_step

        if self.wandb_report:
            wandb.login(key=wandb_key)

            wandb.init(
                project=wandb_project,
                name='test_run',
                # id='',  # wandb.util.generate_id()
                dir=wandb_registry,
                resume='allow',
                anonymous='never'
            )

            if self.upload_every_step:
                wandb.save(f'{self.aux_optimizer.state_path.absolute()}*')

    def stream(self) -> Generator[GlobalMetrics, None, None]:
        while True:
            if (metrics_entry := self.dht.get(self.metrics_key, latest=True)) and (metrics_dict := metrics_entry.value):
                metrics = [LocalMetrics.parse_obj(metrics_dict[peer].value) for peer in metrics_dict]

                if (latest_step := max(item.step for item in metrics)) != self.current_step:
                    self.current_step = latest_step

                    yield self._average_peers_metrics(
                        [peer_metrics for peer_metrics in metrics if peer_metrics.step == latest_step]
                    )

            log.project_console.print('Fetching metrics...', style='yellow', justify='right')
            time.sleep(self.refresh_period)

    def start(self) -> None:
        try:
            for metrics in self.stream():
                self._print_metrics(metrics)

                if self.wandb_report:
                    self._report(metrics)

        except KeyboardInterrupt:
            log.project_console.print('Interrupted', style='yellow')

        finally:
            if self.wandb_report:
                wandb.finish()

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

    def _print_metrics(self, metrics: GlobalMetrics) -> None:
        panel_group = Group(
            Text(f'Global loss: {metrics.loss}', style='bright_blue', justify='left'),
            Text(f'Global accuracy: {metrics.accuracy}', style='bright_blue', justify='left'),
            Text(f'Learning rate: {metrics.lr}',
                 style='bright_blue', justify='left'),
            Text(f'Min-max noise range: {f"{metrics.min_noise}-{metrics.max_noise}"}',
                 style='bright_blue', justify='left'),
            Text(f'Samples accumulated: {metrics.samples_accumulated}', style='bright_blue', justify='left'),
            Text(f'Performance: {metrics.samples_per_second} samples/sec', style='bright_blue', justify='left'),
            Text(f'Peers alive: {metrics.alive_peers}', style='bright_blue', justify='left')
        )

        log.project_console.print(
            Panel(panel_group, title=f'Global step {self.current_step}',
                  title_align='left', border_style='magenta'),
            justify='full'
        )

    def _report(self, metrics: GlobalMetrics) -> None:
        wandb.log(metrics.dict(), step=self.current_step)

        if self.aux_optimizer and self.upload_every_step and self.current_step % self.upload_every_step == 0:
            log.project_console.print('Sync state with other peers and upload...', style='salmon1', justify='right')
            self.aux_optimizer.backup_state()  # TODO in different thread
