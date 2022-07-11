import time
from typing import Generator, List, Optional

import hivemind
import numpy as np
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from trecover.config import log
from trecover.train.collab.dht import LocalMetrics, GlobalMetrics


# TODO WandB log and optimizer initialization ?visualization? and loading_from_peers
class MetricsMonitor(object):
    def __init__(self, dht: hivemind.DHT, experiment_prefix: str, aux_optimizer: Optional[hivemind.Optimizer] = None):
        self.dht = dht
        self.metrics_key = f'{experiment_prefix}_metrics'
        self.aux_optimizer = aux_optimizer
        self.current_step = -1
        self.refresh_period = 2

    @staticmethod
    def average_peers_metrics(metrics: List[LocalMetrics]) -> GlobalMetrics:
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

    def stream(self) -> Generator[GlobalMetrics, None, None]:
        while True:
            if metrics_entry := self.dht.get(self.metrics_key, latest=True):
                if metrics_dict := metrics_entry.value:
                    metrics = [LocalMetrics.parse_obj(metrics_dict[peer].value) for peer in metrics_dict]

                    if (latest_step := max(item.step for item in metrics)) != self.current_step:
                        self.current_step = latest_step

                        yield self.average_peers_metrics(metrics)

            if self.aux_optimizer:
                try:
                    self.aux_optimizer.step()
                except Exception as e:
                    log.project_logger.exception(e, exc_info=True)

            log.project_console.print('Fetching metrics...', style='yellow', justify='right')
            time.sleep(self.refresh_period)

    def start(self) -> None:
        for metrics in self.stream():
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
