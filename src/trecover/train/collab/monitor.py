import time
from typing import Generator, List, Optional

import hivemind
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
        loss = 0
        accuracy = 0
        samples_accumulated = 0
        samples_per_second = 0
        mini_steps = 0

        for item in metrics:
            loss += item.loss
            accuracy += item.accuracy
            samples_accumulated += item.samples_accumulated
            samples_per_second += item.samples_per_second
            mini_steps += item.mini_steps

        if mini_steps:
            loss /= mini_steps
            accuracy /= mini_steps

        return GlobalMetrics(
            samples_per_second=samples_per_second,
            samples_accumulated=samples_accumulated,
            loss=loss,
            accuracy=accuracy,
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
                self.aux_optimizer.step()

            log.project_console.print('Fetching metrics...', style='yellow', justify='right')
            time.sleep(self.refresh_period)

    def start(self) -> None:
        for metrics in self.stream():
            panel_group = Group(
                Text(f'Global loss: {metrics.loss}', style='bright_blue', justify='left'),
                Text(f'Global accuracy: {metrics.accuracy}', style='bright_blue', justify='left'),
                Text(f'Samples accumulated: {metrics.samples_accumulated}', style='bright_blue', justify='left'),
                Text(f'Performance: {metrics.samples_per_second} samples/sec', style='bright_blue', justify='left'),
                Text(f'Peers alive: {metrics.alive_peers}', style='bright_blue', justify='left')
            )

            log.project_console.print(
                Panel(panel_group, title=f'Global step {self.current_step}',
                      title_align='left', border_style='magenta'),
                justify='full'
            )
