from typing import Optional, Dict, Any

import hivemind
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.trainer.states import TrainerFn
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from trecover.config import log
from trecover.train.collab.dht import DHTManager, LocalMetrics
from trecover.train.collab.optim import CollaborativeOptimizer
from trecover.train.collab.wrapper import BaseModelWrapper


class CollabCheckpoint(Callback):
    def __init__(self,
                 dht_manager: DHTManager,
                 statistics_expiration: float,
                 backup_every_step: int,
                 sync_period: Optional[int] = None):
        self.dht_manager: DHTManager = dht_manager
        self.wrapped_model: Optional[BaseModelWrapper] = None
        self.collab_opt: Optional[CollaborativeOptimizer] = None

        self.statistics_expiration = statistics_expiration
        self.last_reported_step = None
        self.min_noise = None
        self.max_noise = None
        self.lr = None
        self.samples = 0
        self.steps = 0
        self.loss = 0
        self.accuracy = 0
        self.total_samples_processed = 0
        self.samples_per_second = 0
        self.alive_peers = 0
        self.backup_every_step = backup_every_step
        self.sync_period = sync_period

    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: Dict[str, Any],
                           *args, **kwargs
                           ) -> None:
        if self.collab_opt is None:
            assert len(trainer.strategy.optimizers) == 1, 'Hivemind only supports training with one optimizer.'
            self.collab_opt = trainer.strategy.collab_opt
            self.last_reported_step = self.collab_opt.local_epoch

        import time
        log.project_console.print('Sleep in callback on_train_batch_end', style='yellow', justify='center')
        time.sleep(10)
        log.project_console.print('End Sleep in callback on_train_batch_end', style='yellow', justify='center')

        if not self.collab_opt.params_are_finite:
            log.project_console.print('Model parameters are not finite', style='red', justify='right')
            self.collab_opt.recover_state()
            return

        self.steps += 1
        self.loss += outputs['loss'].item()
        self.accuracy += outputs['accuracy']

        if (current_step := self.collab_opt.local_epoch) != self.last_reported_step and current_step != 0:
            self._report_metrics(step=current_step)

            if not self._should_skip_saving_checkpoint(trainer):
                self.collab_opt.backup_state()
            else:
                log.project_console.print('Skip backup', style='yellow', justify='right')

            if self.sync_period and current_step % self.sync_period == 0:
                self.collab_opt.sync_collate()

            self.last_reported_step = current_step

        self.samples = self.collab_opt.local_samples_accumulated

    def _report_metrics(self, step: int) -> None:
        self.total_samples_processed += self.samples
        self.samples_per_second = self.collab_opt.samples_per_second
        self.lr = self.collab_opt.lr
        self.min_noise = self.collab_opt.min_noise
        self.max_noise = self.collab_opt.max_noise
        self.alive_peers = self.collab_opt.num_peers

        statistics = LocalMetrics(
            loss=self.loss,
            accuracy=self.accuracy,
            lr=self.lr,
            min_noise=self.min_noise,
            max_noise=self.max_noise,
            samples_per_second=self.samples_per_second,
            samples_accumulated=self.samples,
            mini_steps=self.steps,
            step=step - 1
        )

        self._print_metrics(step=step - 1)

        if self.collab_opt.local_epoch == self.collab_opt.global_epoch:
            if not self.dht_manager.dht.store(
                    key=f'{self.collab_opt.run_id}_metrics',
                    subkey=self.dht_manager.local_public_key,
                    value=statistics.dict(),
                    expiration_time=hivemind.get_dht_time() + self.statistics_expiration,
                    return_future=True
            ):
                log.project_console.print('Failed to store metrics', style='red', justify='right')

        self.steps = 0
        self.loss = 0
        self.accuracy = 0

    def _print_metrics(self, step: int) -> None:
        panel_group = Group(Text(f'Local loss: {self.loss / self.steps:.5f}',
                                 style='bright_blue', justify='left'),
                            Text(f'Local accuracy: {self.accuracy / self.steps:.5f}',
                                 style='bright_blue', justify='left'),
                            Text(f'Learning rate: {self.lr}',
                                 style='bright_blue', justify='left'),
                            Text(f'Min-max noise range: {f"{self.min_noise}-{self.max_noise}"}',
                                 style='bright_blue', justify='left'),
                            Text(f'Your current contribution: {self.samples:,} samples',
                                 style='bright_blue', justify='left'),
                            Text(f'Your total contribution: {self.total_samples_processed:,} samples',
                                 style='bright_blue', justify='left'),
                            Text(f'Performance: {self.samples_per_second:.2f} samples/sec',
                                 style='bright_blue', justify='left'),
                            Text(f'Peers alive: {self.alive_peers}',
                                 style='bright_blue', justify='left'))

        log.project_console.print(
            Panel(panel_group, title=f'Local step {step:_}', title_align='left', border_style='magenta'),
            justify='full'
        )

    def _should_skip_saving_checkpoint(self, trainer: pl.Trainer) -> bool:
        return (
                trainer.fast_dev_run  # disable checkpointing with fast_dev_run
                or trainer.state.fn != TrainerFn.FITTING  # don't save anything during non-fit
                or trainer.sanity_checking  # don't save anything during sanity check
                or self.last_reported_step == self.collab_opt.local_epoch  # already saved the last step
                or self.backup_every_step is None  # backup is disabled
                or (self.collab_opt.local_epoch + 1) % self.backup_every_step != 0  # not at the current step
        )
