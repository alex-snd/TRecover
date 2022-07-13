from pathlib import Path
from typing import Optional, Dict, Any

import hivemind
import pytorch_lightning as pl
import torch.nn
from pytorch_lightning.callbacks.base import Callback
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from trecover.config import log
from trecover.train.collab.dht import DHTManager, LocalMetrics


class CollabCheckpoint(Callback):
    def __init__(self, dht_manager: DHTManager,
                 statistics_expiration: float,
                 backup_every_step: int,
                 state_path: Path):
        self.dht_manager: DHTManager = dht_manager
        self.pl_module: Optional[pl.LightningModule] = None
        self.optimizer: Optional[hivemind.Optimizer] = None

        self.statistics_expiration = statistics_expiration
        self.last_reported_collaboration_step = None
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
        self.state_path = state_path

    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: Dict[str, Any],
                           *args, **kwargs
                           ) -> None:
        if self.optimizer is None:
            assert len(trainer.strategy.optimizers) == 1, 'Hivemind only supports training with one optimizer.'
            self.optimizer = trainer.strategy.optimizers[0]
            self.last_reported_collaboration_step = self.optimizer.local_epoch

        if self.pl_module is None:
            self.pl_module = pl_module
            self.min_noise = pl_module.args.min_noise  # TODO as property or not, torch.hub.load?
            self.max_noise = pl_module.args.max_noise

        if not self._params_are_finite():
            log.project_console.print('Model parameters are not finite', style='red')

            if not self.state_path.exists():
                raise RuntimeError('Encountered broken parameters, but there is no backup to fall back to.')

            trainer.strategy.restore_from_backup()
            return

        self.steps += 1
        self.loss += outputs['loss'].item()
        self.accuracy += outputs['accuracy']

        if (current_step := self.optimizer.local_epoch) != self.last_reported_collaboration_step:
            self.total_samples_processed += self.samples
            self.samples_per_second = self.optimizer.tracker.performance_ema.samples_per_second
            self.lr = self.optimizer.opt.param_groups[0]['lr']
            self.alive_peers = trainer.strategy.num_peers

            self._report_metrics(trainer, step=current_step)

    def _params_are_finite(self):
        for param in self.pl_module.model.parameters():
            if not torch.all(torch.isfinite(param)):
                return False

        return True

    def _report_metrics(self, trainer: pl.Trainer, step: int) -> None:
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

        if self.optimizer.local_epoch == self.optimizer.tracker.global_epoch:
            if not self.dht_manager.dht.store(
                    key=self.optimizer.run_id + "_metrics",
                    subkey=self.dht_manager.local_public_key,
                    value=statistics.dict(),
                    expiration_time=hivemind.get_dht_time() + self.statistics_expiration,
                    return_future=True
            ):
                log.project_console.print('Failed to store metrics', style='red')

        if not self._should_skip_saving_checkpoint(trainer):
            log.project_console.print('Backup collab state', style='magenta')
            trainer.strategy.backup_state()
        else:
            log.project_console.print('Skip backup', style='yellow')

        self.steps = 0
        self.loss = 0
        self.accuracy = 0
        self.samples = self.optimizer.grad_averager.local_samples_accumulated
        self.last_reported_collaboration_step = step

    def _print_metrics(self, step: int) -> None:
        panel_group = Group(Text(f'Local loss: {self.loss / self.steps}',
                                 style='bright_blue', justify='left'),
                            Text(f'Local accuracy: {self.accuracy / self.steps}',
                                 style='bright_blue', justify='left'),
                            Text(f'Learning rate: {self.lr}',
                                 style='bright_blue', justify='left'),
                            Text(f'Min-max noise range: {f"{self.min_noise}-{self.max_noise}"}',
                                 style='bright_blue', justify='left'),
                            Text(f'Your current contribution: {self.total_samples_processed} samples',
                                 style='bright_blue', justify='left'),
                            Text(f'Performance: {self.samples_per_second} samples/sec',
                                 style='bright_blue', justify='left'),
                            Text(f'Peers alive: {self.alive_peers}',
                                 style='bright_blue', justify='left'))

        log.project_console.print(
            Panel(panel_group, title=f'Local step {step}', title_align='left', border_style='magenta'),
            justify='full'
        )

    def _should_skip_saving_checkpoint(self, trainer: pl.Trainer) -> bool:
        from pytorch_lightning.trainer.states import TrainerFn

        return (
                trainer.fast_dev_run  # disable checkpointing with fast_dev_run
                or trainer.state.fn != TrainerFn.FITTING  # don't save anything during non-fit
                or trainer.sanity_checking  # don't save anything during sanity check
                or self.last_reported_collaboration_step == self.optimizer.local_epoch  # already saved at the last step
                or self.backup_every_step is None  # backup is disabled
                or self.optimizer.local_epoch % self.backup_every_step != 0  # not at the current step
        )
