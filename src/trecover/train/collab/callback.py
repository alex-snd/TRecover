from typing import Optional, Dict, Any

import hivemind
import pytorch_lightning as pl
import torch.nn
from pytorch_lightning.callbacks.base import Callback

from trecover.config import log
from trecover.train.collab.arguments import TrainingPeerArguments
from trecover.train.collab.dht import DHTManager, LocalMetrics


class CollabCheckpoint(Callback):
    def __init__(self, dht_manager: DHTManager, peer_args: TrainingPeerArguments):
        self.dht_manager: DHTManager = dht_manager
        self.pl_module: Optional[pl.LightningModule] = None
        self.optimizer: Optional[hivemind.Optimizer] = None

        self.statistics_expiration = peer_args.statistics_expiration
        self.last_reported_collaboration_step = -1
        self.samples = 0
        self.steps = 0
        self.loss = 0
        self.accuracy = 0
        self.total_samples_processed = 0
        self.backup_every_step = peer_args.backup_every_step
        self.state_path = peer_args.state_path

    def params_are_finite(self):
        for param in self.pl_module.model.parameters():
            if not torch.all(torch.isfinite(param)):
                return False

        return True

    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: Dict[str, Any],
                           *args, **kwargs
                           ) -> None:
        if self.optimizer is None:
            assert len(trainer.strategy.optimizers) == 1, 'Hivemind only supports training with one optimizer.'
            self.optimizer = trainer.strategy.optimizers[0]

        if self.pl_module is None:
            self.pl_module = pl_module

        if not self.params_are_finite():
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
            samples_per_second = self.optimizer.tracker.performance_ema.samples_per_second

            statistics = LocalMetrics(
                step=current_step,
                samples_per_second=samples_per_second,
                samples_accumulated=self.samples,
                loss=self.loss,
                accuracy=self.accuracy,
                mini_steps=self.steps,
            )

            log.project_console.print(f'Current step: {current_step}')
            log.project_console.print(f'Your current contribution: {self.total_samples_processed} samples')
            log.project_console.print(f'Performance: {samples_per_second} samples/sec')
            log.project_console.print(f'Local loss: {self.loss / self.steps}')
            log.project_console.print(f'Local accuracy: {self.accuracy / self.steps}')

            self.loss = 0
            self.steps = 0

            if self.optimizer.local_epoch == self.optimizer.tracker.global_epoch:
                self.dht_manager.dht.store(
                    key=self.optimizer.run_id + "_metrics",
                    subkey=self.dht_manager.local_public_key,
                    value=statistics.dict(),
                    expiration_time=hivemind.get_dht_time() + self.statistics_expiration,
                    return_future=True,
                )

            if not self._should_skip_saving_checkpoint(trainer):
                log.project_console.print('Backup collab state', style='magenta')
                trainer.strategy.backup_state()
            else:
                log.project_console.print('Backup failed', style='red')

            self.last_reported_collaboration_step = current_step

        self.samples = self.optimizer.grad_averager.local_samples_accumulated

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
