from argparse import Namespace
from typing import Tuple, Dict, Any, Callable, Optional, Union

import hivemind
import pytorch_lightning as pl
import torch
from hivemind import DHT
from pytorch_lightning.strategies.strategy import Strategy, TBroadcast
from pytorch_lightning.utilities.data import extract_batch_size
from pytorch_lightning.utilities.enums import PrecisionType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from torch import Tensor

from trecover.config.log import project_console
from trecover.train.collab.dht import DHTManager
from trecover.train.collab.optim import create_collab_opt


class CollaborativeStrategy(Strategy):
    def __init__(self,
                 args: Namespace,
                 dht: Optional[hivemind.DHT] = None,
                 use_init_peers: Optional[bool] = None,
                 verbose: Optional[bool] = None,
                 tune: bool = False):
        super().__init__()

        self.args = args
        self.tune_batch_size = args.tune_batch_size_init
        self.use_init_peers = not tune if use_init_peers is None else use_init_peers
        self.verbose = not tune if verbose is None else verbose
        self.tune = tune
        self._dht: Optional[hivemind.DHT] = dht
        self._collab_opt: Optional[hivemind.Optimizer] = None
        self._optimizer_zero_grad_original: Optional[Callable] = None
        self._collab_opt_initialized = False

    @property
    def num_peers(self) -> int:
        if self._collab_opt:
            return self._collab_opt.tracker.global_progress.num_peers
        return 1

    @property
    def root_device(self) -> torch.device:
        from pytorch_lightning.accelerators.cpu import CPUAccelerator
        from pytorch_lightning.accelerators.gpu import GPUAccelerator

        if isinstance(self.accelerator, GPUAccelerator):
            return torch.device(f'cuda:{torch.cuda.current_device()}')
        elif isinstance(self.accelerator, CPUAccelerator):
            return torch.device('cpu')
        raise MisconfigurationException(
            f'Was unable to infer device type from the accelerator: {self.accelerator.__class__.__name__}.'
        )

    @property
    def global_rank(self) -> int:
        return 0

    @property
    def is_global_zero(self) -> bool:
        return True

    @property
    def dht(self) -> DHT:
        if self._dht is None:
            self._dht = DHTManager(self.args, self.use_init_peers).dht

        return self._dht

    def setup(self, trainer: pl.Trainer) -> None:
        self.model_to_device()
        super().setup(trainer)

        if self.precision_plugin.precision in (PrecisionType.HALF, PrecisionType.MIXED):
            self.precision_plugin.scaler = hivemind.GradScaler()

    def _init_collab_opt(self) -> None:
        assert len(self.optimizers) == 1, 'Hivemind only supports training with one optimizer.'

        if self.args.batch_size is None:
            batch_size_per_step = self.tune_batch_size
        else:
            batch_size_per_step = self.args.batch_size * self.args.accumulate_batches

        self._collab_opt = create_collab_opt(wrapped_optimizer=self.model.wrapped_optimizer,
                                             params=self.model.trainable_params,
                                             dht=self.dht,
                                             args=self.args,
                                             wrapped_scheduler=self.model.wrapped_scheduler,
                                             auxiliary=False,
                                             verbose=self.verbose,
                                             batch_size_per_step=batch_size_per_step)

        self.optimizers = [self._collab_opt]
        self._collab_opt_initialized = True

        if not self.tune:
            self.restore_from_backup()

            project_console.print('Sync with other peers', style='magenta')
            self._collab_opt.load_state_from_peers()

            if not self.args.state_path.exists():
                project_console.print('Backup the collab state as it does not exist', style='magenta')
                self.backup_state()

        if not self.args.no_reuse_grad_buffers:
            assert self.lightning_module is not None
            self._optimizer_zero_grad_original = self.lightning_module.optimizer_zero_grad
            self._disable_zero_grad()

    def _disable_zero_grad(self) -> None:
        assert self.lightning_module is not None

        if is_overridden('optimizer_zero_grad', self.lightning_module):
            project_console.print(
                'You have overridden `optimizer_zero_grad` which will be disabled. '
                'When `CollaborativeStrategy(reuse_grad_buffers=True)`, the optimizer cannot call zero grad, '
                'as this would delete the gradients before they are averaged.',
                style='yellow'
            )

        self.lightning_module.optimizer_zero_grad = None

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor],
                      *args, **kwargs
                      ) -> Dict[str, Tensor]:
        if not self._collab_opt_initialized:
            if self.args.batch_size is None:
                self.tune_batch_size = extract_batch_size(batch)

            self._init_collab_opt()

        with self.precision_plugin.train_step_context():
            return self.model.training_step(batch, *args, **kwargs)

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor],
                        *args, **kwargs
                        ) -> Dict[str, Tensor]:
        if not self._collab_opt_initialized:
            if self.args.batch_size is None:
                self.tune_batch_size = extract_batch_size(batch)

            self._init_collab_opt()

        with self.precision_plugin.val_step_context():
            return self.model.validation_step(batch, *args, **kwargs)

    def reduce(self, tensor: Union[Any, Tensor], *args: Any, **kwargs: Any) -> Union[Any, Tensor]:
        return tensor

    def all_gather(self, tensor: Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> Tensor:
        return tensor

    def model_to_device(self) -> None:
        assert self.model is not None
        self.model.to(self.root_device)

    def barrier(self, *args: Any, **kwargs: Any) -> None:
        pass

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        return obj

    def state_dict(self) -> Dict[str, Any]:
        if self._collab_opt:
            return {
                'model': self.model.model.state_dict(),
                'optimizer': self._collab_opt.state_dict(),
                'scheduler': self._collab_opt.state_averager.scheduler.state_dict(),
                'local_epoch': self._collab_opt.local_epoch,
            }
        else:
            return {'model': self.model.model.state_dict(), 'optimizer': {}, 'scheduler': {}, 'local_epoch': {}}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.model.load_state_dict(state_dict['model'])

        if self._collab_opt:
            self._collab_opt.load_state_dict(state_dict['optimizer'])
            self._collab_opt.state_averager.scheduler.load_state_dict(state_dict['scheduler'])
            self._collab_opt.state_averager.local_epoch = state_dict['local_epoch']

            if self._collab_opt.offload_optimizer:
                state_averager = self._collab_opt.state_averager
                offloaded_parameters = [
                    param for group in state_averager.optimizer.param_groups for param in group['params']
                ]

                assert len(offloaded_parameters) == len(state_averager.main_parameters)

                for main_param, offloaded_param in zip(state_averager.main_parameters, offloaded_parameters):
                    offloaded_param.copy_(main_param, non_blocking=True)

    @torch.no_grad()
    def backup_state(self) -> None:
        torch.save(self.state_dict(), self.args.state_path)

    @torch.no_grad()
    def restore_from_backup(self, check_step: bool = False) -> None:
        if self.args.state_path.exists():
            state_dict = torch.load(self.args.state_path)
            current_step = self._collab_opt.local_epoch
            backup_step = state_dict['local_epoch']

            if not check_step or backup_step >= current_step:
                self.load_state_dict(state_dict)
                project_console.print('CollaborativeStrategy: Collab sate is restored from backup.', style='green')

            else:
                project_console.print(
                    'CollaborativeStrategy: Bypassed restoring collab state '
                    'from local backup - backup state is too old.',
                    style='yellow'
                )

        else:
            project_console.print(
                'CollaborativeStrategy: Backup does not exist.',
                style='yellow'
            )

    def teardown(self) -> None:
        # re-enable `optimizer_zero_grad`
        if self._optimizer_zero_grad_original is not None and self.lightning_module is not None:
            self.lightning_module.optimizer_zero_grad = self._optimizer_zero_grad_original

        if self._collab_opt:
            if self.verbose:
                project_console.print('Shutting down hivemind optimizer', style='yellow')

            self._collab_opt.shutdown()
            self._collab_opt = None
            self._collab_opt_initialized = False

        if self._dht:
            if self.verbose:
                project_console.print('Shutting down hivemind DHT', style='yellow')

            self._dht.shutdown()
            self._dht = None

        super().teardown()
