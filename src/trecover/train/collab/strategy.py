from argparse import Namespace
from typing import Tuple, Dict, Any, Callable, Optional, Union

import hivemind
import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies.strategy import Strategy, TBroadcast
from pytorch_lightning.utilities.data import extract_batch_size
from pytorch_lightning.utilities.enums import PrecisionType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from torch import Tensor

from trecover.config import log
from trecover.train.collab.dht import DHTManager
from trecover.train.collab.optim import CollaborativeOptimizer


class CollaborativeStrategy(Strategy):
    def __init__(self,
                 args: Namespace,
                 dht_manager: Optional[DHTManager] = None,
                 use_init_peers: Optional[bool] = None,
                 verbose: Optional[bool] = None,
                 tune: bool = False):
        super().__init__()

        self.args = args
        self.tune_batch_size = args.tune_batch_size_init
        self.use_init_peers = not tune if use_init_peers is None else use_init_peers
        self.verbose = not tune if verbose is None else verbose
        self.tune = tune
        self._dht_manager: Optional[DHTManager] = dht_manager
        self._collab_opt: Optional[CollaborativeOptimizer] = None
        self._optimizer_zero_grad_original: Optional[Callable] = None
        self._collab_opt_initialized = False

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
    def dht_manager(self) -> DHTManager:
        if self._dht_manager is None:
            self._dht_manager = DHTManager(self.args, self.use_init_peers)

        return self._dht_manager

    @property
    def collab_opt(self) -> CollaborativeOptimizer:
        assert self._collab_opt is not None, 'Collaborative optimizer is not initializer yet'

        return self._collab_opt

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

        self._collab_opt = CollaborativeOptimizer(dht_manager=self.dht_manager,
                                                  wrapped_model=self.lightning_module,
                                                  args=self.args,
                                                  batch_size_per_step=batch_size_per_step,
                                                  verbose=self.verbose)

        self.optimizers = [self._collab_opt.opt]
        self._collab_opt_initialized = True

        if not self.tune:
            self._collab_opt.restore_from_backup()
            if self.collab_opt.local_epoch < self.collab_opt.global_epoch:
                self._collab_opt.sync_state()
            self._collab_opt.backup_state()

        if not self.args.no_reuse_grad_buffers:
            assert self.lightning_module is not None
            self._optimizer_zero_grad_original = self.lightning_module.optimizer_zero_grad
            self._disable_zero_grad()

    def _disable_zero_grad(self) -> None:
        assert self.lightning_module is not None

        if is_overridden('optimizer_zero_grad', self.lightning_module):
            log.project_console.print(
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

    def teardown(self) -> None:
        # re-enable `optimizer_zero_grad`
        if self._optimizer_zero_grad_original is not None and self.lightning_module is not None:
            self.lightning_module.optimizer_zero_grad = self._optimizer_zero_grad_original

        if self._collab_opt:
            if self.verbose:
                log.project_console.print('Shutting down hivemind optimizer', style='yellow')

            self._collab_opt.opt.shutdown()
            self._collab_opt = None
            self._collab_opt_initialized = False

        if self._dht_manager:
            if self.verbose:
                log.project_console.print('Shutting down hivemind DHT', style='yellow')

            self._dht_manager.dht.shutdown()
            self._dht_manager = None

        super().teardown()
