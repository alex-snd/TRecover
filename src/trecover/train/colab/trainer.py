from dataclasses import asdict
from pathlib import Path
from time import time
from typing import Dict, Any, Iterable, Callable, Optional, Tuple, List, Union
from os import getpid

import hivemind
import pytorch_lightning as pl
import torch
from hivemind import SizeAdaptiveCompression, Float16Compression, Uniform8BitQuantization
from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from trecover.config.log import project_logger, project_console
from trecover.model import TRecover
from trecover.train.colab import utils
from trecover.train.colab.arguments import (DataArguments, ModelArguments, PLTrainerArguments, BasePeerArguments,
                                            CollaborativeArguments)
from trecover.train.data import WikiDataset, StandardCollate
from trecover.train.loss import CustomCrossEntropyLoss
from trecover.utils.train import transfer
from trecover.utils.transform import tensor_to_columns, tensor_to_target
from trecover.utils.visualization import visualize_columns, visualize_target


class Task:
    def __init__(self,
                 data_args: DataArguments,
                 model_args: ModelArguments,
                 peer_args: BasePeerArguments,
                 trainer_args: PLTrainerArguments,
                 collab_args: CollaborativeArguments):
        self.data_args = data_args
        self.model_args = model_args
        self.peer_args = peer_args
        self.trainer_args = trainer_args
        self.collab_args = collab_args

        self._dht = None
        self._optimizer = None
        self.validators, self.local_public_key = utils.make_validators(peer_args.experiment_prefix)
        self.model = TRecover(model_args.token_size, model_args.pe_max_len, model_args.n_layers, model_args.d_model,
                              model_args.n_heads, model_args.d_ff, model_args.dropout)

    @property
    def dht(self) -> hivemind.DHT:
        if self._dht is None:
            self._dht = hivemind.DHT(
                start=True,
                initial_peers=self.peer_args.initial_peers,
                client_mode=self.peer_args.client_mode,
                host_maddrs=self.peer_args.host_maddrs,
                announce_maddrs=self.peer_args.announce_maddrs,
                use_ipfs=self.peer_args.use_ipfs,
                record_validators=self.validators,
                identity_path=self.peer_args.identity_path,
            )

            if self.peer_args.client_mode:
                project_logger.info(f'Created client mode peer with peer_id={self._dht.peer_id}')
            else:
                visible_maddrs = self._dht.get_visible_maddrs()
                initial_peers = utils.get_initial_peers(visible_maddrs, only_p2p=self.peer_args.use_ipfs)

                if initial_peers:
                    project_logger.info(f'To connect other peers to this one over the Internet, use '
                                        f'--initial_peers {initial_peers}')

                project_logger.info(f'Full list of visible multi addresses: {[str(addr) for addr in visible_maddrs]}')

        return self._dht

    @property
    def optimizer(self) -> hivemind.Optimizer:
        params = self._get_trainable_params()
        optimizer = self._get_local_optimizer()
        scheduler = self._get_local_scheduler()

        averaging_compression = SizeAdaptiveCompression(
            threshold=2 ** 16 + 1, less=Float16Compression(), greater_equal=Uniform8BitQuantization())

        print(f'Create New Optimizer for {getpid()} process')

        self._optimizer = hivemind.Optimizer(dht=self.dht,
                                             run_id=self.peer_args.experiment_prefix,
                                             params=params,
                                             optimizer=optimizer,
                                             scheduler=scheduler,
                                             offload_optimizer=True,
                                             delay_grad_averaging=False,
                                             delay_optimizer_step=True,
                                             batch_size_per_step=self.trainer_args.batch_size_per_step,
                                             grad_compression=averaging_compression,
                                             state_averaging_compression=averaging_compression,
                                             client_mode=self.peer_args.client_mode,
                                             verbose=True,
                                             **asdict(self.collab_args))
        print(f'Optimizer Created for {getpid()} process')
        return self._optimizer

    def _get_trainable_params(self) -> Iterable[Dict[str, Any]]:
        no_decay = ["bias", "LayerNorm.weight"]

        return [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.trainer_args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

    def _get_local_optimizer(self) -> Callable[[Iterable[Dict[str, Any]], ], Optimizer]:
        def optimizer(params: Iterable[Dict[str, Any]]) -> Optimizer:
            return torch.optim.Adam(params=params,
                                    lr=self.trainer_args.learning_rate,
                                    betas=(self.trainer_args.adam_beta1, self.trainer_args.adam_beta2),
                                    eps=self.trainer_args.adam_epsilon,
                                    weight_decay=self.trainer_args.weight_decay)

        return optimizer

    def _get_local_scheduler(self) -> Callable[[Optimizer, ], LambdaLR]:
        def scheduler(optimizer: Optimizer) -> LambdaLR:
            return utils.get_linear_schedule_with_warmup(optimizer=optimizer,
                                                         num_warmup_steps=self.trainer_args.warmup_steps,
                                                         num_training_steps=self.trainer_args.total_steps)

        return scheduler

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int) -> None:
        if not self.collab_args.reuse_grad_buffers:
            optimizer.zero_grad()


class LightningWrapper(pl.LightningModule):
    def __init__(self,
                 data_args: DataArguments,
                 model_args: ModelArguments,
                 peer_args: BasePeerArguments,
                 trainer_args: PLTrainerArguments,
                 collab_args: CollaborativeArguments,
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.data_args = data_args
        self.model_args = model_args
        self.peer_args = peer_args
        self.trainer_args = trainer_args
        self.collab_args = collab_args

        self._dht = None
        self._optimizer = None
        self.validators, self.local_public_key = utils.make_validators(peer_args.experiment_prefix)
        self.model = TRecover(model_args.token_size, model_args.pe_max_len, model_args.n_layers, model_args.d_model,
                              model_args.n_heads, model_args.d_ff, model_args.dropout)
        self.criterion = CustomCrossEntropyLoss(ignore_index=-1)
        self.collate = StandardCollate(min_noise=self.data_args.min_noise, max_noise=self.data_args.max_noise)
        self.batch_size = trainer_args.batch_size

    def on_train_start(self) -> None:
        project_console.print(f'Train process pid: {getpid()}', style='green')
        project_console.print('TRAIN STARTED', style='green')
        project_console.print(f'DHT: {self.task.dht.is_alive()}', style='red')
        # project_console.print(f'Opt: {self.task.optimizer.is_alive()}', style='red')
        # self.task.optimizer.is_alive()

    def forward(self, batch: Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor]
                ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor, Tensor]:
        src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask = batch

        tgt_out = self.model(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
        tgt_out = tgt_out.reshape(-1, self.model.token_size)
        tgt = tgt.view(-1)
        src = src.reshape(-1, self.model.token_size)

        project_console.print(f'Forward process pid: {getpid()}', style='green')

        return src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask, tgt_out

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor],
                      *args, **kwargs
                      ) -> Dict[str, Tensor]:
        start_time = time()

        src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask, tgt_out = self(batch)
        loss = self.criterion(src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask, tgt_out)
        accuracy = (torch.argmax(tgt_out, dim=1) == tgt).float().sum().item() / tgt.size(0)

        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_time': time() - start_time},
                      batch_size=self.batch_size)

        return {'loss': loss, 'accuracy': accuracy}

    def training_epoch_end(self, step_outputs: Union[Dict[str, Tensor], List[Dict[str, Tensor]]]) -> None:
        if isinstance(step_outputs, list):
            self.log_dict({
                'train_epoch_loss': torch.tensor([output['loss'] for output in step_outputs]).mean(),
                'train_epoch_accuracy': torch.tensor([output['accuracy'] for output in step_outputs]).mean(),
            },
                batch_size=self.batch_size)
        else:
            self.log_dict({'train_epoch_loss': step_outputs['loss'], 'train_epoch_accuracy': step_outputs['accuracy']},
                          batch_size=self.batch_size)

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor],
                        *args, **kwargs
                        ) -> Dict[str, Tensor]:
        start_time = time()

        src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask, tgt_out = self(batch)
        loss = self.criterion(src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask, tgt_out).item()
        accuracy = ((torch.argmax(tgt_out, dim=1) == tgt).float().sum() / tgt.size(0)).item()

        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy, 'val_time': time() - start_time},
                      batch_size=self.batch_size)

        return {'loss': loss, 'accuracy': accuracy}

    def validation_epoch_end(self, step_outputs: Union[Dict[str, Tensor], List[Dict[str, Tensor]]]) -> None:
        if isinstance(step_outputs, list):
            self.log_dict({
                'val_epoch_loss': torch.tensor([output['loss'] for output in step_outputs]).mean(),
                'val_epoch_accuracy': torch.tensor([output['accuracy'] for output in step_outputs]).mean(),
            },
                batch_size=self.batch_size)
        else:
            self.log_dict({'val_epoch_loss': step_outputs['loss'], 'val_epoch_accuracy': step_outputs['accuracy']},
                          batch_size=self.batch_size)

        self.visualize_on_epoch_end()

    @torch.no_grad()
    def visualize_on_epoch_end(self) -> None:
        for batch_idx, vis_tensors in enumerate(self.vis_dataloader(), start=1):
            src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask = transfer(vis_tensors, to_device=self.device)

            tgt_out = self.model(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
            tgt_out = tgt_out.reshape(-1, self.model.token_size)
            prediction = torch.argmax(tgt_out, dim=1).view_as(tgt)

            for i in range(src.shape[0]):
                columns = tensor_to_columns(src[i, : self.trainer_args.n_columns_to_show])
                columns = visualize_columns(columns, delimiter=self.trainer_args.delimiter, as_rows=True)
                columns = (Text(row, style='bright_blue', overflow='ellipsis', no_wrap=True) for row in columns)
                target = tensor_to_target(prediction[i, : self.trainer_args.n_columns_to_show])
                predicted = visualize_target(target, delimiter=self.trainer_args.delimiter)
                original = tensor_to_target(tgt[i, : self.trainer_args.n_columns_to_show])
                original = visualize_target(original, delimiter=self.trainer_args.delimiter)

                panel_group = Group(
                    Text('Columns', style='magenta', justify='center'),
                    *columns,
                    Text('Predicted', style='magenta', justify='center'),
                    Text(predicted, style='cyan', justify='center', overflow='ellipsis'),
                    Text('Original', style='magenta', justify='center'),
                    Text(original, justify='center', overflow='ellipsis')
                )

                project_console.print('\n')
                project_console.print(
                    Panel(panel_group, title=f'Example {(batch_idx - 1) * self.batch_size + i + 1}',
                          border_style='magenta'),
                    justify='center'
                )

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor],
                  *args, **kwargs
                  ) -> Dict[str, Tensor]:
        start_time = time()

        src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask, tgt_out = self(batch)
        loss = self.criterion(src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask, tgt_out).item()
        accuracy = ((torch.argmax(tgt_out, dim=1) == tgt).float().sum() / tgt.size(0)).item()

        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy, 'test_time': time() - start_time},
                      batch_size=self.batch_size)

        return {'loss': loss, 'accuracy': accuracy}

    def test_epoch_end(self, step_outputs: Union[Dict[str, Tensor], List[Dict[str, Tensor]]]) -> None:
        if isinstance(step_outputs, list):
            self.log_dict({
                'test_epoch_loss': torch.tensor([output['loss'] for output in step_outputs]).mean(),
                'test_epoch_accuracy': torch.tensor([output['accuracy'] for output in step_outputs]).mean(),
            },
                batch_size=self.batch_size)
        else:
            self.log_dict({'test_epoch_loss': step_outputs['loss'], 'test_epoch_accuracy': step_outputs['accuracy']},
                          batch_size=self.batch_size)

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.data_args.train_files,
                                       self.data_args.train_dataset_size)

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.data_args.val_files,
                                       self.data_args.val_dataset_size)

    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.data_args.test_files,
                                       self.data_args.test_dataset_size)

    def vis_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.data_args.vis_files,
                                       self.data_args.vis_dataset_size)

    def _create_dataloader(self, files: Path, dataset_size: int) -> DataLoader:
        files = [files / file for file in files.iterdir()]
        dataset = WikiDataset(datafiles=files, min_threshold=self.data_args.min_threshold,
                              max_threshold=self.data_args.max_threshold, dataset_size=dataset_size)

        return dataset.create_dataloader(batch_size=self.batch_size,
                                         collate=self.collate,
                                         num_workers=self.trainer_args.dataloader_num_workers)

    @property
    def dht(self) -> hivemind.DHT:
        if self._dht is None:
            self._dht = hivemind.DHT(
                start=True,
                initial_peers=self.peer_args.initial_peers,
                client_mode=self.peer_args.client_mode,
                host_maddrs=self.peer_args.host_maddrs,
                announce_maddrs=self.peer_args.announce_maddrs,
                use_ipfs=self.peer_args.use_ipfs,
                record_validators=self.validators,
                identity_path=self.peer_args.identity_path,
            )

            if self.peer_args.client_mode:
                project_logger.info(f'Created client mode peer with peer_id={self._dht.peer_id}')
            else:
                visible_maddrs = self._dht.get_visible_maddrs()
                initial_peers = utils.get_initial_peers(visible_maddrs, only_p2p=self.peer_args.use_ipfs)

                if initial_peers:
                    project_logger.info(f'To connect other peers to this one over the Internet, use '
                                        f'--initial_peers {initial_peers}')

                project_logger.info(f'Full list of visible multi addresses: {[str(addr) for addr in visible_maddrs]}')

        return self._dht

    def configure_optimizers(self) -> Optimizer:
        params = self._get_trainable_params()
        optimizer = self._get_local_optimizer()
        scheduler = self._get_local_scheduler()

        averaging_compression = SizeAdaptiveCompression(
            threshold=2 ** 16 + 1, less=Float16Compression(), greater_equal=Uniform8BitQuantization())

        self._optimizer = hivemind.Optimizer(dht=self.dht,
                                             run_id=self.peer_args.experiment_prefix,
                                             params=params,
                                             optimizer=optimizer,
                                             scheduler=scheduler,
                                             offload_optimizer=True,
                                             delay_grad_averaging=False,
                                             delay_optimizer_step=True,
                                             batch_size_per_step=self.trainer_args.batch_size_per_step,
                                             grad_compression=averaging_compression,
                                             state_averaging_compression=averaging_compression,
                                             client_mode=self.peer_args.client_mode,
                                             verbose=True,
                                             **asdict(self.collab_args))
        return self._optimizer

        # return self.task.optimizer

    def _get_trainable_params(self) -> Iterable[Dict[str, Any]]:
        no_decay = ["bias", "LayerNorm.weight"]

        return [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.trainer_args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

    def _get_local_optimizer(self) -> Callable[[Iterable[Dict[str, Any]], ], Optimizer]:
        def optimizer(params: Iterable[Dict[str, Any]]) -> Optimizer:
            return torch.optim.Adam(params=params,
                                    lr=self.trainer_args.learning_rate,
                                    betas=(self.trainer_args.adam_beta1, self.trainer_args.adam_beta2),
                                    eps=self.trainer_args.adam_epsilon,
                                    weight_decay=self.trainer_args.weight_decay)

        return optimizer

    def _get_local_scheduler(self) -> Callable[[Optimizer, ], LambdaLR]:
        def scheduler(optimizer: Optimizer) -> LambdaLR:
            return utils.get_linear_schedule_with_warmup(optimizer=optimizer,
                                                         num_warmup_steps=self.trainer_args.warmup_steps,
                                                         num_training_steps=self.trainer_args.total_steps)

        return scheduler

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int) -> None:
        if not self.collab_args.reuse_grad_buffers:
            optimizer.zero_grad()
