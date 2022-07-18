from argparse import Namespace
from pathlib import Path
from time import time
from typing import Dict, Any, Iterable, Optional, Tuple, List, Union, Callable

import pytorch_lightning as pl
import torch
from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from trecover.config import log
from trecover.model import TRecover
from trecover.train.collab.optim import CPULamb8Bit
from trecover.train.data import WikiDataset, StandardCollate
from trecover.train.loss import CustomCrossEntropyLoss
from trecover.train.scheduler import get_linear_scheduler_with_warmup
from trecover.utils.train import transfer
from trecover.utils.transform import tensor_to_columns, tensor_to_target
from trecover.utils.visualization import visualize_columns, visualize_target


class BaseModelWrapper(pl.LightningModule):
    def __init__(self, args: Namespace, *pl_args: Any, **pl_kwargs: Any):
        super(BaseModelWrapper, self).__init__(*pl_args, **pl_kwargs)

        self.args = args
        self.model = TRecover(args.token_size, args.pe_max_len, args.n_layers, args.d_model,
                              args.n_heads, args.d_ff, args.dropout)
        self.criterion = CustomCrossEntropyLoss(ignore_index=-1)
        self.collate = StandardCollate(min_noise=args.min_noise, max_noise=args.max_noise)
        self.batch_size = args.batch_size

    def forward(self, batch: Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor]
                ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor, Tensor]:
        src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask = batch

        tgt_out = self.model(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
        tgt_out = tgt_out.reshape(-1, self.model.token_size)
        tgt = tgt.view(-1)
        src = src.reshape(-1, self.model.token_size)

        return src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask, tgt_out

    def configure_optimizers(self) -> Optimizer:  # fictive optimizer
        return torch.optim.Adam(params=self.trainable_params,
                                lr=self.args.lr,
                                betas=(self.args.adam_beta1, self.args.adam_beta2),
                                eps=self.args.adam_epsilon,
                                weight_decay=self.args.weight_decay)

    @property
    def wrapped_optimizer(self) -> Callable[[Iterable[Dict[str, Any]]], Optimizer]:
        def optimizer(params: Iterable[Dict[str, Any]]) -> Optimizer:
            return CPULamb8Bit(params=params,  # TODO params
                               lr=self.args.lr,
                               betas=(self.args.adam_beta1, self.args.adam_beta2),
                               eps=self.args.adam_epsilon,
                               weight_decay=self.args.weight_decay,
                               reuse_grad_buffers=not self.args.no_reuse_grad_buffers,
                               bias_correction=True)

        return optimizer

    @property
    def wrapped_scheduler(self) -> Callable[[Optimizer, ], LambdaLR]:
        def scheduler(optimizer: Optimizer) -> LambdaLR:
            return get_linear_scheduler_with_warmup(optimizer=optimizer,
                                                    warmup_steps=self.args.warmup_steps,
                                                    total_steps=self.args.total_steps)

        return scheduler

    @property
    def trainable_params(self) -> Iterable[Dict[str, Any]]:
        no_decay = ['bias', 'LayerNorm.weight']

        return [
            {
                'params': [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.args.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
            },
        ]

    def _create_dataloader(self, files: Path, dataset_size: int) -> DataLoader:
        files = [files / file for file in files.iterdir()]
        dataset = WikiDataset(datafiles=files, min_threshold=self.args.min_threshold,
                              max_threshold=self.args.max_threshold, dataset_size=dataset_size)

        return dataset.create_dataloader(batch_size=self.batch_size,
                                         collate=self.collate,
                                         num_workers=self.args.n_workers)


class PeerModelWrapper(BaseModelWrapper):
    def __init__(self, params: Namespace, *pl_args: Any, **pl_kwargs: Any):
        super(PeerModelWrapper, self).__init__(params, *pl_args, **pl_kwargs)

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

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.args.train_files, self.args.train_dataset_size)

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.args.val_files, self.args.val_dataset_size)


class FullModelWrapper(PeerModelWrapper):
    def __init__(self, params: Namespace, *pl_args: Any, **pl_kwargs: Any):
        super(FullModelWrapper, self).__init__(params, *pl_args, **pl_kwargs)

    def validation_epoch_end(self, step_outputs: Union[Dict[str, Tensor], List[Dict[str, Tensor]]]) -> None:
        self.visualize_on_epoch_end()

    # TODO move out
    @torch.no_grad()
    def visualize_on_epoch_end(self) -> None:
        for batch_idx, vis_tensors in enumerate(self.vis_dataloader(), start=1):
            src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask = transfer(vis_tensors, to_device=self.device)

            tgt_out = self.model(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
            tgt_out = tgt_out.reshape(-1, self.model.token_size)
            prediction = torch.argmax(tgt_out, dim=1).view_as(tgt)

            for i in range(src.shape[0]):
                columns = tensor_to_columns(src[i, : self.args.n_columns_to_show])
                columns = visualize_columns(columns, delimiter=self.args.delimiter, as_rows=True)
                columns = (Text(row, style='bright_blue', overflow='ellipsis', no_wrap=True) for row in columns)
                target = tensor_to_target(prediction[i, : self.args.n_columns_to_show])
                predicted = visualize_target(target, delimiter=self.args.delimiter)
                original = tensor_to_target(tgt[i, : self.args.n_columns_to_show])
                original = visualize_target(original, delimiter=self.args.delimiter)

                panel_group = Group(
                    Text('Columns', style='magenta', justify='center'),
                    *columns,
                    Text('Predicted', style='magenta', justify='center'),
                    Text(predicted, style='cyan', justify='center', overflow='ellipsis'),
                    Text('Original', style='magenta', justify='center'),
                    Text(original, justify='center', overflow='ellipsis')
                )

                log.project_console.print('\n')
                log.project_console.print(
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

    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.args.test_files, self.args.test_dataset_size)

    def vis_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.args.vis_files, self.args.vis_dataset_size)
