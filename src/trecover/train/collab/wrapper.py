import os
from argparse import Namespace
from pathlib import Path
from time import time
from typing import List, Dict, Any, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from trecover.config import log
from trecover.model import TRecover
from trecover.train.data import WikiDataset, BaseCollate, StandardCollate, CollabCollate
from trecover.train.loss import CustomCrossEntropyLoss
from trecover.utils.train import transfer
from trecover.utils.transform import tensor_to_columns, tensor_to_target


class BaseModelWrapper(pl.LightningModule):
    def __init__(self, args: Namespace, *pl_args: Any, **pl_kwargs: Any):
        super(BaseModelWrapper, self).__init__(*pl_args, **pl_kwargs)

        self.args = args
        self.model = TRecover(args.token_size, args.pe_max_len, args.n_layers, args.d_model,
                              args.n_heads, args.d_ff, args.dropout)
        self.criterion = CustomCrossEntropyLoss(ignore_index=-1)
        self.batch_size = args.batch_size
        self._collate = None

    @property
    def collate(self) -> BaseCollate:
        if self._collate is None:
            if self.args.sync_args:
                self._collate = CollabCollate()
            else:
                self._collate = StandardCollate(min_noise=self.args.min_noise, max_noise=self.args.max_noise)

        return self._collate

    def forward(self, batch: Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor]
                ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor, Tensor]:
        src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask = batch

        tgt_out = self.model(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
        tgt_out = tgt_out.reshape(-1, self.model.token_size)
        tgt = tgt.view(-1)
        src = src.reshape(-1, self.model.token_size)

        return src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask, tgt_out

    def configure_optimizers(self) -> Optimizer:  # fictive optimizer
        return torch.optim.Adam(params=self.model.parameters(),
                                lr=self.args.lr,
                                betas=(self.args.adam_beta1, self.args.adam_beta2),
                                eps=self.args.adam_epsilon,
                                weight_decay=self.args.weight_decay)

    @torch.no_grad()
    def perform(self) -> List[Tuple[List[str], List[str], List[str]]]:
        performance = list()

        for batch_idx, vis_tensors in enumerate(self.performance_dataloader(), start=1):
            src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask = transfer(vis_tensors, to_device=self.device)

            tgt_out = self.model(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
            tgt_out = tgt_out.reshape(-1, self.model.token_size)
            prediction = torch.argmax(tgt_out, dim=1).view_as(tgt)

            for i in range(src.shape[0]):
                columns = tensor_to_columns(src[i, :])
                predicted = tensor_to_target(prediction[i, :])
                original = tensor_to_target(tgt[i, :])

                performance.append((columns, predicted, original))

        return performance

    def performance_dataloader(self) -> DataLoader:
        return self._create_dataloader(files=self.args.vis_files,
                                       dataset_size=self.args.vis_dataset_size,
                                       batch_size=self.batch_size or 1,
                                       num_workers=self.args.n_workers)

    def _create_dataloader(self, files: Path, dataset_size: int, batch_size: int, num_workers: int) -> DataLoader:
        files = [files / file for file in files.iterdir()]
        dataset = WikiDataset(datafiles=files, min_threshold=self.args.min_threshold,
                              max_threshold=self.args.max_threshold, dataset_size=dataset_size)

        return dataset.create_dataloader(batch_size=batch_size,
                                         collate=self.collate,
                                         num_workers=num_workers)


class PeerModelWrapper(BaseModelWrapper):
    def __init__(self, args: Namespace, *pl_args: Any, **pl_kwargs: Any):
        super(PeerModelWrapper, self).__init__(args, *pl_args, **pl_kwargs)

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
        return self._create_dataloader(self.args.train_files, self.args.train_dataset_size, self.batch_size,
                                       self.args.n_workers)

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.args.val_files, self.args.val_dataset_size, self.batch_size,
                                       self.args.n_workers)


class FullModelWrapper(PeerModelWrapper):
    def __init__(self, args: Namespace, *pl_args: Any, **pl_kwargs: Any):
        super(FullModelWrapper, self).__init__(args, *pl_args, **pl_kwargs)

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
        return self._create_dataloader(self.args.test_files, self.args.test_dataset_size, self.batch_size,
                                       self.args.n_workers)
