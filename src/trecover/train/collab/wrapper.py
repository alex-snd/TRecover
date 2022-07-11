from pathlib import Path
from time import time
from typing import Dict, Any, Iterable, Optional, Tuple, List, Union

import pytorch_lightning as pl
import torch
from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from trecover.config.log import project_console
from trecover.model import TRecover
from trecover.train.collab.arguments import DataArguments, ModelArguments, PLTrainerArguments
from trecover.train.data import WikiDataset, StandardCollate
from trecover.train.loss import CustomCrossEntropyLoss
from trecover.utils.train import transfer
from trecover.utils.transform import tensor_to_columns, tensor_to_target
from trecover.utils.visualization import visualize_columns, visualize_target


class BaseWrapper(pl.LightningModule):
    def __init__(self,
                 data_args: DataArguments,
                 model_args: ModelArguments,
                 trainer_args: PLTrainerArguments,
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.data_args = data_args
        self.model_args = model_args
        self.trainer_args = trainer_args

        self.model = TRecover(model_args.token_size, model_args.pe_max_len, model_args.n_layers, model_args.d_model,
                              model_args.n_heads, model_args.d_ff, model_args.dropout)
        self.criterion = CustomCrossEntropyLoss(ignore_index=-1)
        self.collate = StandardCollate(min_noise=self.data_args.min_noise, max_noise=self.data_args.max_noise)
        self.batch_size = trainer_args.batch_size

    def forward(self, batch: Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor]
                ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor, Tensor]:
        src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask = batch

        tgt_out = self.model(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
        tgt_out = tgt_out.reshape(-1, self.model.token_size)
        tgt = tgt.view(-1)
        src = src.reshape(-1, self.model.token_size)

        return src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask, tgt_out

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(params=self._get_trainable_params(),
                                lr=self.trainer_args.learning_rate,
                                betas=(self.trainer_args.adam_beta1, self.trainer_args.adam_beta2),
                                eps=self.trainer_args.adam_epsilon,
                                weight_decay=self.trainer_args.weight_decay)

    def _create_dataloader(self, files: Path, dataset_size: int) -> DataLoader:
        files = [files / file for file in files.iterdir()]
        dataset = WikiDataset(datafiles=files, min_threshold=self.data_args.min_threshold,
                              max_threshold=self.data_args.max_threshold, dataset_size=dataset_size)

        return dataset.create_dataloader(batch_size=self.batch_size,
                                         collate=self.collate,
                                         num_workers=self.trainer_args.dataloader_num_workers)

    def _get_trainable_params(self) -> Iterable[Dict[str, Any]]:
        no_decay = ['bias', 'LayerNorm.weight']

        return [
            {
                'params': [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.trainer_args.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
            },
        ]


class TuneWrapper(BaseWrapper):
    def __init__(self,
                 data_args: DataArguments,
                 model_args: ModelArguments,
                 trainer_args: PLTrainerArguments,
                 *args: Any, **kwargs: Any):
        super().__init__(data_args, model_args, trainer_args, *args, **kwargs)

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

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.data_args.val_files,
                                       self.data_args.val_dataset_size)


class PeerWrapper(BaseWrapper):
    def __init__(self,
                 data_args: DataArguments,
                 model_args: ModelArguments,
                 trainer_args: PLTrainerArguments,
                 *args: Any, **kwargs: Any):
        super().__init__(data_args, model_args, trainer_args, *args, **kwargs)

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

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.data_args.train_files,
                                       self.data_args.train_dataset_size)


class FullWrapper(TuneWrapper, PeerWrapper):
    def __init__(self,
                 data_args: DataArguments,
                 model_args: ModelArguments,
                 trainer_args: PLTrainerArguments,
                 *args: Any, **kwargs: Any):
        super(FullWrapper, self).__init__(data_args, model_args, trainer_args, *args, **kwargs)

    def validation_epoch_end(self, step_outputs: Union[Dict[str, Tensor], List[Dict[str, Tensor]]]) -> None:
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

    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.data_args.test_files,
                                       self.data_args.test_dataset_size)

    def vis_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.data_args.vis_files,
                                       self.data_args.vis_dataset_size)
