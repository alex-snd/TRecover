import shutil
import tempfile
import traceback
from pathlib import Path
from time import time
from typing import Callable, Optional, Tuple, Union, Type

import torch
from rich.console import Group, Console
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.text import Text
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from trecover.config import log
from trecover.model import TRecover
from trecover.train.monitor import BaseMonitor, IdentityMonitor
from trecover.train.scheduler import BaseScheduler, IdentityScheduler
from trecover.utils.model import save_params
from trecover.utils.train import ExperimentParams, optimizer_to_str, transfer
from trecover.utils.transform import tensor_to_columns, tensor_to_target
from trecover.utils.visualization import visualize_columns, visualize_target


# TODO docs after RemoteTrainer implementation using pytorch-lightning


class LocalTrainer(object):
    def __init__(self,
                 params: ExperimentParams,
                 model: TRecover,
                 criterion: Callable[..., Tensor],
                 optimizer: Optimizer,
                 exp_dir: Path,
                 scheduler: Optional[BaseScheduler] = None,
                 monitor: Optional[BaseMonitor] = None,
                 device: Optional[torch.device] = None,
                 accumulation_step: int = 1,
                 log_interval: int = 1,
                 saving_interval: int = 1,
                 vis_interval: int = 1,
                 n_columns_to_show: Optional[int] = None,
                 delimiter: str = '',
                 console: Console = log.project_console):
        self.params = params
        self.model = model
        self.criterion = criterion  # should return average on the batch
        self.optimizer = optimizer
        self.exp_dir = exp_dir
        self.scheduler = scheduler or IdentityScheduler()
        self.monitor = monitor or IdentityMonitor()
        self.device = device or torch.device("cpu")
        self.accumulation_step = accumulation_step
        self.log_interval = log_interval
        self.saving_interval = saving_interval
        self.vis_interval = vis_interval
        self.n_columns_to_show = n_columns_to_show
        self.delimiter = delimiter

        self.experiment_folder = self.exp_dir / self.monitor.experiment_name
        self.weights_folder = self.experiment_folder / 'weights'

        self.experiment_folder.mkdir(parents=True, exist_ok=True)
        self.weights_folder.mkdir(parents=True, exist_ok=True)

        self.log_file = self.experiment_folder / f'log_{self.monitor.experiment_name}.html'
        self.console = console

        self.__log_init_params()

    def __enter__(self) -> Tuple['LocalTrainer', BaseMonitor]:
        self.monitor.start()

        return self, self.monitor

    def __exit__(self, exc_type: Union[None, Type[BaseException]],
                 exc_val: Union[None, BaseException],
                 exc_tb: traceback.TracebackException
                 ) -> None:
        self.save_html_log()

        with tempfile.TemporaryDirectory() as dp:
            save_params(self.params.jsonify(), Path(dp, 'params.json'))
            self.model.save(Path(dp, 'model.pt'))
            shutil.copy(self.log_file, dp)

            self.monitor.log_artifact(dp)

        self.monitor.finish()

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f'Trainer(model={self.model}, ' \
               f'criterion={self.criterion}, ' \
               f'optimizer={optimizer_to_str(self.optimizer)}), ' \
               f'exp_dir={self.exp_dir}, ' \
               f'scheduler={self.scheduler}, ' \
               f'device={self.device}, ' \
               f'accumulation_step={self.accumulation_step}, ' \
               f'log_interval={self.log_interval}, ' \
               f'saving_interval={self.saving_interval}, ' \
               f'vis_interval={self.vis_interval}, ' \
               f'n_columns_to_show={self.n_columns_to_show}, ' \
               f'delimiter={self.delimiter})'

    @property
    def __lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def __log_init_params(self) -> None:
        self.console.print(f'Date: {self.monitor.experiment_name}')
        self.console.print(f'Model: {self.model}')
        self.console.print(f'Trainable parameters: {self.model.params_count:,}')
        self.console.print(f'Optimizer: {optimizer_to_str(self.optimizer)}')
        self.console.print(f'Accumulation step: {self.accumulation_step}')
        self.console.print(f'Scheduler: {self.scheduler}')

    def __train_step(self, offset: int, train_loader: DataLoader, accumulation_step: int = 1) -> None:
        self.model.train()

        self.console.rule(Text('Training step', style='bold magenta'), style='magenta')

        train_loss = 0.0

        self.optimizer.zero_grad(set_to_none=True)
        start_time = time()

        for batch_idx, train_tensors in enumerate(train_loader, start=1):
            src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask = transfer(train_tensors,
                                                                                    to_device=self.device)

            tgt_out = self.model(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
            tgt_out = tgt_out.reshape(-1, self.model.token_size)
            tgt = tgt.view(-1)
            src = src.reshape(-1, self.model.token_size)

            loss = self.criterion(src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask, tgt_out)
            loss.backward()

            train_loss += loss.item()

            if batch_idx % accumulation_step == 0:
                self.scheduler.step()
                self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

                if (offset + batch_idx) % self.log_interval == 0:
                    accuracy = (torch.argmax(tgt_out, dim=1) == tgt).float().sum().item() / tgt.size(0)
                    train_loss /= accumulation_step

                    self.console.print(f'Train Batch:  {offset + batch_idx:^7} | '
                                       f'Loss: {train_loss:>10.6f} | Accuracy: {accuracy:>6.3f} | '
                                       f'Elapsed: {time() - start_time:>7.3f} | LR {round(self.__lr, 6):>8}')

                    self.monitor.log_metrics({'Train loss': train_loss,
                                              'Train accuracy': accuracy,
                                              'LR': self.__lr,
                                              'train_step': offset + batch_idx},
                                             step=offset + batch_idx)

                    train_loss = 0.0
                    start_time = time()

    @torch.no_grad()
    def __val_step(self, offset: int, val_loader: DataLoader) -> None:
        self.model.eval()

        self.console.rule(Text('Validation step', style='bold magenta'), style='magenta')

        val_loss = 0
        val_accuracy = 0
        start_time = time()

        for batch_idx, val_tensors in enumerate(val_loader, start=1):
            src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask = transfer(val_tensors, to_device=self.device)

            tgt_out = self.model(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
            tgt_out = tgt_out.reshape(-1, self.model.token_size)
            tgt = tgt.view(-1)
            src = src.reshape(-1, self.model.token_size)

            loss = self.criterion(src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask, tgt_out).item()
            accuracy = ((torch.argmax(tgt_out, dim=1) == tgt).float().sum() / tgt.size(0)).item()

            val_loss += loss
            val_accuracy += accuracy

            self.monitor.log_metrics({"Val loss": loss,
                                      "Val accuracy": accuracy,
                                      'val_step': offset + batch_idx},
                                     step=offset + batch_idx)

            self.console.print(f'Val Batch:    {offset + batch_idx:^7} | Loss: {loss:>10.6f} | '
                               f'Accuracy: {accuracy:>6.3f} | Elapsed: {time() - start_time:>7.3f}')

            start_time = time()

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)

        self.console.print(Panel(f'Val Loss:    {val_loss:>10.6f} \nVal Accuracy: {val_accuracy:>6.3f}',
                                 title='Validation average', highlight=True, border_style='magenta'))

    @torch.no_grad()
    def __vis_step(self, vis_loader: DataLoader) -> None:
        self.model.eval()

        self.console.rule(Text('Visualization step', style='bold magenta'), style='magenta')

        for batch_idx, vis_tensors in enumerate(vis_loader, start=1):
            src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask = transfer(vis_tensors, to_device=self.device)

            tgt_out = self.model(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
            tgt_out = tgt_out.reshape(-1, self.model.token_size)
            prediction = torch.argmax(tgt_out, dim=1).view_as(tgt)

            for i in range(src.size(0)):
                columns = tensor_to_columns(src[i, : self.n_columns_to_show])
                columns = visualize_columns(columns, delimiter=self.delimiter, as_rows=True)
                columns = (Text(row, style='bright_blue', overflow='ellipsis', no_wrap=True) for row in columns)
                target = tensor_to_target(prediction[i, : self.n_columns_to_show])
                predicted = visualize_target(target, delimiter=self.delimiter)
                original = tensor_to_target(tgt[i, : self.n_columns_to_show])
                original = visualize_target(original, delimiter=self.delimiter)

                panel_group = Group(
                    Text('Columns', style='magenta', justify='center'),
                    *columns,
                    Text('Predicted', style='magenta', justify='center'),
                    Text(predicted, style='cyan', justify='center', overflow='ellipsis'),
                    Text('Original', style='magenta', justify='center'),
                    Text(original, justify='center', overflow='ellipsis')
                )

                self.console.print(
                    Panel(panel_group, title=f'Example {(batch_idx - 1) * vis_loader.batch_size + i + 1}',
                          border_style='magenta'),
                    justify='center'
                )
                self.console.print('\n')

    def train(self,
              n_epochs: int,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              vis_loader: Optional[DataLoader] = None,
              epoch_seek: int = 0
              ) -> None:
        if len(train_loader) % self.accumulation_step != 0:
            self.console.print('WARNING: Train dataset size must be evenly divisible by batch_size * accumulation_step',
                               style='bold red')
        if self.device.type == 'cpu':
            self.console.print('WARNING: Training without GPU usage', style='bold red')

        try:
            for epoch_idx in range(epoch_seek + 1, epoch_seek + n_epochs + 1):
                train_offset = len(train_loader) * (epoch_idx - 1)
                val_offset = len(val_loader) * (epoch_idx - 1)

                self.__train_step(train_offset, train_loader, self.accumulation_step)

                if val_loader:
                    self.__val_step(val_offset, val_loader)

                if vis_loader and epoch_idx % self.vis_interval == 0:
                    self.__vis_step(vis_loader)

                if epoch_idx % self.saving_interval == 0:
                    self.save_html_log()
                    self.save_model(str(train_offset + len(train_loader)))

        except KeyboardInterrupt:
            self.console.print('Training is interrupted', style='yellow')

    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()

        test_loss = 0
        test_accuracy = 0

        with Progress(
                TextColumn('{task.description}', style='bright_blue'),
                BarColumn(complete_style='bright_blue'),
                TextColumn('{task.percentage:>3.0f}%', style='bright_blue'),
                TextColumn('Remaining', style='bright_blue'),
                TimeRemainingColumn(),
                TextColumn('Elapsed', style='bright_blue'),
                TimeElapsedColumn(),
                console=log.project_console,
                transient=True,
        ) as progress:
            test_progress = progress.add_task('Testing', total=len(test_loader))

            for batch_idx, test_tensors in enumerate(test_loader, start=1):
                src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask = transfer(test_tensors,
                                                                                        to_device=self.device)

                tgt_out = self.model(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
                tgt_out = tgt_out.reshape(-1, self.model.token_size)
                tgt = tgt.view(-1)
                src = src.reshape(-1, self.model.token_size)

                loss = self.criterion(src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask, tgt_out).item()
                accuracy = ((torch.argmax(tgt_out, dim=1) == tgt).float().sum() / tgt.size(0)).item()

                test_loss += loss
                test_accuracy += accuracy

                self.monitor.log_metrics({"Test loss": loss,
                                          "Test accuracy": accuracy,
                                          'test_step': batch_idx},
                                         step=batch_idx)

                progress.update(test_progress, advance=1)

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)

        self.monitor.log_variables({'Test Loss': test_loss, 'Test Accuracy': test_accuracy})

        self.console.print(Panel(f'Test Loss:    {test_loss:>10.6f} \nTest Accuracy: {test_accuracy:>6.3f}',
                                 title='Testing', highlight=True, border_style='magenta'))

        return test_loss, test_accuracy

    def save_model(self, weights_name: str) -> None:
        self.model.save(filename=self.weights_folder / f'{self.monitor.experiment_name}_{weights_name}.pt')

    def save_html_log(self) -> None:
        self.console.save_html(str(self.log_file), clear=False)

    # TODO find_batch_size
    def find_batch_size(self):
        pass
