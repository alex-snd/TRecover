import shutil
import tempfile
import traceback
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from time import time
from typing import Callable, Optional, Tuple, Union, Type

import mlflow
import torch
from rich.console import Group, Console
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.text import Text
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import config
from src.ml.data import WikiDataset, Collate
from src.ml.model import ZReader
from src.ml.scheduler import BaseScheduler, Scheduler, IdentityScheduler
from src.utils.model import get_model
from src.utils.train import set_seeds, optimizer_to_str
from src.utils.visualization import visualize_columns, visualize_target


class Trainer(object):
    def __init__(self,
                 model: ZReader,
                 criterion: Callable[[Tensor, Tensor], Tensor],
                 optimizer: Optimizer,
                 working_dir: Path,
                 scheduler: Optional[BaseScheduler] = None,
                 device: Optional[torch.device] = None,
                 log_interval: int = 1,
                 n_columns_to_show: Optional[int] = None,
                 delimiter: str = '',
                 console: Console = config.project_console
                 ) -> None:
        self.model = model
        self.criterion = criterion  # should return average on the batch
        self.optimizer = optimizer
        self.working_dir = working_dir
        self.scheduler = scheduler or IdentityScheduler()
        self.device = device or torch.device("cpu")
        self.log_interval = log_interval
        self.n_columns_to_show = n_columns_to_show
        self.delimiter = delimiter

        date = datetime.now()
        self.experiment_mark = f'{date.month:0>2}{date.day:0>2}_{date.hour:0>2}{date.minute:0>2}'

        self.experiment_folder = Path(self.working_dir, self.experiment_mark)
        self.weights_folder = Path(self.experiment_folder, 'weights')

        self.experiment_folder.mkdir(parents=True, exist_ok=True)
        self.weights_folder.mkdir(parents=True, exist_ok=True)

        self.log_file = Path(self.experiment_folder, f'{self.experiment_mark}.html')
        self.console = console

        self.__log_init_params()

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Union[None, Type[BaseException]],
                 exc_val: Union[None, BaseException],
                 exc_tb: traceback.TracebackException) -> None:
        self.save_html()
        self.save_model('last_saving')

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f'Trainer(model={self.model}, ' \
               f'criterion={self.criterion}, ' \
               f'optimizer={optimizer_to_str(self.optimizer)}), ' \
               f'working_dir={self.working_dir}, ' \
               f'scheduler={self.scheduler}, ' \
               f'device={self.device}, ' \
               f'log_interval={self.log_interval}, ' \
               f'n_columns_to_show={self.n_columns_to_show}, ' \
               f'delimiter={self.delimiter})'

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def __log_init_params(self) -> None:
        self.console.print(f'Date: {self.experiment_mark}')
        self.console.print(f'Model: {self.model}')
        self.console.print(f'Trainable parameters: {self.model.params_count:,}')
        self.console.print(f'Optimizer: {optimizer_to_str(self.optimizer)}')
        self.console.print(f'Scheduler: {self.scheduler}')

    def __train_step(self, offset: int, train_loader: DataLoader, accumulation_step: int = 1) -> None:
        self.model.train()

        self.console.rule(Text('Training step', style='bold magenta'), style='magenta')

        train_loss = 0.0

        self.optimizer.zero_grad()
        start_time = time()

        for batch_idx, (src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask) in enumerate(train_loader,
                                                                                                   start=1):
            src = src.to(self.device)
            tgt_inp = tgt_inp.to(self.device)
            tgt = tgt.to(self.device)
            src_pad_mask = src_pad_mask.to(self.device)
            tgt_pad_mask = tgt_pad_mask.to(self.device)
            tgt_attn_mask = tgt_attn_mask.to(self.device)

            tgt_out = self.model(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
            tgt_out = tgt_out.reshape(-1, self.model.token_size)
            tgt = tgt.view(-1)

            loss = self.criterion(tgt_out, tgt)
            loss.backward()

            train_loss += loss.item()

            if batch_idx % accumulation_step == 0:
                self.scheduler.step()
                self.optimizer.step()

                self.optimizer.zero_grad()

                if (offset + batch_idx) % self.log_interval == 0:
                    accuracy = (torch.argmax(tgt_out, dim=1) == tgt).float().sum() / tgt.size(0)
                    train_loss /= accumulation_step

                    self.console.print(f'Train Batch:  {offset + batch_idx:^7} | '
                                       f'Loss: {train_loss:>10.6f} | Accuracy: {accuracy:>6.3f} | '
                                       f'Elapsed: {time() - start_time:>7.3f} | LR {round(self.lr, 6):>8}')

                    mlflow.log_metrics({"Train loss": train_loss}, step=offset + batch_idx)

                    train_loss = 0.0
                    start_time = time()

    @torch.no_grad()
    def __val_step(self, offset: int, val_loader: DataLoader) -> None:
        self.model.eval()

        self.console.rule(Text('Validation step', style='bold magenta'), style='magenta')

        val_loss = 0
        val_accuracy = 0
        start_time = time()

        for batch_idx, (src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask) in enumerate(val_loader, start=1):
            src = src.to(self.device)
            tgt_inp = tgt_inp.to(self.device)
            tgt = tgt.to(self.device)
            src_pad_mask = src_pad_mask.to(self.device)
            tgt_pad_mask = tgt_pad_mask.to(self.device)
            tgt_attn_mask = tgt_attn_mask.to(self.device)

            tgt_out = self.model(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
            tgt_out = tgt_out.reshape(-1, self.model.token_size)
            tgt = tgt.view(-1)

            loss = self.criterion(tgt_out, tgt).item()
            accuracy = ((torch.argmax(tgt_out, dim=1) == tgt).float().sum() / tgt.size(0)).item()

            val_loss += loss
            val_accuracy += accuracy

            mlflow.log_metrics({"Val loss": loss}, step=offset + batch_idx)
            mlflow.log_metrics({"Val accuracy": accuracy}, step=offset + batch_idx)

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

        for batch_idx, (src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask) in enumerate(vis_loader, start=1):
            src = src.to(self.device)
            tgt_inp = tgt_inp.to(self.device)
            tgt = tgt.to(self.device)
            src_pad_mask = src_pad_mask.to(self.device)
            tgt_pad_mask = tgt_pad_mask.to(self.device)
            tgt_attn_mask = tgt_attn_mask.to(self.device)

            tgt_out = self.model(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
            tgt_out = tgt_out.reshape(-1, self.model.token_size)
            prediction = torch.argmax(tgt_out, dim=1).view_as(tgt)

            for i in range(src.size(0)):
                columns = visualize_columns(src[i, : self.n_columns_to_show], delimiter=self.delimiter, as_rows=True)
                columns = (Text(row, style='bright_blue', overflow='ellipsis', no_wrap=True) for row in columns)
                predicted = visualize_target(prediction[i, : self.n_columns_to_show], delimiter=self.delimiter)
                original = visualize_target(tgt[i, : self.n_columns_to_show], delimiter=self.delimiter)

                panel_group = Group(
                    Text('Columns', style='magenta', justify='center'),
                    *columns,
                    Text('Predicted', style='magenta', justify='center'),
                    Text(predicted, style='cyan', justify='center', overflow='ellipsis'),
                    Text('Original', style='magenta', justify='center'),
                    Text(original, justify='center', overflow='ellipsis')
                )

                self.console.print(
                    Panel(panel_group, title=f'Example {i + 1}', border_style='magenta'),
                    justify='center'
                )
                self.console.print('\n')

    def train(self,
              n_epochs: int,
              train_loader: DataLoader,
              val_loader: DataLoader,
              vis_loader: DataLoader,
              epoch_seek: int = 0,
              accumulation_step: int = 1,
              vis_interval: int = 1,
              saving_interval: int = 1
              ) -> None:
        self.console.print(f'Batch size: {train_loader.batch_size}')
        self.console.print(f'Min threshold: {train_loader.dataset.min_threshold}')
        self.console.print(f'Max threshold: {train_loader.dataset.max_threshold}')
        self.console.print(f'Accumulation step: {accumulation_step}')

        if len(train_loader) % accumulation_step != 0:
            self.console.print('Train dataset size must be evenly divisible by batch_size * accumulation_step',
                               style='bold red')

        try:
            for epoch_idx in range(epoch_seek + 1, epoch_seek + n_epochs + 1):
                offset = len(train_loader) * (epoch_idx - 1)

                self.__train_step(offset, train_loader, accumulation_step)

                self.__val_step(offset, val_loader)

                if epoch_idx % vis_interval == 0:
                    self.__vis_step(vis_loader)

                if epoch_idx % saving_interval == 0:
                    self.save_model(str(offset + len(train_loader)))

        except KeyboardInterrupt:
            self.console.print('Interrupted')

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
                transient=True,
        ) as progress:
            test_progress = progress.add_task('Testing', total=len(test_loader))

            for batch_idx, (src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask) in enumerate(test_loader,
                                                                                                       start=1):
                src = src.to(self.device)
                tgt_inp = tgt_inp.to(self.device)
                tgt = tgt.to(self.device)
                src_pad_mask = src_pad_mask.to(self.device)
                tgt_pad_mask = tgt_pad_mask.to(self.device)
                tgt_attn_mask = tgt_attn_mask.to(self.device)

                tgt_out = self.model(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
                tgt_out = tgt_out.reshape(-1, self.model.token_size)
                tgt = tgt.view(-1)

                loss = self.criterion(tgt_out, tgt).item()
                accuracy = ((torch.argmax(tgt_out, dim=1) == tgt).float().sum() / tgt.size(0)).item()

                test_loss += loss
                test_accuracy += accuracy

                mlflow.log_metrics({"Test loss": loss}, step=batch_idx)
                mlflow.log_metrics({"Test accuracy": accuracy}, step=batch_idx)

                progress.update(test_progress, advance=1)

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)

        self.console.print(Panel(f'Test Loss:    {test_loss:>10.6f} \nTest Accuracy: {test_accuracy:>6.3f}',
                                 title='Testing', highlight=True, border_style='magenta'))

        return test_loss, test_accuracy

    def save_model(self, weights_name: str) -> None:
        self.model.save(filename=Path(self.weights_folder, f'{self.experiment_mark}_{weights_name}'))

    def save_html(self) -> None:
        self.console.save_html(self.log_file)


def train(params: Namespace) -> None:
    if params.n_columns_to_show > params.pe_max_len:
        config.project_logger.error(f'[red]Parameter n_to_show={params.n_columns_to_show} '
                                    f'must be less than {params.pe_max_len}')
        return

    set_seeds(seed=params.seed)

    train_dataset = WikiDataset(datafiles=params.train_files, min_threshold=params.min_threshold,
                                max_threshold=params.max_threshold, dataset_size=params.train_dataset_size)
    val_dataset = WikiDataset(datafiles=params.val_files, min_threshold=params.min_threshold,
                              max_threshold=params.max_threshold, dataset_size=params.val_dataset_size)
    vis_dataset = WikiDataset(datafiles=params.vis_files, min_threshold=params.min_threshold,
                              max_threshold=params.max_threshold, dataset_size=params.vis_dataset_size)
    test_dataset = WikiDataset(datafiles=params.test_files, min_threshold=params.min_threshold,
                               max_threshold=params.max_threshold, dataset_size=params.test_dataset_size)

    train_loader = train_dataset.create_dataloader(batch_size=params.batch_size, min_noise=params.min_noise,
                                                   max_noise=params.max_noise, num_workers=params.num_workers)
    val_loader = val_dataset.create_dataloader(batch_size=params.batch_size, min_noise=params.min_noise,
                                               max_noise=params.max_noise, num_workers=params.num_workers)
    vis_loader = vis_dataset.create_dataloader(batch_size=params.batch_size, min_noise=params.min_noise,
                                               max_noise=params.max_noise, num_workers=params.num_workers)
    test_loader = test_dataset.create_dataloader(batch_size=params.batch_size, min_noise=params.min_noise,
                                                 max_noise=params.max_noise, num_workers=params.num_workers)

    z_reader = get_model(params.token_size, params.pe_max_len, params.num_layers, params.d_model, params.n_heads,
                         params.d_ff, params.dropout, params.device,
                         weights=Path(params.weights_folder, params.weights_name))

    optimizer = torch.optim.Adam(z_reader.parameters(), lr=params.lr, betas=(0.9, 0.98), eps=1e-9)

    scheduler = Scheduler(optimizer, params.d_model, params.warmup, params.lr_step_size, seek=params.step_seek)

    trainer = Trainer(model=z_reader, criterion=params.criterion, optimizer=optimizer,
                      working_dir=config.EXPERIMENTS_DIR, scheduler=scheduler, device=params.device,
                      log_interval=params.log_interval, n_columns_to_show=params.n_columns_to_show,
                      delimiter=params.delimiter)

    mlflow.set_experiment(experiment_name='ZReader')

    with mlflow.start_run(run_name=f'l{params.num_layers}_h{params.n_heads}_d{params.d_model}_ff{params.d_ff}'):
        with trainer:
            trainer.train(params.n_epochs, train_loader, val_loader, vis_loader, params.epoch_seek,
                          params.accumulation_step, params.vis_interval, params.saving_interval)

            test_loss, test_accuracy = trainer.test(test_loader=test_loader)

        mlflow.log_metrics({'Test loss': test_loss})
        mlflow.log_metrics({'Test accuracy': test_accuracy})

        with tempfile.TemporaryDirectory() as dp:
            # save_artifacts(vars(params), Path(dp, "params.json"))  # TODO convert for mlflow
            z_reader.save(Path(dp, "z_reader.pt"))
            shutil.copy(trainer.log_file, dp)

            mlflow.log_artifacts(dp)

        # mlflow.log_params(vars(params))


def main() -> None:
    params = Namespace(  # TODO implement this using argparse
        # ---------------------------------------------DATA PARAMETERS--------------------------------------------------
        seed=2531,
        train_files=[Path(config.TRAIN_DATA, file) for file in config.TRAIN_DATA.iterdir()],
        val_files=[Path(config.VAL_DATA, file) for file in config.VAL_DATA.iterdir()],
        vis_files=[Path(config.VIS_DATA, file) for file in config.VIS_DATA.iterdir()],
        test_files=[Path(config.VIS_DATA, file) for file in config.VIS_DATA.iterdir()],
        min_threshold=256,
        max_threshold=256,
        train_dataset_size=50,
        val_dataset_size=50,
        vis_dataset_size=5,
        test_dataset_size=500,
        num_workers=3,
        min_noise=0,
        max_noise=5,
        # --------------------------------------------MODEL PARAMETERS--------------------------------------------------
        token_size=len(Collate.alphabet_to_num),
        pe_max_len=1000,
        num_layers=6,
        d_model=512,  # d_model % n_heads = 0
        n_heads=16,
        d_ff=2048,
        dropout=0.1,
        weights_folder='',
        weights_name='',
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # -----------------------------------------OPTIMIZATION PARAMETERS----------------------------------------------
        criterion=CrossEntropyLoss(ignore_index=-1),
        lr=0,  # fictive with Scheduler
        step_seek=0,
        warmup=4_000,
        lr_step_size=1,
        # ------------------------------------------TRAIN LOOP PARAMETERS-----------------------------------------------
        n_epochs=1,
        epoch_seek=0,
        batch_size=5,
        accumulation_step=1,  # train_dataset_size % (batch_size * accumulation_step) == 0
        saving_interval=1,
        log_interval=1,
        vis_interval=1,
        n_columns_to_show=92,
        delimiter='',
        # --------------------------------------------------------------------------------------------------------------
    )

    train(params)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        config.project_logger.error(e)
        config.project_console.print_exception(show_locals=True)
        config.error_console.print_exception(show_locals=True)