import shutil
import tempfile
import traceback
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from time import time
from typing import Callable, Optional, Tuple, Union, Type, List

import torch
import wandb
from rich.console import Group, Console
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.text import Text
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from config import var, log, train as train_config
from zreader.data import WikiDataset
from zreader.loss import CustomPenaltyLoss
from zreader.model import ZReader
from zreader.scheduler import BaseScheduler, WarmupScheduler, IdentityScheduler
from zreader.utils.model import get_model, get_recent_weights_path, save_params
from zreader.utils.train import ExperimentParams, set_seeds, optimizer_to_str
from zreader.utils.transform import tensor_to_columns, tensor_to_target
from zreader.utils.visualization import visualize_columns, visualize_target


class Trainer(object):
    def __init__(self,
                 model: ZReader,
                 criterion: Callable[..., Tensor],
                 optimizer: Optimizer,
                 exp_dir: Path,
                 scheduler: Optional[BaseScheduler] = None,
                 device: Optional[torch.device] = None,
                 accumulation_step: int = 1,
                 log_interval: int = 1,
                 saving_interval: int = 1,
                 vis_interval: int = 1,
                 n_columns_to_show: Optional[int] = None,
                 delimiter: str = '',
                 console: Console = log.project_console):
        self.model = model
        self.criterion = criterion  # should return average on the batch
        self.optimizer = optimizer
        self.exp_dir = exp_dir
        self.scheduler = scheduler or IdentityScheduler()
        self.device = device or torch.device("cpu")
        self.accumulation_step = accumulation_step
        self.log_interval = log_interval
        self.saving_interval = saving_interval
        self.vis_interval = vis_interval
        self.n_columns_to_show = n_columns_to_show
        self.delimiter = delimiter

        date = datetime.now()
        self.experiment_mark = f'{date.month:0>2}-{date.day:0>2}_{date.hour:0>2}-{date.minute:0>2}'

        self.experiment_folder = self.exp_dir / self.experiment_mark
        self.weights_folder = self.experiment_folder / 'weights'

        self.experiment_folder.mkdir(parents=True, exist_ok=True)
        self.weights_folder.mkdir(parents=True, exist_ok=True)

        self.log_file = self.experiment_folder / f'log_{self.experiment_mark}.html'
        self.console = console

        self.__log_init_params()

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Union[None, Type[BaseException]],
                 exc_val: Union[None, BaseException],
                 exc_tb: traceback.TracebackException
                 ) -> None:
        self.save_html_log()
        self.save_model('last_saving')

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
    def lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def __log_init_params(self) -> None:
        self.console.print(f'Date: {self.experiment_mark}')
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
            src = src.reshape(-1, self.model.token_size)

            loss = self.criterion(src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask, tgt_out)
            loss.backward()

            train_loss += loss.item()

            if batch_idx % accumulation_step == 0:
                self.scheduler.step()
                self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

                if (offset + batch_idx) % self.log_interval == 0:
                    accuracy = (torch.argmax(tgt_out, dim=1) == tgt).float().sum() / tgt.size(0)
                    train_loss /= accumulation_step

                    self.console.print(f'Train Batch:  {offset + batch_idx:^7} | '
                                       f'Loss: {train_loss:>10.6f} | Accuracy: {accuracy:>6.3f} | '
                                       f'Elapsed: {time() - start_time:>7.3f} | LR {round(self.lr, 6):>8}')

                    wandb.log({'Train loss': train_loss,
                               'Train accuracy': accuracy,
                               'LR': self.lr,
                               'train_step': offset + batch_idx})

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
            src = src.reshape(-1, self.model.token_size)

            loss = self.criterion(src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask, tgt_out).item()
            accuracy = ((torch.argmax(tgt_out, dim=1) == tgt).float().sum() / tgt.size(0)).item()

            val_loss += loss
            val_accuracy += accuracy

            wandb.log({"Val loss": loss, "Val accuracy": accuracy, 'val_step': offset + batch_idx})

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

                # TODO wandb.log examples

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              vis_loader: DataLoader,
              n_epochs: int,
              epoch_seek: int = 0
              ) -> None:
        wandb.define_metric('train_step', hidden=True)
        wandb.define_metric('val_step', hidden=True)
        wandb.define_metric('test_step', hidden=True)
        wandb.define_metric('Train loss', step_metric='train_step', summary='min', goal='minimize')
        wandb.define_metric('Val loss', step_metric='val_step', summary='min', goal='minimize')
        wandb.define_metric('Test loss', step_metric='test_step', summary='min', goal='minimize')
        wandb.define_metric('Train accuracy', step_metric='train_step', summary='max', goal='maximize')
        wandb.define_metric('Val accuracy', step_metric='val_step', summary='max', goal='maximize')
        wandb.define_metric('Test accuracy', step_metric='test_step', summary='max', goal='maximize')

        self.console.print(f'Batch size: {train_loader.batch_size}')
        self.console.print(f'Min threshold: {train_loader.dataset.min_threshold}')
        self.console.print(f'Max threshold: {train_loader.dataset.max_threshold}')

        if len(train_loader) % self.accumulation_step != 0:
            self.console.print('WARNING: Train dataset size must be evenly divisible by batch_size * accumulation_step',
                               style='bold red')
        if self.device == torch.device('cpu'):
            self.console.print('WARNING: Training without GPU usage', style='bold red')

        try:
            for epoch_idx in range(epoch_seek + 1, epoch_seek + n_epochs + 1):
                train_offset = len(train_loader) * (epoch_idx - 1)
                val_offset = len(val_loader) * (epoch_idx - 1)

                self.__train_step(train_offset, train_loader, self.accumulation_step)

                self.__val_step(val_offset, val_loader)

                if epoch_idx % self.vis_interval == 0:
                    self.__vis_step(vis_loader)

                if epoch_idx % self.saving_interval == 0:
                    self.save_html_log()
                    self.save_model(str(train_offset + len(train_loader)))

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
                console=log.project_console,
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
                src = src.reshape(-1, self.model.token_size)

                loss = self.criterion(src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask, tgt_out).item()
                accuracy = ((torch.argmax(tgt_out, dim=1) == tgt).float().sum() / tgt.size(0)).item()

                test_loss += loss
                test_accuracy += accuracy

                wandb.log({"Test loss": loss, "Test accuracy": accuracy, 'test_step': batch_idx})

                progress.update(test_progress, advance=1)

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)

        self.console.print(Panel(f'Test Loss:    {test_loss:>10.6f} \nTest Accuracy: {test_accuracy:>6.3f}',
                                 title='Testing', highlight=True, border_style='magenta'))

        return test_loss, test_accuracy

    def save_model(self, weights_name: str) -> None:
        self.model.save(filename=self.weights_folder / f'{self.experiment_mark}_{weights_name}.pt')

    def save_html_log(self) -> None:
        self.console.save_html(str(self.log_file), clear=False)


def train(params: ExperimentParams) -> None:
    if params.n_columns_to_show > params.pe_max_len:
        log.project_logger.error(f'[red]Parameter n_to_show={params.n_columns_to_show} '
                                 f'must be less than {params.pe_max_len}')
        return

    set_seeds(seed=params.seed)

    train_files = [Path(params.train_files, file) for file in Path(params.train_files).iterdir()]
    val_files = [Path(params.val_files, file) for file in Path(params.val_files).iterdir()]
    vis_files = [Path(params.vis_files, file) for file in Path(params.vis_files).iterdir()]
    test_files = [Path(params.test_files, file) for file in Path(params.test_files).iterdir()]

    train_dataset = WikiDataset(datafiles=train_files, min_threshold=params.min_threshold,
                                max_threshold=params.max_threshold, dataset_size=params.train_dataset_size)
    val_dataset = WikiDataset(datafiles=val_files, min_threshold=params.min_threshold,
                              max_threshold=params.max_threshold, dataset_size=params.val_dataset_size)
    vis_dataset = WikiDataset(datafiles=vis_files, min_threshold=params.min_threshold,
                              max_threshold=params.max_threshold, dataset_size=params.vis_dataset_size)
    test_dataset = WikiDataset(datafiles=test_files, min_threshold=params.min_threshold,
                               max_threshold=params.max_threshold, dataset_size=params.test_dataset_size)

    train_loader = train_dataset.create_dataloader(batch_size=params.batch_size, min_noise=params.min_noise,
                                                   max_noise=params.max_noise, num_workers=params.n_workers)
    val_loader = val_dataset.create_dataloader(batch_size=params.batch_size, min_noise=params.min_noise,
                                               max_noise=params.max_noise, num_workers=params.n_workers)
    vis_loader = vis_dataset.create_dataloader(batch_size=params.batch_size, min_noise=params.min_noise,
                                               max_noise=params.max_noise, num_workers=params.n_workers)
    test_loader = test_dataset.create_dataloader(batch_size=params.batch_size, min_noise=params.min_noise,
                                                 max_noise=params.max_noise, num_workers=params.n_workers)

    device = torch.device('cuda' if torch.cuda.is_available() and not params.no_cuda else 'cpu')
    weights_path = params.abs_weights_name or get_recent_weights_path(Path(params.exp_dir), params.exp_mark,
                                                                      params.weights_name)

    z_reader = get_model(params.token_size, params.pe_max_len, params.n_layers, params.d_model, params.n_heads,
                         params.d_ff, params.dropout, device,
                         weights=weights_path, silently=False)

    # criterion = CustomCrossEntropyLoss(ignore_index=-1)
    criterion = CustomPenaltyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(z_reader.parameters(), lr=params.lr, betas=(0.9, 0.98), eps=1e-9)

    scheduler = WarmupScheduler(optimizer, params.d_model, params.warmup, params.lr_step_size, seek=params.lr_step_seek)

    trainer = Trainer(model=z_reader,
                      criterion=criterion,
                      optimizer=optimizer,
                      exp_dir=Path(params.exp_dir),
                      scheduler=scheduler,
                      device=device,
                      accumulation_step=params.accumulation_step,
                      log_interval=params.log_interval,
                      saving_interval=params.saving_interval,
                      vis_interval=params.vis_interval,
                      n_columns_to_show=params.n_columns_to_show,
                      delimiter=params.delimiter)

    json_params = params.jsonify()

    with wandb.init(project='ZReaderLocal',
                    name=trainer.experiment_mark,
                    config=json_params,
                    dir=var.WANDB_REGISTRY_DIR.absolute()):
        with trainer:
            trainer.train(train_loader, val_loader, vis_loader, params.n_epochs, params.epoch_seek)

            test_loss, test_accuracy = trainer.test(test_loader=test_loader)

        wandb.run.summary['Test loss'] = test_loss
        wandb.run.summary['Test accuracy'] = test_accuracy

        with tempfile.TemporaryDirectory() as dp:
            save_params(json_params, Path(dp, 'params.json'))
            z_reader.save(Path(dp, 'z_reader.pt'))
            shutil.copy(trainer.log_file, dp)

            wandb.log_artifact(artifact_or_path=dp,
                               name=f'l{params.n_layers}_h{params.n_heads}_d{params.d_model}_ff{params.d_ff}',
                               type='result')


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()

    # --------------------------------------------------DATA PARAMETERS-------------------------------------------------

    parser.add_argument('--seed', default=2531, type=int,
                        help='Reproducible seed number')
    parser.add_argument('--train-files', default=train_config.TRAIN_DATA, type=str,
                        help='Path to train files folder')
    parser.add_argument('--val-files', default=train_config.VAL_DATA, type=str,
                        help='Path to validation files folder')
    parser.add_argument('--vis-files', default=train_config.VIS_DATA, type=str,
                        help='Path to visualization files folder')
    parser.add_argument('--test-files', default=train_config.VIS_DATA, type=str,
                        help='Path to test files folder')
    parser.add_argument('--min-threshold', default=256, type=int,
                        help='Min sentence lengths')
    parser.add_argument('--max-threshold', default=256, type=int,
                        help='Max sentence lengths')
    parser.add_argument('--train-dataset-size', default=5000, type=int,
                        help='Train dataset size')
    parser.add_argument('--val-dataset-size', default=500, type=int,
                        help='Validation dataset size')
    parser.add_argument('--vis-dataset-size', default=5, type=int,
                        help='Visualization dataset size')
    parser.add_argument('--test-dataset-size', default=500, type=int,
                        help='Test dataset size')
    parser.add_argument('--batch-size', default=5, type=int,
                        help='Batch size')
    parser.add_argument('--n-workers', default=3, type=int,
                        help='Number of processes for dataloaders')
    parser.add_argument('--min-noise', default=0, type=int,
                        help='Min noise range')
    parser.add_argument('--max-noise', default=1, type=int,
                        help='Max noise range')

    # ----------------------------------------------MODEL PARAMETERS----------------------------------------------------

    parser.add_argument('--token-size', default=len(var.ALPHABET), type=int,
                        help='Token size')
    parser.add_argument('--pe-max-len', default=256, type=int,
                        help='Positional encoding max length')
    parser.add_argument('--n-layers', default=12, type=int,
                        help='Number of encoder and decoder blocks')
    parser.add_argument('--d-model', default=96, type=int,
                        help='Model dimension - number of expected features in the encoder (decoder) input')
    parser.add_argument('--n-heads', default=12, type=int,
                        help='Number of encoder and decoder attention heads')
    parser.add_argument('--d-ff', default=192, type=int,
                        help='Dimension of the feedforward layer')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout range')
    parser.add_argument('--exp-dir', default=var.EXPERIMENTS_DIR, type=str,
                        help='Experiments folder')
    parser.add_argument('--abs-weights-name', type=str,
                        help='Absolute weights path')
    parser.add_argument('--exp-mark', type=str,
                        help="Experiments folder mark placed in 'exp-dir'")
    parser.add_argument('--weights-name', type=str,
                        help="Weights name in specified using 'exp-mark' experiments folder")
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable cuda usage')

    # --------------------------------------------OPTIMIZATION PARAMETERS-----------------------------------------------

    parser.add_argument('--lr', default=0.001577, type=float,
                        help='Learning rate value. Fictive with defined scheduler')
    parser.add_argument('--lr-step-seek', default=0, type=int,
                        help='Number of steps for WarmupScheduler to seek')
    parser.add_argument('--warmup', default=600, type=int,
                        help='Warmup value for WarmupScheduler')
    parser.add_argument('--lr-step-size', default=1, type=int,
                        help='Step size foe learning rate updating')
    parser.add_argument('--accumulation-step', default=1, type=int,
                        help='Number of steps for gradients accumulation')

    # ---------------------------------------------TRAIN LOOP PARAMETERS------------------------------------------------

    parser.add_argument('--n-epochs', default=1000, type=int,
                        help='Number of epochs for training')
    parser.add_argument('--epoch-seek', default=4, type=int,
                        help='Number of epochs to seek. necessary for correct weights naming'
                             ' in case of an interrupted model training process')
    parser.add_argument('--saving-interval', default=1, type=int,
                        help='Weights saving interval per epoch')
    parser.add_argument('--log-interval', default=1, type=int,
                        help='Metrics logging interval per batch-step')
    parser.add_argument('--vis-interval', default=1, type=int,
                        help='Visualization interval per epoch')
    parser.add_argument('--n-columns-to-show', default=96, type=int,
                        help='Number of visualization columns to show')
    parser.add_argument('--delimiter', default='', type=str,
                        help='Visualization columns delimiter')

    return parser


def get_experiment_params(args: Optional[List[str]] = None) -> ExperimentParams:
    return ExperimentParams(get_parser().parse_args(args=args))


if __name__ == '__main__':
    try:
        train(get_experiment_params())
    except Exception as e:
        log.project_logger.error(e)
        log.project_console.print_exception()
        log.error_console.print_exception(show_locals=True)
