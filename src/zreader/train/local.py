from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

import torch

from config import var, log, train as train_config
from zreader.train.data import WikiDataset
from zreader.train.loss import CustomPenaltyLoss
from zreader.train.monitor import WandbMonitor, MlflowMonitor
from zreader.train.scheduler import WarmupScheduler
from zreader.train.trainer import LocalTrainer
from zreader.utils.model import get_model, get_recent_weights_path
from zreader.utils.train import ExperimentParams, set_seeds, get_experiment_mark


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()

    # ------------------------------------------------GENERAL PARAMETERS------------------------------------------------

    parser.add_argument('--project-name', default='ZReaderLocal', type=str,
                        help='Monitor project name')
    parser.add_argument('--mlflow', action='store_true',
                        help='Use Mlflow as monitor. Default is W&B')

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
    parser.add_argument('--train-dataset-size', default=2000, type=int,
                        help='Train dataset size')
    parser.add_argument('--val-dataset-size', default=400, type=int,
                        help='Validation dataset size')
    parser.add_argument('--vis-dataset-size', default=5, type=int,
                        help='Visualization dataset size')
    parser.add_argument('--test-dataset-size', default=200, type=int,
                        help='Test dataset size')
    parser.add_argument('--batch-size', default=2, type=int,
                        help='Batch size')
    parser.add_argument('--n-workers', default=3, type=int,
                        help='Number of processes for dataloaders')
    parser.add_argument('--min-noise', default=0, type=int,
                        help='Min noise range')
    parser.add_argument('--max-noise', default=0, type=int,
                        help='Max noise range')
    parser.add_argument('--allocate-on-device', action='store_true',
                        help='Allocate train data on specified device during batch generation')

    # ------------------------------------------MODEL PARAMETERS--------------------------------------------------------

    parser.add_argument('--token-size', default=len(var.ALPHABET), type=int,
                        help='Token size')
    parser.add_argument('--pe-max-len', default=256, type=int,
                        help='Positional encoding max length')
    parser.add_argument('--n-layers', default=12, type=int,
                        help='Number of encoder and decoder blocks')
    parser.add_argument('--d-model', default=768, type=int,
                        help='Model dimension - number of expected features in the encoder (decoder) input')
    parser.add_argument('--n-heads', default=12, type=int,
                        help='Number of encoder and decoder attention heads')
    parser.add_argument('--d-ff', default=768, type=int,
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

    # ----------------------------------------OPTIMIZATION PARAMETERS---------------------------------------------------

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
    parser.add_argument('--penalty-coefficient', default=1.0, type=float,
                        help='Penalty coefficient for CustomPenaltyLoss')

    # -----------------------------------------TRAIN LOOP PARAMETERS----------------------------------------------------

    parser.add_argument('--n-epochs', default=1000, type=int,
                        help='Number of epochs for training')
    parser.add_argument('--epoch-seek', default=0, type=int,
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


def train(args: Optional[List[str]] = None) -> None:
    params = get_experiment_params(args)

    if params.n_columns_to_show > params.pe_max_len:
        log.project_logger.error(f'[red]Parameter n_to_show={params.n_columns_to_show} '
                                 f'must be less than {params.pe_max_len}')
        return

    set_seeds(seed=params.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and not params.no_cuda else 'cpu')
    batch_generation_device = device if params.allocate_on_device else None
    weights_path = params.abs_weights_name or get_recent_weights_path(Path(params.exp_dir), params.exp_mark,
                                                                      params.weights_name)

    z_reader = get_model(params.token_size, params.pe_max_len, params.n_layers, params.d_model, params.n_heads,
                         params.d_ff, params.dropout, device,
                         weights=weights_path, silently=False)

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
                                                   max_noise=params.max_noise, num_workers=params.n_workers,
                                                   device=batch_generation_device)
    val_loader = val_dataset.create_dataloader(batch_size=params.batch_size, min_noise=params.min_noise,
                                               max_noise=params.max_noise, num_workers=params.n_workers,
                                               device=batch_generation_device)
    vis_loader = vis_dataset.create_dataloader(batch_size=params.batch_size, min_noise=params.min_noise,
                                               max_noise=params.max_noise, num_workers=params.n_workers,
                                               device=batch_generation_device)
    test_loader = test_dataset.create_dataloader(batch_size=params.batch_size, min_noise=params.min_noise,
                                                 max_noise=params.max_noise, num_workers=params.n_workers,
                                                 device=batch_generation_device)

    # criterion = CustomCrossEntropyLoss(ignore_index=-1)
    criterion = CustomPenaltyLoss(coefficient=params.penalty_coefficient, ignore_index=-1)
    optimizer = torch.optim.Adam(z_reader.parameters(), lr=params.lr, betas=(0.9, 0.98), eps=1e-9)

    scheduler = WarmupScheduler(optimizer, params.d_model, params.warmup, params.lr_step_size, seek=params.lr_step_seek)

    experiment_mark = get_experiment_mark()

    if params.mlflow:
        monitor = MlflowMonitor(params.project_name, experiment_mark, params.jsonify(),
                                var.MLFLOW_REGISTRY_DIR.absolute().as_uri())
    else:
        monitor = WandbMonitor(params.project_name, experiment_mark, params.jsonify(),
                               var.WANDB_REGISTRY_DIR.absolute())

    with LocalTrainer(params=params,
                      model=z_reader,
                      criterion=criterion,
                      optimizer=optimizer,
                      exp_dir=Path(params.exp_dir),
                      scheduler=scheduler,
                      monitor=monitor,
                      device=device,
                      accumulation_step=params.accumulation_step,
                      log_interval=params.log_interval,
                      saving_interval=params.saving_interval,
                      vis_interval=params.vis_interval,
                      n_columns_to_show=params.n_columns_to_show,
                      delimiter=params.delimiter
                      ) as (trainer, _):
        trainer.train(params.n_epochs, train_loader, val_loader, vis_loader, params.epoch_seek)
        trainer.test(test_loader=test_loader)
