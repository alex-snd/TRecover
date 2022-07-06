import time
from argparse import ArgumentParser
from typing import List, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from trecover.config import var, exp_var, log
from trecover.train.collab.arguments import (ModelArguments, TrainingPeerArguments, PLTrainerArguments, DataArguments,
                                             CollaborativeArguments)
from trecover.train.collab.callback import CollabCheckpoint
from trecover.train.collab.dht import DHTManager, LocalMetrics
from trecover.train.collab.strategy import CollaborativeStrategy
from trecover.train.collab.trainer import LightningWrapper, LightningTuneWrapper
from trecover.utils.train import parse_dataclasses

rank_zero_only.rank = 1


def get_collab_parser() -> ArgumentParser:
    # TODO docs

    parser = ArgumentParser('LocalTrainer')

    # ------------------------------------------------GENERAL PARAMETERS------------------------------------------------

    parser.add_argument('--project-name', default='TRecoverLocal', type=str,
                        help='Monitor project name')
    parser.add_argument('--mlflow', action='store_true',
                        help='Use Mlflow as monitor. Default is W&B')

    # --------------------------------------------------DATA PARAMETERS-------------------------------------------------

    parser.add_argument('--seed', default=2531, type=int,
                        help='Reproducible seed number')
    parser.add_argument('--train-files', default=exp_var.TRAIN_DATA, type=str,
                        help='Path to train files folder')
    parser.add_argument('--val-files', default=exp_var.VAL_DATA, type=str,
                        help='Path to validation files folder')
    parser.add_argument('--vis-files', default=exp_var.VIS_DATA, type=str,
                        help='Path to visualization files folder')
    parser.add_argument('--test-files', default=exp_var.VIS_DATA, type=str,
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

    # ----------------------------------------OPTIMIZATION PARAMETERS---------------------------------------------------

    parser.add_argument('--lr', default=0.001577, type=float,
                        help='Learning rate value.')
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


def monitor(args: Optional[List[str]] = None) -> None:
    data_args, model_args, peer_args, trainer_args, collab_args = parse_dataclasses(DataArguments,
                                                                                    ModelArguments,
                                                                                    TrainingPeerArguments,
                                                                                    PLTrainerArguments,
                                                                                    CollaborativeArguments,
                                                                                    args=args)
    if peer_args.initial_peers:
        peer_args.initial_peers = [peer_args.initial_peers, ]
    else:
        peer_args.initial_peers = []

    dht_manager = DHTManager(peer_args)

    current_step = -1

    while True:
        metrics_entry = dht_manager.dht.get(peer_args.experiment_prefix + "_metrics", latest=True)

        if metrics_entry is not None and len(metrics_entry.value) > 0:
            metrics_dict = metrics_entry.value
            metrics = [LocalMetrics.parse_obj(metrics_dict[peer].value) for peer in metrics_dict]
            latest_step = max(item.step for item in metrics)

            if latest_step != current_step:
                current_step = latest_step

                log.project_console.print(metrics_entry.value)

                sum_loss = 0
                sum_acc = 0
                num_samples = 0
                sum_perf = 0
                sum_mini_steps = 0

                for item in metrics:
                    sum_loss += item.loss
                    sum_acc += item.accuracy
                    sum_perf += item.samples_per_second
                    num_samples += item.samples_accumulated
                    sum_mini_steps += item.mini_steps

                alive_peers = len(metrics)
                current_loss = sum_loss / sum_mini_steps if sum_mini_steps else sum_loss
                current_accuracy = sum_acc / sum_mini_steps if sum_mini_steps else sum_acc

                log.project_console.print(f'Step: {current_step}')
                log.project_console.print(f'Peers alive: {alive_peers}')
                log.project_console.print(f'Loss: {current_loss}')
                log.project_console.print(f'Accuracy: {current_accuracy}')
                log.project_console.print(f'Samples: {num_samples}')
                log.project_console.print(f'Performance: {sum_perf}')

        log.project_console.print('Fetching', style='yellow')

        time.sleep(3)


def train(args: Optional[List[str]] = None) -> None:
    data_args, model_args, peer_args, trainer_args, collab_args = parse_dataclasses(DataArguments,
                                                                                    ModelArguments,
                                                                                    TrainingPeerArguments,
                                                                                    PLTrainerArguments,
                                                                                    CollaborativeArguments,
                                                                                    args=args)
    if peer_args.initial_peers:
        peer_args.initial_peers = [peer_args.initial_peers, ]
    else:
        peer_args.initial_peers = []

    if trainer_args.batch_size is None:
        trainer_args.batch_size = tune(args)

    dht_manager = DHTManager(peer_args)
    wrapped_model = LightningWrapper(data_args, model_args, trainer_args)
    collab_strategy = CollaborativeStrategy(peer_args, trainer_args, collab_args, dht_manager=dht_manager)

    collab_checkpoint = CollabCheckpoint(dht_manager, peer_args)

    trainer = pl.Trainer(default_root_dir=exp_var.LIGHTNING_REGISTRY_DIR,
                         max_epochs=4,
                         num_sanity_val_steps=0,
                         log_every_n_steps=1,
                         enable_progress_bar=False,
                         strategy=collab_strategy,
                         auto_select_gpus=True,
                         accelerator='auto',
                         enable_checkpointing=False,
                         callbacks=[collab_checkpoint])
    trainer.fit(wrapped_model)


def tune(args: Optional[List[str]] = None) -> int:
    data_args, model_args, peer_args, trainer_args, collab_args = parse_dataclasses(DataArguments,
                                                                                    ModelArguments,
                                                                                    TrainingPeerArguments,
                                                                                    PLTrainerArguments,
                                                                                    CollaborativeArguments,
                                                                                    args=args)
    if peer_args.initial_peers:
        peer_args.initial_peers = [peer_args.initial_peers, ]
    else:
        peer_args.initial_peers = []

    log.project_console.print('Trying to find appropriate batch size for this machine', style='magenta')

    if trainer_args.tune_strategy:
        tune_strategy = CollaborativeStrategy(peer_args, trainer_args, collab_args, tune=True)
    else:
        tune_strategy = None

    trainer = pl.Trainer(default_root_dir=exp_var.LIGHTNING_REGISTRY_DIR,
                         auto_scale_batch_size=True,
                         auto_select_gpus=True,
                         accelerator='auto',
                         strategy=tune_strategy)

    result = trainer.tune(model=LightningTuneWrapper(data_args, model_args, trainer_args),
                          scale_batch_size_kwargs={
                              'init_val': trainer_args.scale_batch_size_init_val,
                              'mode': 'binsearch',
                              'max_trials': trainer_args.tune_max_trials
                          })

    batch_size = result['scale_batch_size']
    log.project_console.print(f'Found batch size: {batch_size}', style='green')

    if trainer_args.scale_tuned_batch_size:
        batch_size = int(batch_size // var.BATCH_SIZE_SCALE_FACTOR)
        log.project_console.print(f'Batch size was scaled to: {trainer_args.batch_size}', style='green')

    return batch_size
