import gc
from argparse import ArgumentParser
from typing import List, Optional

import pytorch_lightning as pl
from hivemind import get_dht_time
from pytorch_lightning.utilities import rank_zero_only

from trecover.config import var, exp_var, log
from trecover.train.collab.arguments import (ModelArguments, TrainingPeerArguments, PLTrainerArguments, DataArguments,
                                             CollaborativeArguments)
from trecover.train.collab.dht import DHTManager
from trecover.train.collab.strategy import CollaborativeStrategy
from trecover.train.collab.trainer import LightningWrapper, LightningTuneWrapper
from trecover.utils.train import get_experiment_mark, parse_dataclasses

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
    dht_manager = DHTManager(peer_args)

    dht_manager.dht.store('my_key', ('i', 'love', 'bees', get_experiment_mark()),
                          expiration_time=get_dht_time() + 6000)

    while True:
        pass


def train(args: Optional[List[str]] = None) -> None:
    data_args, model_args, peer_args, trainer_args, collab_args = parse_dataclasses(DataArguments,
                                                                                    ModelArguments,
                                                                                    TrainingPeerArguments,
                                                                                    PLTrainerArguments,
                                                                                    CollaborativeArguments,
                                                                                    args=args)

    peer_args.initial_peers = [peer_args.initial_peers, ]

    if not trainer_args.batch_size:
        log.project_console.print('Trying to find appropriate batch size for this machine', style='magenta')

        trainer_args.batch_size = pl.Trainer(auto_scale_batch_size=True).tune(
            model=LightningTuneWrapper(data_args, model_args, trainer_args),
            scale_batch_size_kwargs={'init_val': 1, 'mode': 'binsearch'}
        )['scale_batch_size']

        log.project_console.print(f'Found batch size: {trainer_args.batch_size}', style='green')

        gc.collect()

    dht_manager = DHTManager(peer_args)
    wrapped_model = LightningWrapper(data_args, model_args, trainer_args, dht_manager)
    collab_strategy = CollaborativeStrategy(peer_args, trainer_args, collab_args, dht_manager)

    # callback = Callback()  # TODO validation and visualization as callback?

    trainer = pl.Trainer(default_root_dir=var.EXPERIMENTS_DIR,
                         max_epochs=1,
                         num_sanity_val_steps=0,
                         log_every_n_steps=1,
                         enable_progress_bar=False,
                         strategy=collab_strategy,
                         auto_select_gpus=True,
                         accelerator='auto')

    trainer.fit(wrapped_model)


if __name__ == '__main__':
    train()
