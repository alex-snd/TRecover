# TODO docs
from argparse import ArgumentParser
from os import getpid
from typing import List, Optional

import pytorch_lightning as pl

from trecover.config import var, exp_var, log
from trecover.train.colab import utils
from trecover.train.colab.arguments import (ModelArguments, TrainingPeerArguments, PLTrainerArguments, DataArguments,
                                            CollaborativeArguments)
from trecover.train.colab.trainer import LightningWrapper, Task


def get_colab_parser() -> ArgumentParser:
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
    import hivemind
    from trecover.config.log import project_logger
    from trecover.utils.train import get_experiment_mark
    from hivemind import DHT, get_dht_time

    params = TrainingPeerArguments()

    validators, local_public_key = utils.make_validators('trecover')

    dht = DHT(
        start=True,
        # client_mode=params.client_mode,
        record_validators=validators,
        # use_ipfs=params.use_ipfs,
        # use_ipfs=True,
        # host_maddrs=params.host_maddrs,
        host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
        # identity_path=params.identity_path,
    )

    project_logger.info(f'To join the training, use initial_peers:\n'
                        f'{[str(addr) for addr in dht.get_visible_maddrs()]}')
    project_logger.info(f'Global IP: {hivemind.utils.networking.choose_ip_address(dht.get_visible_maddrs())}')

    dht.store('my_key', ('i', 'love', 'bees', get_experiment_mark()),
              expiration_time=get_dht_time() + 6000)

    while True:
        pass


def train(args: Optional[List[str]] = None) -> None:
    data_args, model_args, peer_args, trainer_args, collab_args = utils.parse_dataclasses(DataArguments,
                                                                                          ModelArguments,
                                                                                          TrainingPeerArguments,
                                                                                          PLTrainerArguments,
                                                                                          CollaborativeArguments,
                                                                                          args=args)

    peer_args.initial_peers = [peer_args.initial_peers, ]

    # parser = HfArgumentParser((DataArguments, ModelArguments, TrainingPeerArguments,
    #                            PLTrainerArguments, CollaborativeArguments))
    # data_args, model_args, peer_args, trainer_args, collab_args = parser.parse_args_into_dataclasses(args=args)

    log.project_logger.info(f'Found {len(peer_args.initial_peers)} '
                            f'initial peers: {peer_args.initial_peers}')

    trainer = pl.Trainer(default_root_dir=var.EXPERIMENTS_DIR,
                         max_epochs=1,
                         num_sanity_val_steps=0,
                         log_every_n_steps=1,
                         auto_scale_batch_size=True)

    task = Task(data_args, model_args, peer_args, trainer_args, collab_args)
    # opt = task.optimizer
    # print('Optimizer is created in main process')
    print(f'Main process pid: {getpid()}')
    model_wrapper = LightningWrapper(task, data_args, model_args, peer_args, trainer_args, collab_args)

    if not model_wrapper.batch_size:
        # Dict with scale_batch_size key
        trainer.tune(model=model_wrapper, scale_batch_size_kwargs={"init_val": 1, "mode": "power"})

    # while True:
    #     pass

    print(f'Fit model')
    trainer.fit(model_wrapper)

    # callback = Callback()

    # trainer = PLTrainer()
    # trainer(train)

    # set_seeds(seed=params.seed)
    # transformers.set_seed(trainer_args.seed)  # TODO move to trainer and print model param_count

    # device = torch.device('cuda' if torch.cuda.is_available() and not params.no_cuda else 'cpu')
    # batch_generation_device = device if params.allocate_on_device else None
    #
    # weights_path = get_recent_weights_path(Path(params.exp_dir), params.exp_mark,
    #                                        params.weights_name)
    #
    # model = get_model(params.token_size, params.pe_max_len, params.n_layers, params.d_model, params.n_heads,
    #                   params.d_ff, params.dropout, device=device, silently=False,
    #                   weights=None)  # TODO weights path weights_path
    #
    # train_files = [Path(params.train_files, file) for file in Path(params.train_files).iterdir()]
    # val_files = [Path(params.val_files, file) for file in Path(params.val_files).iterdir()]
    # vis_files = [Path(params.vis_files, file) for file in Path(params.vis_files).iterdir()]
    # test_files = [Path(params.test_files, file) for file in Path(params.test_files).iterdir()]
    #
    # train_dataset = WikiDataset(datafiles=train_files, min_threshold=params.min_threshold,
    #                             max_threshold=params.max_threshold, dataset_size=params.train_dataset_size)
    # val_dataset = WikiDataset(datafiles=val_files, min_threshold=params.min_threshold,
    #                           max_threshold=params.max_threshold, dataset_size=params.val_dataset_size)
    # vis_dataset = WikiDataset(datafiles=vis_files, min_threshold=params.min_threshold,
    #                           max_threshold=params.max_threshold, dataset_size=params.vis_dataset_size)
    # test_dataset = WikiDataset(datafiles=test_files, min_threshold=params.min_threshold,
    #                            max_threshold=params.max_threshold, dataset_size=params.test_dataset_size)
    #
    # collate = StandardCollate(min_noise=params.min_noise, max_noise=params.max_noise, device=batch_generation_device)
    #
    # train_loader = train_dataset.create_dataloader(batch_size=params.per_device_train_batch_size, collate=collate,
    #                                                num_workers=params.n_workers)
    # val_loader = val_dataset.create_dataloader(batch_size=params.per_device_train_batch_size, collate=collate,
    #                                            num_workers=params.n_workers)
    # vis_loader = vis_dataset.create_dataloader(batch_size=params.per_device_train_batch_size, collate=collate,
    #                                            num_workers=params.n_workers)
    # test_loader = test_dataset.create_dataloader(batch_size=params.per_device_train_batch_size, collate=collate,
    #                                              num_workers=params.n_workers)
    #
    # validators, local_public_key = utils.make_validators(params.run_id)
    #
    # dht = DHT(
    #     start=True,
    #     initial_peers=params.initial_peers,
    #     # client_mode=params.client_mode,
    #     record_validators=validators,
    #     # use_ipfs=params.use_ipfs,
    #     # use_ipfs=True,
    #     # host_maddrs=params.host_maddrs,
    #     host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
    #     announce_maddrs=params.announce_maddrs,
    #     # identity_path=params.identity_path,
    # )
    #
    # project_logger.info(f'To join the training, use initial_peers:\n'
    #                     f'{[str(addr) for addr in dht.get_visible_maddrs()]}')
    # project_logger.info(f'Global IP: {hivemind.utils.networking.choose_ip_address(dht.get_visible_maddrs())}')

    # criterion = CustomCrossEntropyLoss(ignore_index=-1)
    #
    # # We need to make such a lambda function instead of just an optimizer instance
    # # to make hivemind.Optimizer(..., offload_optimizer=True) work
    # unwrapped_optimizer = lambda parameters: torch.optim.Adam(
    #     parameters,
    #     lr=params.learning_rate,
    #     betas=(params.adam_beta1, params.adam_beta2),
    #     eps=params.adam_epsilon,
    #     weight_decay=params.weight_decay,
    # )
    #
    # no_decay = ["bias", "LayerNorm.weight"]
    # model_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": params.weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #     },
    # ]
    #
    # scheduler = lambda opt: utils.get_linear_schedule_with_warmup(
    #     opt, num_warmup_steps=params.warmup_steps, num_training_steps=params.total_steps
    # )
    #
    # optimizer = Optimizer(
    #     dht=dht,
    #     run_id=params.run_id,
    #     target_batch_size=params.target_batch_size,
    #     batch_size_per_step=params.per_device_train_batch_size,
    #     optimizer=unwrapped_optimizer,
    #     params=model_parameters,
    #     scheduler=scheduler,
    #     matchmaking_time=params.matchmaking_time,
    #     averaging_timeout=params.averaging_timeout,
    #     offload_optimizer=True,
    #     delay_optimizer_step=True,
    #     delay_grad_averaging=True,
    #     client_mode=params.client_mode,
    #     grad_compression=Float16Compression(),
    #     state_averaging_compression=Float16Compression(),
    #     # averager_opts={"bandwidth": params.bandwidth, **asdict(params)},
    #     # tracker_opts=asdict(tracker_args),
    #     verbose=True,
    # )
    #
    # experiment_mark = get_experiment_mark()
    #
    # with LocalTrainer(params=params,
    #                   model=model,
    #                   criterion=criterion,
    #                   optimizer=optimizer,
    #                   exp_dir=Path(params.exp_dir),
    #                   scheduler=scheduler,
    #                   monitor=monitor,
    #                   device=device,
    #                   accumulation_step=params.accumulation_step,
    #                   log_interval=params.log_interval,
    #                   saving_interval=params.saving_interval,
    #                   vis_interval=params.vis_interval,
    #                   n_columns_to_show=params.n_columns_to_show,
    #                   delimiter=params.delimiter
    #                   ) as (trainer, _):
    #     trainer.train(params.n_epochs, train_loader, val_loader, vis_loader, params.epoch_seek)
    #     trainer.test(test_loader=test_loader)
    #
    # dht.store('my_key', ('i', 'love', 'bees', experiment_mark),
    #           expiration_time=get_dht_time() + 6000)
    #
    # while True:
    #     pass


if __name__ == '__main__':
    train()
