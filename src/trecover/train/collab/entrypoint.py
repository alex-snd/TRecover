import os
from typing import List, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from trecover.config import log
from trecover.train.collab import arguments
from trecover.train.collab.callback import CollabCheckpoint
from trecover.train.collab.dht import DHTManager
from trecover.train.collab.monitor import CollaborativeMonitor
from trecover.train.collab.optim import AuxiliaryOptimizer
from trecover.train.collab.status import CommonStatus
from trecover.train.collab.strategy import CollaborativeStrategy
from trecover.train.collab.visualization import CollaborativeVisualizer
from trecover.train.collab.wrapper import BaseModelWrapper, PeerModelWrapper

rank_zero_only.rank = 1


def monitor(cli_args: Optional[List[str]] = None) -> None:
    args = arguments.sync_base_args(arguments.get_monitor_parser().parse_args(cli_args))

    if args.assist_in_averaging and args.client_mode:
        log.project_console.print('Client-mode peers cannot assist in averaging', style='red')
        return

    common_status = CommonStatus()
    dht_manager = DHTManager(args)
    aux_opt = None
    visualizer = None

    if args.upload_state or args.assist_in_averaging or args.visualize_every_step:
        aux_opt = AuxiliaryOptimizer(dht_manager=dht_manager,
                                     wrapped_model=BaseModelWrapper(args),
                                     args=args,
                                     common_status=common_status)

        if args.assist_in_averaging:
            aux_opt.start_assistant()

        if args.visualize_every_step:
            visualizer = CollaborativeVisualizer(aux_opt=aux_opt,
                                                 delimiter=args.delimiter,
                                                 visualize_every_step=args.visualize_every_step,
                                                 refresh_period=args.visualizer_refresh_period,
                                                 delay_in_steps=args.delay_in_steps,
                                                 delay_in_seconds=args.delay_in_seconds,
                                                 wandb_key=args.wandb_key,
                                                 wandb_project=args.wandb_project,
                                                 wandb_id=args.wandb_id,
                                                 wandb_registry=args.wandb_registry,
                                                 common_status=common_status)
            visualizer.start()

    try:
        metrics_monitor = CollaborativeMonitor(dht=dht_manager.dht,
                                               experiment_prefix=args.experiment_prefix,
                                               delay_in_steps=args.delay_in_steps,
                                               delay_in_seconds=args.delay_in_seconds,
                                               refresh_period=args.refresh_period,
                                               upload_state=args.upload_state,
                                               wandb_key=args.wandb_key,
                                               wandb_project=args.wandb_project,
                                               wandb_id=args.wandb_id,
                                               wandb_registry=args.wandb_registry,
                                               aux_opt=aux_opt,
                                               common_status=common_status)
        metrics_monitor.start()

    finally:
        if aux_opt and args.assist_in_averaging:
            aux_opt.finish(join=True)
        if visualizer and args.visualize_every_step:
            visualizer.finish(join=True)

        common_status.disable()


def train(cli_args: Optional[List[str]] = None) -> None:
    args = arguments.sync_base_args(arguments.get_train_parser().parse_args(cli_args))

    os.system('ulimit -n 16384')

    if args.batch_size is None:
        args.batch_size = tune(cli_args)

    dht_manager = DHTManager(args)
    wrapped_model = PeerModelWrapper(args)
    collab_strategy = CollaborativeStrategy(args=args, dht_manager=dht_manager)

    collab_checkpoint = CollabCheckpoint(dht_manager=dht_manager,
                                         statistics_expiration=args.statistics_expiration,
                                         backup_every_step=args.backup_every_step,
                                         sync_period=args.sync_period if args.sync_args else None)

    trainer = pl.Trainer(default_root_dir=args.pl_registry,
                         max_epochs=args.n_epochs,
                         accelerator=args.accelerator,
                         accumulate_grad_batches=args.accumulate_batches,
                         enable_progress_bar=False,
                         enable_checkpointing=False,
                         num_sanity_val_steps=0,
                         strategy=collab_strategy,
                         callbacks=[collab_checkpoint])

    trainer.fit(wrapped_model)


def tune(cli_args: Optional[List[str]] = None) -> int:
    args = arguments.sync_base_args(arguments.get_train_parser().parse_args(cli_args))

    log.project_console.print('Trying to find appropriate batch size for this machine', style='magenta')

    tune_strategy = CollaborativeStrategy(args=args, tune=True) if args.tune_strategy else None

    trainer = pl.Trainer(default_root_dir=args.pl_registry,
                         accelerator=args.accelerator,
                         accumulate_grad_batches=args.accumulate_batches,
                         auto_scale_batch_size=True,
                         strategy=tune_strategy)

    result = trainer.tune(model=PeerModelWrapper(args),
                          scale_batch_size_kwargs={
                              'init_val': args.tune_batch_size_init,
                              'mode': args.tune_mode,
                              'max_trials': args.tune_trials
                          })

    batch_size = result['scale_batch_size']
    log.project_console.print(f'Found batch size: {batch_size}', style='green')

    if args.scale_tuned_batch_size:
        batch_size = int(batch_size // args.batch_size_scale_factor)
        log.project_console.print(f'Batch size was scaled to: {batch_size}', style='green')

    return batch_size


def auxiliary(cli_args: Optional[List[str]] = None) -> None:
    args = arguments.sync_base_args(arguments.get_auxiliary_parser().parse_args(cli_args))

    if args.client_mode:
        log.project_console.print('Client-mode peers cannot assist in averaging', style='red')
        return

    os.system('ulimit -n 16384')
    aux_optimizer = AuxiliaryOptimizer(dht_manager=DHTManager(args), wrapped_model=BaseModelWrapper(args), args=args)

    aux_optimizer.start_assistant(attach=True)


def visualize(cli_args: Optional[List[str]] = None) -> None:
    args = arguments.sync_base_args(arguments.get_visualization_parser().parse_args(cli_args))

    if args.assist_in_averaging and args.client_mode:
        log.project_console.print('Client-mode peers cannot assist in averaging', style='red')
        return

    aux_opt = AuxiliaryOptimizer(dht_manager=DHTManager(args),
                                 wrapped_model=BaseModelWrapper(args),
                                 args=args)

    if args.assist_in_averaging:
        aux_opt.start_assistant()

    try:
        visualizer = CollaborativeVisualizer(aux_opt=aux_opt,
                                             delimiter=args.delimiter,
                                             visualize_every_step=args.visualize_every_step,
                                             refresh_period=args.visualizer_refresh_period,
                                             delay_in_steps=args.delay_in_steps,
                                             delay_in_seconds=args.delay_in_seconds,
                                             wandb_key=args.wandb_key,
                                             wandb_project=args.wandb_project,
                                             wandb_id=args.wandb_id,
                                             wandb_registry=args.wandb_registry)
        visualizer.start(attach=True)

    finally:
        if aux_opt and args.assist_in_averaging:
            aux_opt.finish(join=True)
