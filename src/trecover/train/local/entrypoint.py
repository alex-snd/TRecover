from pathlib import Path
from typing import List, Optional

import torch

from trecover.config import exp_var, log
from trecover.train.data import WikiDataset, StandardCollate
from trecover.train.local import get_local_parser
from trecover.train.local.monitor import WandbMonitor, MlflowMonitor
from trecover.train.local.trainer import LocalTrainer
from trecover.train.loss import CustomPenaltyLoss
from trecover.train.scheduler import WarmupScheduler
from trecover.utils.model import get_model, get_recent_weights_path
from trecover.utils.train import set_seeds, get_experiment_params, get_experiment_mark


def train(cli_args: Optional[List[str]] = None) -> None:
    params = get_experiment_params(get_local_parser(), cli_args)

    if params.n_columns_to_show > params.pe_max_len:
        log.project_logger.error(f'[red]Parameter n_to_show={params.n_columns_to_show} '
                                 f'must be less than {params.pe_max_len}')
        return

    set_seeds(seed=params.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and not params.no_cuda else 'cpu')
    batch_generation_device = device if params.allocate_on_device else None
    weights_path = params.abs_weights_name or get_recent_weights_path(Path(params.exp_dir), params.exp_mark,
                                                                      params.weights_name)

    model = get_model(params.token_size, params.pe_max_len, params.n_layers, params.d_model, params.n_heads,
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

    collate = StandardCollate(min_noise=params.min_noise, max_noise=params.max_noise, device=batch_generation_device)

    train_loader = train_dataset.create_dataloader(batch_size=params.batch_size, collate=collate,
                                                   num_workers=params.n_workers)
    val_loader = val_dataset.create_dataloader(batch_size=params.batch_size, collate=collate,
                                               num_workers=params.n_workers)
    vis_loader = vis_dataset.create_dataloader(batch_size=params.batch_size, collate=collate,
                                               num_workers=params.n_workers)
    test_loader = test_dataset.create_dataloader(batch_size=params.batch_size, collate=collate,
                                                 num_workers=params.n_workers)

    # criterion = CustomCrossEntropyLoss(ignore_index=-1)
    criterion = CustomPenaltyLoss(coefficient=params.penalty_coefficient, ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.98), eps=1e-9)

    scheduler = WarmupScheduler(optimizer, params.d_model, params.warmup, params.lr_step_size, seek=params.lr_step_seek)

    experiment_mark = get_experiment_mark()

    if params.mlflow:
        monitor = MlflowMonitor(params.project_name, experiment_mark, params.jsonify(),
                                exp_var.MLFLOW_REGISTRY_DIR.absolute().as_uri())
    else:
        monitor = WandbMonitor(params.project_name, experiment_mark, params.jsonify(),
                               exp_var.WANDB_REGISTRY_DIR.absolute())

    with LocalTrainer(params=params,
                      model=model,
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
