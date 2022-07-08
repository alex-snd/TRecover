from dataclasses import asdict
from typing import Optional

import hivemind
from hivemind import SizeAdaptiveCompression, Float16Compression, Uniform8BitQuantization
from torch.optim import Optimizer

from trecover.train.collab.arguments import CollaborativeArguments
from trecover.train.scheduler import get_wrapped_linear_scheduler_with_warmup


def create_collab_opt(optimizer: Optimizer,
                      dht: hivemind.DHT,
                      experiment_prefix: str,
                      collab_args: CollaborativeArguments,
                      warmup_steps: int,
                      total_steps: int,
                      batch_size_per_step: Optional[int] = None,
                      client_mode: bool = False,
                      verbose: bool = True,
                      ) -> hivemind.Optimizer:
    params = optimizer.param_groups
    wrapped_scheduler = get_wrapped_linear_scheduler_with_warmup(warmup_steps, total_steps)

    averaging_compression = SizeAdaptiveCompression(
        threshold=2 ** 16 + 1, less=Float16Compression(), greater_equal=Uniform8BitQuantization())

    return hivemind.Optimizer(dht=dht,
                              run_id=experiment_prefix,
                              params=params,
                              optimizer=type(optimizer),
                              scheduler=wrapped_scheduler,
                              offload_optimizer=True,
                              delay_grad_averaging=False,
                              delay_optimizer_step=True,
                              batch_size_per_step=batch_size_per_step,
                              grad_compression=averaging_compression,
                              state_averaging_compression=averaging_compression,
                              client_mode=client_mode,
                              verbose=verbose,
                              **asdict(collab_args))
