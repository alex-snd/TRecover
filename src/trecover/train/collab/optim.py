import threading
import time
from argparse import Namespace
from typing import Optional, Dict, Any, Iterable, Callable

import hivemind
import torch
from hivemind import SizeAdaptiveCompression, Float16Compression, Uniform8BitQuantization
from torch.optim.lr_scheduler import LambdaLR

from trecover.config import log
from trecover.train.collab.wrapper import BaseModelWrapper
from trecover.train.scheduler import get_wrapped_linear_scheduler_with_warmup


class AuxiliaryOptimizer(object):
    def __init__(self, dht: hivemind.DHT, args: Namespace):
        self.lock = threading.Lock()
        self.finished = threading.Event()
        self.state_path = args.monitor_state_path
        self.assist_refresh = args.assist_refresh

        self.wrapped_model = BaseModelWrapper(args)
        self.wrapped_scheduler = get_wrapped_linear_scheduler_with_warmup(args.warmup, args.total_steps)
        self.collab_opt = create_collab_opt(wrapped_optimizer=self.wrapped_model.configure_optimizers(),
                                            params=self.wrapped_model.get_trainable_params(),
                                            dht=dht,
                                            args=args,
                                            wrapped_scheduler=self.wrapped_scheduler,
                                            assist_in_averaging=args.assist_in_averaging,
                                            verbose=args.verbose,
                                            batch_size_per_step=None)

    def start_assistant(self) -> None:
        def assist_averaging_in_background(lock: threading.Lock,
                                           collab_opt: hivemind.Optimizer,
                                           assist_refresh: float,
                                           finished: threading.Event):
            while not finished.is_set():
                try:
                    with lock:
                        collab_opt.step()

                    log.project_console.print('Assist in averaging...', style='bright_blue', justify='right')
                    time.sleep(assist_refresh)

                except Exception as e:
                    log.project_logger.exception(e, exc_info=True)

        averaging_thread = threading.Thread(name='AveragingAuxThread', target=assist_averaging_in_background,
                                            args=[self.lock, self.collab_opt, self.assist_refresh, self.finished],
                                            daemon=True)
        averaging_thread.start()

    def is_finished(self) -> None:
        self.finished.set()

    def state_dict(self) -> Dict[str, Any]:
        return {
            'model': self.wrapped_model.model.state_dict(),
            'optimizer': self.collab_opt.state_dict(),
            'scheduler': self.collab_opt.state_averager.scheduler.state_dict(),
            'local_epoch': self.collab_opt.local_epoch
        }

    def backup_state(self) -> None:
        with self.lock:
            self.collab_opt.load_state_from_peers()
            torch.save(self.state_dict(), self.state_path)


def create_collab_opt(wrapped_optimizer: Callable[[Iterable[Dict[str, Any]]], torch.optim.Optimizer],
                      params: [Iterable[Dict[str, Any]]],
                      dht: hivemind.DHT,
                      args: Namespace,
                      wrapped_scheduler: Optional[Callable[[torch.optim.Optimizer, ], LambdaLR]] = None,
                      assist_in_averaging: bool = False,
                      verbose: bool = True,
                      batch_size_per_step: Optional[int] = None
                      ) -> hivemind.Optimizer:
    averaging_compression = SizeAdaptiveCompression(
        threshold=2 ** 16 + 1, less=Float16Compression(), greater_equal=Uniform8BitQuantization())

    return hivemind.Optimizer(dht=dht,
                              run_id=args.experiment_prefix,
                              params=params,
                              optimizer=wrapped_optimizer,
                              scheduler=wrapped_scheduler,
                              offload_optimizer=True,
                              delay_grad_averaging=False,
                              delay_optimizer_step=True,
                              target_batch_size=args.target_batch_size,
                              batch_size_per_step=batch_size_per_step,
                              grad_compression=averaging_compression,
                              state_averaging_compression=averaging_compression,
                              client_mode=args.client_mode,
                              verbose=verbose,
                              auxiliary=assist_in_averaging,
                              matchmaking_time=args.matchmaking_time,
                              allreduce_timeout=args.allreduce_timeout,
                              averaging_timeout=args.averaging_timeout,
                              reuse_grad_buffers=not args.no_reuse_grad_buffers)
