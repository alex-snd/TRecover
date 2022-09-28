import math
import threading
import time
from argparse import Namespace
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Iterable, Callable, Union, Tuple

import hivemind
import torch
from bitsandbytes.functional import dequantize_blockwise, quantize_blockwise
from bitsandbytes.optim.optimizer import Optimizer2State
from hivemind import SizeAdaptiveCompression, Float16Compression, Uniform8BitQuantization
from speedtest import Speedtest, SpeedtestException
from torch.optim.lr_scheduler import LambdaLR

from trecover.config import log
from trecover.train.collab.dht import DHTManager, OptimizerStatus
from trecover.train.collab.status import CommonStatus, Status
from trecover.train.collab.wrapper import BaseModelWrapper
from trecover.train.scheduler import get_linear_scheduler_with_warmup

_ATOMIC_RLOCK = threading.RLock()


def atomic(method: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(method)
    def atomic_method(*args, **kwargs) -> Any:
        with _ATOMIC_RLOCK:
            return method(*args, **kwargs)

    return atomic_method


class CPULamb8Bit(Optimizer2State):
    """
    The optimizer implementation was taken from this repository https://github.com/NCAI-Research/CALM

    Implements Lamb with quantized 8-bit statistics. The statistics are stored in host memory in the quantized form.
    The LAMB optimizer and block-wise quantization are described in the following papers:
    - LAMB: "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes" https://arxiv.org/abs/1904.00962
    - Quantization: "8-bit Optimizers via Block-wise Quantization" https://arxiv.org/abs/2110.02861
    This specific implementation of LAMB is based on https://github.com/cybertronai/pytorch-lamb
    - bias correction defaults to False because paper v3 does not use debiasing
    - it has baked in clipping by global max_grad_norm

    Parameters
    ----------
    params : Union[Iterable[Dict[str, Any]], Dict[str, Any]]
        Iterable of parameters to optimize or dicts defining parameter groups
    lr : float, default=1e-3
        Learning rate
    betas : Tuple[float, float], default=(0.9, 0.999)
        Coefficients used for computing running averages of gradient and its square
    eps : float, default=1e-8
        Term added to the denominator to improve numerical stability
    weight_decay : float, default=0
        Weight decay (L2 penalty)
    clamp_value : float, default=10
        Clamp weight_norm in (0,clamp_value). Set to a high value to avoid it (e.g. 10e9)
    bias_correction : bool, default=True
        Debias statistics by (1 - beta**step)
    min_8bit_size : int
        Statistics for parameters with fewer than this many elements will not be quantized
    reuse_grad_buffers : bool, default=False
        if True, optimizer will modify gradients in-place to save memory. If enabled, one must ensure
        that .zero_grad() is called after each optimizer step.
    update_chunk_size : int, default=2 ** 24
        Quantized statistics will be de-quantized in chunks of up to this many elements.
    max_grad_norm : Optional[float], default=None
        Max norm of the gradients for clipping

    """

    def __init__(self,
                 params: Union[Iterable[Dict[str, Any]], Dict[str, Any]],
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-6,
                 weight_decay: float = 0,
                 clamp_value: float = 10,
                 bias_correction: bool = True,
                 min_8bit_size: int = 65536,
                 reuse_grad_buffers: bool = False,
                 update_chunk_size: int = 2 ** 24,
                 max_grad_norm: Optional[float] = None,
                 ) -> None:
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if clamp_value < 0.0:
            raise ValueError("Invalid clamp value: {}".format(clamp_value))

        self.clamp_value = clamp_value
        self.bias_correction = bias_correction
        self.reuse_grad_buffers = reuse_grad_buffers
        self.update_chunk_size = update_chunk_size
        self.max_grad_norm = max_grad_norm

        super(CPULamb8Bit, self).__init__("cpu-lamb",
                                          params,
                                          lr,
                                          betas,
                                          eps,
                                          weight_decay,
                                          optim_bits=8,
                                          min_8bit_size=min_8bit_size,
                                          args=None,
                                          percentile_clipping=100,
                                          block_wise=4096,
                                          max_unorm=0)

    @torch.no_grad()
    def step(self, closure=None):
        if self.max_grad_norm is not None:
            iter_params = (param for group in self.param_groups for param in group["params"])
            torch.nn.utils.clip_grad_norm_(iter_params, self.max_grad_norm)
        return super().step(closure=closure)

    @torch.no_grad()
    def init_state(self, group, p, gindex, pindex):
        config = self.get_config(gindex, pindex, group)
        assert config["percentile_clipping"] == 100, "percentile clipping is not implemented on CPU"
        assert config["max_unorm"] == 0

        if config["optim_bits"] == 32:
            dtype = torch.float32
        elif config["optim_bits"] == 8:
            dtype = torch.uint8
        else:
            raise NotImplementedError(f'Amount of optimizer bits not supported: {config["optim_bits"]}')

        if p.numel() < config["min_8bit_size"]:
            dtype = torch.float32

        state = self.state[p]
        state["step"] = 0

        if dtype == torch.float32 or (dtype == torch.uint8 and p.numel() < 4096):
            state["state1"] = torch.zeros_like(
                p,
                memory_format=torch.preserve_format,
                dtype=torch.float32,
                device=p.device,
            )
            state["state2"] = torch.zeros_like(
                p,
                memory_format=torch.preserve_format,
                dtype=torch.float32,
                device=p.device,
            )
        elif dtype == torch.uint8:
            if state["step"] == 0:
                if "dynamic" not in self.name2qmap:
                    self.fill_qmap()
                self.name2qmap["dynamic"] = self.name2qmap["dynamic"].to(p.device)
                self.name2qmap["udynamic"] = self.name2qmap["udynamic"].to(p.device)

            n = p.numel()
            blocks = (n - 1) // config["block_wise"] + 1

            state["state1"] = torch.zeros_like(
                p,
                memory_format=torch.preserve_format,
                dtype=torch.uint8,
                device=p.device,
            )
            state["qmap1"] = self.name2qmap["dynamic"]

            state["state2"] = torch.zeros_like(
                p,
                memory_format=torch.preserve_format,
                dtype=torch.uint8,
                device=p.device,
            )
            state["qmap2"] = self.name2qmap["udynamic"]

            state["absmax1"] = torch.zeros((blocks,), dtype=torch.float32, device=p.device)
            state["absmax2"] = torch.zeros((blocks,), dtype=torch.float32, device=p.device)

    @torch.no_grad()
    def update_step(self, group: Dict[str, Any], p: torch.Tensor, gindex: int, pindex: int):
        state = self.state[p]
        config = self.get_config(gindex, pindex, group)

        p_cpu, grad_cpu = p.cpu(), p.grad.cpu()
        # this is a no-op if parameters are already on CPU

        step = state["step"] = state["step"] + 1
        beta1, beta2 = group["betas"]

        param_delta = self._update_moments_and_compute_delta(
            state,
            config,
            p_cpu,
            grad_cpu,
            beta1,
            beta2,
            group["eps"],
            group["weight_decay"],
        )
        del grad_cpu  # grad_cpu is no longer needed and may be modified if self.reuse_grad_buffers

        step_norm = torch.norm(param_delta)
        weight_norm = p_cpu.norm().clamp(0, self.clamp_value)

        trust_ratio = weight_norm / step_norm if weight_norm != 0 and step_norm != 0 else 1.0
        state["weight_norm"], state["step_norm"], state["trust_ratio"] = (
            weight_norm,
            step_norm,
            trust_ratio,
        )

        # Apply bias to lr to avoid broadcast.
        bias_correction = math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step) if self.bias_correction else 1
        step_size = group["lr"] * bias_correction
        p.data.add_(param_delta.to(p.device), alpha=-step_size * trust_ratio)

    def _update_moments_and_compute_delta(self,
                                          state: Dict,
                                          config: Dict,
                                          p_cpu: torch.Tensor,
                                          grad_cpu: torch.Tensor,
                                          beta1: float,
                                          beta2: float,
                                          eps: float,
                                          weight_decay: float,
                                          ) -> torch.Tensor:
        step, block_size, chunk_size = (
            state["step"],
            config["block_wise"],
            self.update_chunk_size,
        )

        if state["state1"].dtype != torch.uint8:
            # not quantized: update normally
            exp_avg, exp_avg_sq = state["state1"], state["state2"]
            exp_avg.mul_(beta1).add_(grad_cpu, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad_cpu, grad_cpu, value=1 - beta2)

            sqrt_out = grad_cpu if self.reuse_grad_buffers else None
            _denominator = torch.sqrt(exp_avg_sq, out=sqrt_out).add_(eps)
            param_delta = torch.div(exp_avg, _denominator, out=_denominator)
            if weight_decay != 0:
                param_delta.add_(p_cpu, alpha=weight_decay)
            return param_delta
        elif p_cpu.numel() <= chunk_size:
            # quantized tensor within chunk size
            exp_avg = dequantize_blockwise(
                state["state1"],
                (state["absmax1"], state["qmap1"]),
                blocksize=block_size,
            )
            exp_avg_sq = dequantize_blockwise(
                state["state2"],
                (state["absmax2"], state["qmap2"]),
                blocksize=block_size,
            )

            exp_avg.mul_(beta1).add_(grad_cpu, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad_cpu, grad_cpu, value=1 - beta2)

            quantize_blockwise(exp_avg, state["qmap1"], state["absmax1"], out=state["state1"])
            quantize_blockwise(exp_avg_sq, state["qmap2"], state["absmax2"], out=state["state2"])
            # note: quantize_blockwise also modifies qmap and absmax in-place

            param_delta = exp_avg.div_(exp_avg_sq.sqrt_().add_(eps))
            # note: this changes statistics in-place, but it's okay b/c we saved quantized version

            if weight_decay != 0:
                param_delta.add_(p_cpu, alpha=weight_decay)
            return param_delta

        else:
            # very large quantized tensor, compute updates in chunks to save RAM
            flat_p, flat_grad, flat_state1, flat_state2 = (
                tensor.view(-1) for tensor in (p_cpu, grad_cpu, state["state1"], state["state2"])
            )
            output_buffer = flat_grad if self.reuse_grad_buffers else torch.empty_like(flat_grad)

            for chunk_index, chunk_start in enumerate(range(0, len(flat_p), chunk_size)):
                chunk = slice(chunk_start, chunk_start + chunk_size)
                chunk_blocks = slice(chunk_start // block_size, (chunk_start + chunk_size) // block_size)

                chunk_p, chunk_grad = flat_p[chunk], flat_grad[chunk]
                chunk_state1, chunk_state2 = flat_state1[chunk], flat_state2[chunk]
                chunk_absmax1, chunk_absmax2 = (
                    state["absmax1"][chunk_blocks],
                    state["absmax2"][chunk_blocks],
                )
                if chunk_state1.storage_offset() != 0:
                    chunk_state1, chunk_state2, chunk_absmax1, chunk_absmax2 = map(
                        torch.clone,
                        (chunk_state1, chunk_state2, chunk_absmax1, chunk_absmax2),
                    )  # clone chunks to ensure that tensors do not have offsets

                exp_avg_chunk = dequantize_blockwise(
                    chunk_state1, (chunk_absmax1, state["qmap1"]), blocksize=block_size
                )
                exp_avg_sq_chunk = dequantize_blockwise(
                    chunk_state2, (chunk_absmax2, state["qmap2"]), blocksize=block_size
                )

                exp_avg_chunk.mul_(beta1).add_(chunk_grad, alpha=1 - beta1)
                exp_avg_sq_chunk.mul_(beta2).addcmul_(chunk_grad, chunk_grad, value=1 - beta2)

                # note: output_buffer cannot be modified until this line because it shares memory with grad_cpu
                del chunk_grad

                flat_state1[chunk], (
                    state["absmax1"][chunk_blocks],
                    state["qmap1"],
                ) = quantize_blockwise(exp_avg_chunk, state["qmap1"], chunk_absmax1, out=chunk_state1)
                flat_state2[chunk], (
                    state["absmax2"][chunk_blocks],
                    state["qmap2"],
                ) = quantize_blockwise(exp_avg_sq_chunk, state["qmap2"], chunk_absmax2, out=chunk_state2)
                # note: we need to explicitly assign new quantized tensors because of cloning earlier

                torch.div(
                    exp_avg_chunk,
                    exp_avg_sq_chunk.sqrt_().add_(eps),
                    out=output_buffer[chunk],
                )
                # note: this changes statistics in-place, but it's okay b/c we saved quantized version

                if weight_decay != 0:
                    output_buffer[chunk].add_(flat_p[chunk], alpha=weight_decay)

            param_delta = output_buffer.view_as(grad_cpu)

            return param_delta


class CollaborativeOptimizer(object):
    transaction = _ATOMIC_RLOCK

    def __init__(self,
                 dht_manager: DHTManager,
                 wrapped_model: BaseModelWrapper,
                 args: Namespace,
                 batch_size_per_step: Optional[int],
                 verbose: bool = True):
        self.dht_manager = dht_manager
        self.wrapped_model = wrapped_model
        self.args = args
        self.status_key = f'{args.experiment_prefix}_optimizer_status'
        self.state_path: Path = args.state_path
        self.batch_size_per_step = batch_size_per_step
        self.verbose = verbose
        self.auxiliary = False
        self._opt = None
        self._original_allow_state_sharing = None

    def __enter__(self) -> 'CollaborativeOptimizer':
        self.transaction.acquire()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.transaction.release()

    @property
    @atomic
    def trainable_params(self) -> Iterable[Dict[str, Any]]:
        no_decay = ['bias', 'LayerNorm.weight']

        return [
            {
                'params': [p for n, p in self.wrapped_model.model.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.args.weight_decay,
            },
            {
                'params': [p for n, p in self.wrapped_model.model.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
            },
        ]

    @property
    def wrapped_optimizer(self) -> Callable[[Iterable[Dict[str, Any]]], CPULamb8Bit]:
        def optimizer(params: Iterable[Dict[str, Any]]) -> CPULamb8Bit:
            return CPULamb8Bit(params=params,
                               lr=self.args.lr,
                               betas=(self.args.adam_beta1, self.args.adam_beta2),
                               max_grad_norm=self.args.max_grad_norm,
                               clamp_value=self.args.clamp_value,
                               eps=self.args.adam_epsilon,
                               weight_decay=self.args.weight_decay,
                               reuse_grad_buffers=not self.args.no_reuse_grad_buffers,
                               bias_correction=True)

        return optimizer

    @property
    def wrapped_scheduler(self) -> Callable[[torch.optim.Optimizer, ], LambdaLR]:
        def scheduler(optimizer: torch.optim.Optimizer) -> LambdaLR:
            return get_linear_scheduler_with_warmup(optimizer=optimizer,
                                                    warmup_steps=self.args.warmup_steps,
                                                    total_steps=self.args.total_steps,
                                                    min_lr=self.args.min_lr)

        return scheduler

    @property
    @atomic
    def opt(self) -> hivemind.Optimizer:
        if not self._opt:
            assert (self.batch_size_per_step is None and self.auxiliary) or \
                   (self.batch_size_per_step is not None and not self.auxiliary)

            log.project_console.print('Configure collaborative optimizer', style='magenta', justify='right')

            averaging_compression = SizeAdaptiveCompression(
                threshold=2 ** 16 + 1, less=Float16Compression(), greater_equal=Uniform8BitQuantization())

            self._opt = hivemind.Optimizer(dht=self.dht_manager.dht,
                                           run_id=self.args.experiment_prefix,
                                           params=self.trainable_params,
                                           optimizer=self.wrapped_optimizer,
                                           scheduler=self.wrapped_scheduler,
                                           offload_optimizer=True,
                                           delay_grad_averaging=True,
                                           delay_optimizer_step=True,
                                           average_state_every=self.args.average_state_every,
                                           target_batch_size=self.args.target_batch_size,
                                           batch_size_per_step=self.batch_size_per_step,
                                           grad_compression=averaging_compression,
                                           state_averaging_compression=averaging_compression,
                                           client_mode=self.args.client_mode,
                                           verbose=self.verbose,
                                           auxiliary=self.auxiliary,
                                           matchmaking_time=self.args.matchmaking_time,
                                           allreduce_timeout=self.args.allreduce_timeout,
                                           averaging_timeout=self.args.averaging_timeout,
                                           reuse_grad_buffers=not self.args.no_reuse_grad_buffers,
                                           averager_opts={
                                               'min_vector_size': self.args.min_vector_size,
                                               'bandwidth': self.bandwidth
                                           })

            self._original_allow_state_sharing = self._opt.state_averager.allow_state_sharing

            log.project_console.print('Optimizer configuration is done', style='magenta', justify='right')

            if not self._opt.state_averager.allow_state_sharing:
                log.project_console.print(
                    'Note: Other peers will not be able to download collab state from this one',
                    style='yellow', justify='right'
                )

        return self._opt

    @property
    @atomic
    def run_id(self) -> str:
        return self.opt.run_id

    @property
    @atomic
    def allow_state_sharing(self) -> bool:
        return self.opt.state_averager.allow_state_sharing

    @allow_state_sharing.setter
    @atomic
    def allow_state_sharing(self, value: bool) -> None:
        self.opt.state_averager.allow_state_sharing = value

    @property
    @atomic
    def original_allow_state_sharing(self) -> bool:
        if self._original_allow_state_sharing is None:
            return self.allow_state_sharing

        return self._original_allow_state_sharing

    @property
    @atomic
    def num_peers(self) -> int:
        return self.opt.tracker.global_progress.num_peers

    @property
    @atomic
    def num_client_peers(self) -> int:
        return self.opt.tracker.global_progress.num_clients

    @property
    @atomic
    def num_non_client_peers(self) -> int:
        return self.num_peers - self.num_client_peers

    @property
    @atomic
    def global_epoch(self) -> int:
        return self.opt.tracker.global_epoch

    @property
    @atomic
    def local_epoch(self) -> int:
        return self.opt.local_epoch

    @property
    @atomic
    def local_samples_accumulated(self) -> int:
        return self.opt.grad_averager.local_samples_accumulated

    @property
    @atomic
    def samples_per_second(self) -> float:
        return self.opt.tracker.performance_ema.samples_per_second

    @property
    @atomic
    def lr(self) -> float:
        return self.opt.opt.param_groups[0]['lr']

    @property
    @atomic
    def bandwidth(self) -> Optional[float]:
        if not self.args.bandwidth:
            try:
                log.project_console.print('Measure internet bandwidth...', style='salmon1', justify='right')
                test = Speedtest()
                self.args.bandwidth = max(1, min(test.upload(), test.download()) / 1e6)
                log.project_console.print(f'Internet bandwidth (Mb/s): {self.args.bandwidth}',
                                          style='salmon1', justify='right')
            except SpeedtestException:
                log.project_console.print('Unable to measure internet bandwidth', style='red', justify='right')
                return None

        return self.args.bandwidth

    @property
    @atomic
    def client_mode(self) -> bool:
        return self.opt.client_mode

    @property
    @atomic
    def min_noise(self) -> int:
        return self.wrapped_model.collate.min_noise

    @property
    @atomic
    def max_noise(self) -> int:
        return self.wrapped_model.collate.max_noise

    @atomic
    def outrun(self, gap: Optional[int] = None) -> bool:
        if gap is None:
            gap = self.args.outrun_gap

        if status_dict := self._fetch_status_dict():
            for status in status_dict.values():
                status = OptimizerStatus.parse_obj(status.value)
                if self.opt.local_epoch > status.step + gap:
                    return True

        return False

    @atomic
    def wait_lagging_peers(self) -> None:
        log.project_console.print('Waiting for lagging peers...', style='magenta', justify='right')
        while self.outrun(gap=0):
            time.sleep(self.args.wait_period)
        log.project_console.print('Stop waiting for lagging peers', style='magenta', justify='right')

    @atomic
    def report_status(self) -> None:
        self.dht_manager.dht.store(
            key=self.status_key,
            subkey=self.dht_manager.local_public_key,
            value=OptimizerStatus(step=self.local_epoch, client=self.client_mode).dict(),
            expiration_time=hivemind.get_dht_time() + self.args.status_expiration,
            return_future=True
        )

    @torch.no_grad()
    @atomic
    def recover_state(self, sync: bool = False) -> None:
        log.project_console.print('Trying to recover collab state...', style='yellow', justify='right')

        if not self.state_path.exists():
            raise RuntimeError('Encountered broken parameters, but there is no backup to fall back to.')

        t_start = time.monotonic()
        self.allow_state_sharing = False

        if sync:
            while not self.params_are_finite:
                self.restore_from_backup()
                self.sync_state(force=True)
        else:
            self.restore_from_backup()

        self.allow_state_sharing = self.original_allow_state_sharing

        log.project_console.print(
            f'{self.local_epoch:_}-epoch collab state is recovered in {time.monotonic() - t_start:.4} sec',
            style='yellow', justify='right'
        )

    @property
    @torch.no_grad()
    @atomic
    def params_are_finite(self) -> bool:
        for param in self.wrapped_model.model.parameters():
            if not torch.all(torch.isfinite(param)):
                return False

        return True

    @atomic
    def sync_collate(self) -> None:
        self.wrapped_model.collate.sync(verbose=True)

    @torch.no_grad()
    @atomic
    def sync_state(self, force: bool = False) -> None:
        if force or self.local_epoch < self.global_epoch:
            t_start = time.monotonic()
            log.project_console.print('Sync state with other peers...', style='salmon1', justify='right')
            self.opt.load_state_from_peers()
            log.project_console.print(
                f'Sync is finished in {time.monotonic() - t_start:.4} sec', style='salmon1', justify='right'
            )
        else:
            log.project_console.print('No need to sync state with other peers', style='salmon1', justify='right')

    @torch.no_grad()
    @atomic
    def state_dict(self) -> Dict[str, Any]:
        return {
            'model': self.wrapped_model.model.state_dict(),
            'optimizer': self.opt.state_dict(),
            'scheduler': self.opt.state_averager.scheduler.state_dict(),
            'local_epoch': self.opt.local_epoch,
        }

    @torch.no_grad()
    @atomic
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.wrapped_model.model.load_state_dict(state_dict['model'])
        self.opt.load_state_dict(state_dict['optimizer'])
        self.opt.state_averager.scheduler.load_state_dict(state_dict['scheduler'])
        self.opt.state_averager.local_epoch = state_dict['local_epoch']

        if self.opt.offload_optimizer:
            state_averager = self.opt.state_averager
            offloaded_parameters = [
                param for group in state_averager.optimizer.param_groups for param in group['params']
            ]

            assert len(offloaded_parameters) == len(state_averager.main_parameters), \
                'Unable to load collaborative optimizer state dict'

            for main_param, offloaded_param in zip(state_averager.main_parameters, offloaded_parameters):
                offloaded_param.copy_(main_param, non_blocking=True)

    @torch.no_grad()
    @atomic
    def backup_state(self) -> None:
        t_start = time.monotonic()
        log.project_console.print(f'Backup the {self.local_epoch:_}-epoch collab state...',
                                  style='magenta', justify='right')
        torch.save(self.state_dict(), self.state_path)
        log.project_console.print(
            f'Backup done in {time.monotonic() - t_start:.4} sec', style='magenta', justify='right'
        )

    @torch.no_grad()
    @atomic
    def restore_from_backup(self, check_step: bool = False) -> None:
        if self.state_path.exists():
            t_start = time.monotonic()
            state_dict = torch.load(self.state_path)
            current_step = self.opt.local_epoch
            backup_step = state_dict['local_epoch']

            if not check_step or backup_step >= current_step:
                log.project_console.print(f'Restoring state from {backup_step:_}-epoch backup...',
                                          style='green', justify='right')
                self.load_state_dict(state_dict)
                log.project_console.print(
                    f'Collab sate is restored from backup in {time.monotonic() - t_start:.4} sec',
                    style='green', justify='right'
                )
            else:
                log.project_console.print(
                    'Bypassed restoring collab state from local backup - backup state is too old',
                    style='yellow', justify='right'
                )
        else:
            log.project_console.print('Backup does not exist', style='yellow', justify='right')

    def _fetch_status_dict(self) -> Optional[Dict]:
        if ((status_entry := self.dht_manager.dht.get(self.status_key, latest=True))
                and (status_dict := status_entry.value)):
            status_dict.pop(self.dht_manager.local_public_key, None)

            return status_dict


class AuxiliaryOptimizer(CollaborativeOptimizer):
    def __init__(self,
                 dht_manager: DHTManager,
                 wrapped_model: BaseModelWrapper,
                 args: Namespace,
                 common_status: Optional[CommonStatus] = None):
        super(AuxiliaryOptimizer, self).__init__(dht_manager=dht_manager,
                                                 wrapped_model=wrapped_model,
                                                 args=args,
                                                 batch_size_per_step=0 if args.as_active_peer else None,
                                                 verbose=args.verbose)

        self.auxiliary: bool = not args.as_active_peer
        self.stopped = threading.Event()
        self.averaging_thread: Optional[threading.Thread] = None
        self.last_reported_step: int = -1
        self.status = Status(name='Assistant', name_style='dark_violet', status_style=self._status_style,
                             common_status=common_status)

    @property
    def _is_time_to_backup(self) -> bool:
        return (
                self.local_epoch != self.last_reported_step
                and self.local_epoch != 0
                and self.args.backup_every_step
                and self.args.backup_every_step > 0
                and (self.local_epoch + 1) % self.args.backup_every_step == 0
        )

    @property
    def _status_style(self) -> str:
        return 'bright_blue' if self.allow_state_sharing else 'blue'

    def _assist_averaging_in_background(self) -> None:
        try:
            self.status.enable()

            with self.transaction:
                self.restore_from_backup()
                self.sync_state()

                if self.allow_state_sharing or self.args.backup_every_step:
                    self.backup_state()
                    self.last_reported_step = self.local_epoch

            self._assistant_loop()

        except KeyboardInterrupt:
            pass
        finally:
            self.stopped.set()
            self.status.update('Is Stopped', style='yellow')
            self.status.disable()

    def _assistant_loop(self) -> None:
        self.status.update('Pending...', style=self._status_style)

        while not self.stopped.is_set():
            try:
                with self.transaction:
                    if self.stopped.is_set():
                        return

                    self.status.update('Assist in averaging...', style=self._status_style)

                    self._update_state_sharing_status_step()
                    self._check_finiteness_step()
                    self.opt.step()
                    self.report_status()

                    if self._is_time_to_backup:
                        self._backup_step()

                    self.status.update('Pending...', style=self._status_style)

                time.sleep(self.args.assist_refresh)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                log.project_logger.exception(e, exc_info=True)

    def _update_state_sharing_status_step(self) -> None:
        if self.allow_state_sharing and self.num_peers == 1 and self.num_non_client_peers == 1:
            self.allow_state_sharing = False
            log.project_console.print(
                'From now, this auxiliary peer will not be able to share its state '
                'since it is no longer synchronized with one single active peer',
                style='yellow',
                justify='right'
            )

        elif self.allow_state_sharing and self.num_peers == 1 and self.num_client_peers == 1:
            log.project_console.print(
                'WARN: This auxiliary peer can no longer synchronize its state with a single one client-mode peer.',
                style='yellow',
                justify='right'
            )
            log.project_console.print(
                f'That is, new peers will be able to download the actual state '
                f'only for the {self.local_epoch} collaborative step',
                style='yellow',
                justify='right'
            )

        elif self.original_allow_state_sharing and not self.allow_state_sharing and self.num_peers != 1:
            self.status.update('Need to synchronize this auxiliary peer to enable state sharing',
                               style=self._status_style)

            self.restore_from_backup()
            self.sync_state()
            self.allow_state_sharing = True

            log.project_console.print(
                'State sharing is enabled for this auxiliary peer',
                style='green',
                justify='right'
            )
            self.status.update('Assist in averaging...', style=self._status_style)

    def _check_finiteness_step(self) -> None:
        if self.allow_state_sharing and not self.params_are_finite:
            log.project_console.print('Model parameters are not finite', style='red', justify='right')
            self.status.update('State recovering...', style='yellow')
            self.recover_state(sync=True)
            self.status.update('Assist in averaging...', style=self._status_style)

    def _backup_step(self) -> None:
        try:
            if self.auxiliary or not self.allow_state_sharing:
                self.status.update('Since this peer is not active, we need to sync it with others before backup...',
                                   style=self._status_style)

                self.sync_state(force=True)

                if not self.params_are_finite:
                    log.project_console.print(
                        'It is not possible to make a backup since the parameters '
                        'that have just been synchronized are not finite',
                        style='red', justify='right'
                    )
                    return

            self.status.update('Backup state...', style=self._status_style)
            self.backup_state()
            self.last_reported_step = self.local_epoch

        finally:
            self.status.update('Assist in averaging...', style=self._status_style)

    def start_assistant(self, attach: bool = False) -> None:
        if self.args.client_mode:
            log.project_console.print('Client-mode peer cannot assist in averaging', style='red', justify='right')
            return

        if attach:
            self._assist_averaging_in_background()
        else:
            self.averaging_thread = threading.Thread(name='AveragingAuxThread',
                                                     target=self._assist_averaging_in_background,
                                                     daemon=True)
            self.averaging_thread.start()

    def finish(self, join: bool = True) -> None:
        self.stopped.set()

        if self.averaging_thread.is_alive() and join:
            self.averaging_thread.join()
