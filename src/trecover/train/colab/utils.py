from typing import List, Dict, Any, Union, Tuple, Optional

import argparse_dataclass
from hivemind import choose_ip_address
from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.schema import BytesWithPublicKey, SchemaValidator
from hivemind.dht.validation import RecordValidatorBase
from multiaddr import Multiaddr
from pydantic import BaseModel, StrictFloat, confloat, conint
from torch.optim.lr_scheduler import LambdaLR


class ArgumentParser(argparse_dataclass.ArgumentParser):
    def parse_known_args(self, *args, **kwargs) -> Tuple[argparse_dataclass.OptionsType, List[str]]:
        """ Parse known arguments and return as the dataclass type. """

        namespace, others = super().parse_known_args(*args, **kwargs)
        kwargs = {k: v for k, v in vars(namespace).items() if v != argparse_dataclass.MISSING}

        return self._options_type(**kwargs), others


class LocalMetrics(BaseModel):
    step: conint(ge=0, strict=True)
    samples_per_second: confloat(ge=0.0, strict=True)
    samples_accumulated: conint(ge=0, strict=True)
    loss: StrictFloat
    mini_steps: conint(ge=0, strict=True)


class MetricSchema(BaseModel):
    metrics: Dict[BytesWithPublicKey, LocalMetrics]


def make_validators(experiment_prefix: str) -> Tuple[List[RecordValidatorBase], bytes]:
    signature_validator = RSASignatureValidator()
    validators = [SchemaValidator(MetricSchema, prefix=experiment_prefix), signature_validator]

    return validators, signature_validator.local_public_key


def parse_dataclasses(*dataclasses: Union[Any, Tuple[Any, ...]],
                      args: Optional[List[str]] = None
                      ) -> Union[Any, Tuple[Any, ...]]:
    return (ArgumentParser(dc).parse_known_args(args)[0] for dc in dataclasses)


def get_initial_peers(visible_maddrs: List[Multiaddr], only_p2p: bool) -> Optional[List[str]]:
    if only_p2p:
        return [f'/p2p/{addr}' for addr in {addr['p2p'] for addr in visible_maddrs}] or None

    elif available_ips := [Multiaddr(addr) for addr in visible_maddrs if 'ip4' in addr or 'ip6' in addr]:
        preferred_ip = choose_ip_address(available_ips)
        return [str(addr) for addr in visible_maddrs if preferred_ip in str(addr)] or None


# TODO docs type hints
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
