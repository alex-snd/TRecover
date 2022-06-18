from typing import Dict, List, Tuple

from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.schema import BytesWithPublicKey, SchemaValidator
from hivemind.dht.validation import RecordValidatorBase
from hivemind.utils.logging import get_logger
from pydantic import BaseModel, StrictFloat, confloat, conint
from torch.optim.lr_scheduler import LambdaLR

logger = get_logger(__name__)


class LocalMetrics(BaseModel):
    step: conint(ge=0, strict=True)
    samples_per_second: confloat(ge=0.0, strict=True)
    samples_accumulated: conint(ge=0, strict=True)
    loss: StrictFloat
    mini_steps: conint(ge=0, strict=True)


class MetricSchema(BaseModel):
    metrics: Dict[BytesWithPublicKey, LocalMetrics]


def make_validators(run_id: str) -> Tuple[List[RecordValidatorBase], bytes]:
    signature_validator = RSASignatureValidator()
    validators = [SchemaValidator(MetricSchema, prefix=run_id), signature_validator]

    return validators, signature_validator.local_public_key


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
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
