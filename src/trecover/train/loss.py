import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss


class CustomCrossEntropyLoss(torch.nn.Module):
    # TODO docs

    def __init__(self, ignore_index: int = -1):
        super(CustomCrossEntropyLoss, self).__init__()

        self.criterion = CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, *tensors) -> Tensor:
        tgt, tgt_out = tensors[2], tensors[6]

        return self.criterion(tgt_out, tgt)


class CustomPenaltyLoss(torch.nn.Module):
    # TODO docs

    def __init__(self, coefficient: float = 1.0, ignore_index: int = -1):
        super(CustomPenaltyLoss, self).__init__()

        self.coefficient = coefficient
        self.criterion = CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, *tensors) -> Tensor:
        src, _, tgt, src_pad_mask, _, _, tgt_out = tensors

        penalty_tensor = -1.0 * F.log_softmax(tgt_out, dim=-1) * ~src.bool()

        if isinstance(src_pad_mask, Tensor):
            penalty_tensor = torch.masked_select(penalty_tensor, mask=~src_pad_mask.view(-1, 1))

        return self.criterion(tgt_out, tgt) + self.coefficient * penalty_tensor.mean()
