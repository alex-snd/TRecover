import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss


class CustomCrossEntropyLoss(torch.nn.Module):
    def __init__(self, ignore_index: int = -1):
        super(CustomCrossEntropyLoss, self).__init__()

        self.criterion = CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, *tensors) -> Tensor:
        tgt, tgt_out = tensors[2], tensors[6]

        return self.criterion(tgt_out, tgt)


class CustomPenaltyLoss(torch.nn.Module):
    def __init__(self, ignore_index: int = -1):
        super(CustomPenaltyLoss, self).__init__()

        self.criterion = CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, *tensors) -> Tensor:
        src, _, tgt, src_pad_mask, _, _, tgt_out = tensors

        penalty_size = src_pad_mask.shape[0] * src_pad_mask.shape[1]
        penalty_tensor = -1.0 * torch.flatten(F.log_softmax(tgt_out, dim=-1) * ~src.bool())[:penalty_size]

        return self.criterion(tgt_out, tgt) + penalty_tensor.mean()
