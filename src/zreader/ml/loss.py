import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.criterion = CrossEntropyLoss(ignore_index=-1)

    def forward(self, tgt_out: Tensor, tgt: Tensor, src: Tensor) -> Tensor:
        return self.criterion(tgt_out, tgt) + torch.mean(-1.0 * F.log_softmax(tgt_out, dim=-1) * ~src.bool())
