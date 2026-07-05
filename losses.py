import torch.nn.functional as F
from torch import nn


class SaliencyKLLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred_logits, target):
        B = pred_logits.size(0)
        pred = pred_logits.view(B, -1)
        targ = target.view(B, -1)

        log_pred = F.log_softmax(pred, dim=1)
        targ = targ / (targ.sum(dim=1, keepdim=True) + self.eps)

        return F.kl_div(log_pred, targ, reduction='batchmean')
