from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to [1], the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.


    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (Optional[str]): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.

    Shape:
        - Input: :math:`(N, Timestep, C)` where C = number of classes.
        - Target: :math:`(N x Timestep)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    """

    def __init__(self, alpha: float = 0.5, gamma: Optional[float] = 2.0,
                 reduction: Optional[str] = 'mean') -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: Optional[float] = gamma
        self.reduction: Optional[str] = reduction
        self.eps: float = 1e-6

    def forward(self, input, target):

        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=-1) + self.eps

        # create the labels one hot tensor
        target_one_hot = torch.zeros_like(input_soft)
        target_one_hot = target_one_hot.scatter_(1, target.data.unsqueeze(1), 1.0) + self.eps

        mask = torch.nonzero(target.data == 0, as_tuple=False)
        if mask.dim() > 0:
            target_one_hot.index_fill_(0, mask.squeeze(), 0.0)

        # compute the actual focal loss
        weight = torch.pow(1. - input_soft, self.gamma)
        focal = -1 * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target_one_hot * focal, dim=-1)

        loss = -1
        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}"
                                      .format(self.reduction))
        return loss