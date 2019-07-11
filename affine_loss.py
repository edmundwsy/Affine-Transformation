# -*- coding: utf-8 -*-
# @Time     : 18:10
# @File     : affine_loss.py
# @Software : PyCharm
from torch.nn.modules import Module
import numpy as np


class AffineLoss(Module):
    def __init__(self, use_fg, device):
        super(AffineLoss, self).__init__()
        self.device = device
        self.use_fg = use_fg

    def forward(self, origin_density, origin_mask, pre_density, new_mask):
        """
        :param pre_density: predicted density
        :param new_mask: affine mask (from origin_mask)
        """
        mask_max = np.max(origin_mask)
        assert mask_max == np.max(new_mask), 'The old and new masks should have the same maxim'
        origin_count = np.zeros(mask_max + 1)
        for i in range(origin_density):
            for j in range(origin_density[0]):
                origin_count[origin_mask[i, j]] += origin_density[i, j]
        new_count = np.zeros(mask_max + 1)
        for i in range(pre_density):
            for j in range(pre_density[0]):
                new_count[origin_mask[i, j]] += pre_density[i, j]
        affine_loss = np.sum(abs(new_count - origin_count))
        return affine_loss
