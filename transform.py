# -*- coding: utf-8 -*-
# @Time     : 17:20
# @File     : transform.py
# @Software : PyCharm
import numpy as np


class Transform(object):
    def __init__(self, aug_list, prob_list=None, default_mask=None, iter_time=1):
        """
        :param iter_time: time you want to iteration to choose a random opt
        :type iter_time: int
        """
        assert np.sum(prob_list) == 1, "prob list should sum to 1."
        assert len(aug_list) == len(prob_list), 'prob list should has the same len as aug list.'
        self.aug_list = aug_list
        self.prob_list = prob_list
        self.fixed_mask = default_mask
        self.iter_time = iter_time

    def do_aug(self, aug, image, mask):
        output = aug(image, mask)
        return output[0], output[1]

    def random_aug(self, image, mask=None, iter_time=None):
        """
        :param iter_time: set iter_time to -1 to use all the affine methods
        :type iter_time: int
        :return: image, mask
        """
        if self.fixed_mask == None and mask == None:
            raise ValueError('self.fixed mask and mask should not both be None')
        elif mask == None:
            mask = self.fixed_mask
        if not iter_time:
            iter_time = self.iter_time
        if iter_time == -1:
            iter_time = len(self.aug_list)
        applied_aug_list = np.random.choice(self.aug_list, size=iter_time, replace=False, p=self.prob_list)
        for aug in applied_aug_list:
            image, mask = self.do_aug(aug, image, mask)
        return image, mask
