import math
from functools import wraps
from warnings import warn

import cv2
import numpy as np

from albumentations.augmentations.transforms import DualTransform


def affine(img, param=[0, 0, 0, 0, 0, 0], interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT):
    h0, w0 = img.shape[:2]
    center_square = np.float32((h0, w0)) // 2
    square_size = min((h0, w0)) // 3
    pts = np.float32([center_square + square_size,
                    [center_square[0] + square_size, center_square[1] - square_size],
                    center_square - square_size])
    dst = pts + np.float32([
        [square_size * param[0], square_size * param[1]],
        [square_size * param[2], square_size * param[3]],
        [square_size * param[4], square_size * param[5]]])

    height = 3 * max(abs(dst[0][0] - center_square[0]),
                     abs(dst[1][0] - center_square[0]),
                     abs(dst[2][0] - center_square[0]),
                     h0 / 3)
    width = 3 * max(abs(dst[0][1] - center_square[1]),
                    abs(dst[1][1] - center_square[1]),
                    abs(dst[2][1] - center_square[1]),
                    w0 / 3)
    matrix = cv2.getAffineTransform(pts, dst)
    img = cv2.warpAffine(img, matrix, (int(width), int(height)),
                         flags=interpolation, borderMode=border_mode)
    return img


class AffineTransform(DualTransform):
    """
    实行仿射变换，由于不知仿射变换的极坐标表示，因此目前先返回为零
    """
    def __init__(self, param, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101,
                 value=None, always_apply=False, p=.5):
        super(AffineTransform, self).__init__(always_apply, p)
        self.interpolation = interpolation
        self.border_mode = border_mode
        # self.value = value
        self.param = param

    def apply(self, img, param=[0, 0, 1, 0, 0, 1], interpolation=cv2.INTER_LINEAR, **params):
        return affine(img, self.param, self.interpolation, self.border_mode)

    def get_params(self):
        return {'param': self.param}

    # def apply_to_bbox(self, bbox, angle=0, **params):
    #     return F.bbox_rotate(bbox, angle, **params)

    def apply_to_keypoint(self, keypoint, param=[0,0,1,0,0,1], **params):
        return F.keypoint_affine(keypoint, param, **params)

    def get_transform_init_args_names(self):
        return ('interpolation', 'border_mode')
