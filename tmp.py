# -*- coding: utf-8 -*-
# @Time     : 16:37
# @File     : tmp.py
# @Software : PyCharm
import cv2
import numpy as np
from albumentations import *
image = cv2.imread('IMG_319.jpg')
cv2.imshow('im', image)
cv2.waitKey(0)
