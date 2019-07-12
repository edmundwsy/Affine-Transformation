import albumentations as A
import cv2
import matplotlib.pyplot as plt
from grid import grid
from affine import AffineTransform


def random_crop(img, msk, height=512, width=512):
    aug = A.Compose([A.RandomCrop(height=height, width=width)])
    data = aug(image=img, mask=msk)
    return data['image'], data['mask']


def horizontal_flip(img, msk):
    aug = A.Compose([A.HorizontalFlip(always_apply=True)])
    data = aug(image=img, mask=msk)
    return data['image'], data['mask']


def vertical_flip(img, msk):
    aug = A.Compose([A.VerticalFlip(always_apply=True)])
    data = aug(image=img, mask=msk)
    return data['image'], data['mask']


def rotate45(img, msk):
    aug = A.Compose([A.Rotate(limit=(45, 45), border_mode=cv2.BORDER_CONSTANT, always_apply=True)])
    data = aug(image=img, mask=msk)
    return data['image'], data['mask']


def rotate30(img, msk):
    aug = A.Compose([A.Rotate(limit=(30, 30), border_mode=cv2.BORDER_CONSTANT, always_apply=True)])
    data = aug(image=img, mask=msk)
    return data['image'], data['mask']


def rotate60(img, msk):
    aug = A.Compose([A.Rotate(limit=(60, 60), border_mode=cv2.BORDER_CONSTANT, always_apply=True)])
    data = aug(image=img, mask=msk)
    return data['image'], data['mask']


def affine(img, msk, param):
    aug = A.Compose([AffineTransform(param=param, border_mode=cv2.BORDER_CONSTANT, always_apply=True)])
    data = aug(image=img, mask=msk)
    return data['image'], data['mask']


if __name__ == '__main__':
    path = "/home/edmund/projects/pics/1531053735.jpg"
    img = cv2.imread(path)
    mask = grid(img)
    # plt.figure()
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # plt.figure()
    # plt.imshow(mask)
    # plt.show()
    img_, msk_ = affine(img, mask, [-0.01, 0.01, 0.1, -0.01, 0.01, 0.01])
    # img_ = F_affine(img, [0.1, 0.1, 1, 0.2, 0.2, 1])
    plt.figure()
    plt.imshow(img_)
    # plt.imshow(cv2.cvtColor(img_, cv2.COLOR_RGB2BGR))
    plt.figure()
    plt.imshow(msk_)
    plt.show()



