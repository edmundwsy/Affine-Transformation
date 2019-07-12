import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle


def read_pkl(path):
    pkl_path = path.split('.')[0] + '.pkl'
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def visualize(image, keypoints, bboxes, is_box):
    overlay = image.copy()
    for kp in keypoints:
        cv2.circle(overlay, (int(kp[0]), int(kp[1])), 20, (0, 200, 200),
                   thickness=2,
                   lineType=cv2.LINE_AA)
    if is_box:
        for box in bboxes:
            cv2.rectangle(overlay, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (200, 0, 0),
                          thickness=2)

    return overlay


def Plot(image, mask, keypoints, bboxes, is_dot=False):
    aug_image = image
    aug_image = visualize(aug_image, keypoints, bboxes, is_dot)
    aug_map = mask
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
    plt.figure(figsize=(10, 10))
    plt.imshow(aug_map)
    plt.tight_layout()
    plt.show()


def grid(img, num_steps_hori=3, num_steps_verti=3):
    """
    input a big image, and return image list and masks
    :param img: input image
    :param num_steps_hori:  number of horizontal small images
    :param num_steps_verti:  number of vertical small images
    :return:
    """
    height, width = img.shape[:2]
    xx = [0, ]
    x_step = width // num_steps_hori
    prev = 0
    for idx, x in enumerate(range(0, width, x_step)):
        end = x + x_step
        if end > width:
            # end = width
            cur = width
        else:
            cur = prev + x_step
            # print(cur)
        xx.append(cur)
        prev = cur

    y_step = height // num_steps_verti
    yy = [0, ]
    prev = 0
    for idx, y in enumerate(range(0, height, y_step)):
        end = y + y_step
        if end > height:
            # end = height
            cur = height
        else:
            cur = prev + y_step
        yy.append(cur)
        prev = cur
    mask = np.zeros(img.shape[:2])
    for idy in range(len(yy) - 1):
        for idx in range(len(xx) - 1):
            mask[yy[idy]: yy[idy+1], xx[idx]:xx[idx+1]] = idx + idy * num_steps_hori + 1
            # 这里+1防止padding和编号为0的mask重合
            # img_list.append(img[yy[idy]: yy[idy+1], xx[idx]:xx[idx+1]])
    return mask


if __name__ == '__main__':
    path = 'IMG_319.jpg'
    image = cv2.imread(path)
    mask = grid(image.copy(), num_steps_hori=4, num_steps_verti=3)
    plt.figure(1)
