import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

def read_pkl(path):
    pkl_path = path.split('.')[0] + '.pkl'
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def visualize(image, keypoints, bboxes):
    overlay = image.copy()
    for kp in keypoints:
        cv2.circle(overlay, (int(kp[0]), int(kp[1])), 20, (0, 200, 200),
                   thickness=2,
                   lineType=cv2.LINE_AA)

    for box in bboxes:
        cv2.rectangle(overlay, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (200, 0, 0),
                      thickness=2)

    return overlay

def Plot(data, is_dot=False):
    aug_image = data['image']
    if is_dot:
            aug_image = visualize(aug_image, data['keypoints'], data['bboxes'])
    aug_map = data['mask']
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
    img_list = []
    mask = np.zeros(img.shape[:2])
    for idy in range(len(yy) - 1):
        for idx in range(len(xx) - 1):
            mask[yy[idy]: yy[idy+1], xx[idx]:xx[idx+1]] = idx + idy * num_steps_hori
            img_list.append(img[yy[idy]: yy[idy+1], xx[idx]:xx[idx+1]])
    return img_list, mask


if __name__ == '__main__':
    path = '/home/edmund/projects/pics/1531053242.jpg'
    image = cv2.imread(path)
    img_list, mask = grid(image.copy(), num_steps_hori=4, num_steps_verti=3)
    plt.figure(1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    plt.figure(2)
    plt.imshow(cv2.cvtColor(img_list[1], cv2.COLOR_RGB2BGR))
    plt.figure()
    plt.imshow(cv2.cvtColor(img_list[2], cv2.COLOR_RGB2BGR))
    plt.figure()
    plt.imshow(cv2.cvtColor(img_list[7], cv2.COLOR_RGB2BGR))
    plt.figure()
    plt.imshow(mask)
    plt.show()
