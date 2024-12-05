import cv2
import numpy as np
import pandas as pd

def is_grey_scale(img):
    w, h = img.shape[:2]
    for i in range(w):
        for j in range(h):
            r, g, b = img[i,j]
            if r != g != b:
                print(img[i, j])
                print(i, j)
                return False
    return True


def image_color_clust(img, k, output_path, debug=False):
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    if debug:
        cv2.imwrite(output_path + 'img_quant.jpg', res2)
    return res2


def get_background_color(img, debug=False):
    top = img[:3, :]
    top = np.reshape(top, (img.shape[1], 3, 3))
    bottom = img[img.shape[0] - 4: img.shape[0] - 1, :]
    bottom = np.reshape(bottom, (img.shape[1], 3, 3))
    left = img[:, :3]
    right = img[:, img.shape[1] - 4: img.shape[1] - 1]
    img_border = np.concatenate((top, bottom, left, right))
    if debug:
        cv2.imwrite('image_output/img_border.jpg', img_border)
    colors, count = np.unique(img_border.reshape(-1,img_border.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]
