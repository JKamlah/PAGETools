from typing import Tuple, Union

import numpy as np
import cv2


def background_calc_dispatch_table(mode: str):
    dispatch_table = {
        "dominant": calc_dominat_color,
        "mean": calc_mean_color,
        "median": calc_median_color
    }

    return dispatch_table[mode]


def calc_dominat_color(img: np.array) -> Tuple[int]:
    """Calculates the dominant color of an image using bincounts

    :param img:
    :return:
    """
    two_dim = img.reshape(-1, img.shape[-1])
    color_range = (256,)*3
    one_dim = np.ravel_multi_index(two_dim.T, color_range)
    return tuple([int(c) for c in np.unravel_index(np.bincount(one_dim).argmax(), color_range)])


def calc_mean_color(img: np.array) -> Tuple[int]:
    """

    :param img:
    :return:
    """
    return img.mean(axis=0).mean(axis=0)


def calc_median_color(img: np.array) -> np.ndarray:
    """

    :param img:
    :return:
    """
    return np.median(np.median(img, axis=0), axis=0)


def rotate_img(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]], display_all: bool = True) \
        -> np.ndarray:
    """
    :param image:
    :param angle:
    :param background:
    :return:
    """
    # author:   Adrian Rosebrock
    # website:  http://www.pyimagesearch.com
    # source:   https://github.com/PyImageSearch/imutils/convenience.py
    # license:  MIT
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    if display_all:
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # perform the actual rotation and return the image
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH), borderValue=background)
    else:
        return cv2.warpAffine(image, M, (h, w), borderValue=background)
