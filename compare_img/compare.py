# pylint: disable=no-member, C0103
"""
From: Jan Egermaier
File for example how to compare image with python
"""
#Pixel comparing
import math
import operator
from typing import Any
from functools import reduce
from collections import Counter
import numpy as np
import cv2 as cv
from skimage.metrics import structural_similarity
from PIL import ImageChops, Image


def compare_ssim(img1, img2, ssim=1.0):
    """
    Method for comparing image with SSIM.
    Source: https://en.wikipedia.org/wiki/Structural_similarity

    Args:
        img1 (_type_): First image for compare
        img2 (_type_): Second image for compare
        ssim (float, optional): SSIM how can image be different. Defaults to 1.0.

    Returns:
        _type_: Return true or false if image was diff.
    """
    my_img1 = cv.imread(img1, 1)
    my_img2 = cv.imread(img2, 1)

    gray_img1: Any = cv.cvtColor(my_img1, cv.COLOR_BGR2GRAY)
    gray_img2: Any = cv.cvtColor(my_img2, cv.COLOR_BGR2GRAY)

    score = structural_similarity(gray_img1, gray_img2, full=True)

    if ssim > score:
        return False

    return True

def chops_diff(img1, img2):
    """
    Use ImageChops, there is no image similarity algorhytm (except pixel-wise).
    The ImageChops module contains a number of arithmetical image operations,
    called channel operations (“chops”).
    These can be used for various purposes, including special effects,
    image compositions, algorithmic painting, and more.
    Source: http://effbot.org/imagingbook/imagechops.htm

    Args:
        img1 (image): First image for compare
        img2 (image): Second image for compare
    Returns:
        percent_result = lower better (0 = exact image, 1 = total diff)
    """
    im1 = Image.open(img1).convert("RGBA")
    im2 = Image.open(img2).convert("RGBA")

    diff = ImageChops.difference(im1, im2).convert("RGB")
    percent_result = np.mean(np.array(diff))

    diff.save("diff.jpeg", "JPEG")
    return percent_result


def rms_diff(img1, img2) -> float:
    """
    Calculate the root-mean-square difference between two images.
    Source: https://en.wikipedia.org/wiki/Root-mean-square_deviation

    Args:
        img1 (image): First image for compare
        img2 (image): Second image for compare

    Returns:
        float : Number how diff image are
    """
    im1 = Image.open(img1).convert("RGBA")
    im2 = Image.open(img2).convert("RGBA")

    diff = ImageChops.difference(im1, im2).convert("RGB")
    h = diff.histogram()

    diff.save("diff.jpeg", "JPEG")

    # calculate rms
    return math.sqrt(reduce(operator.add,
        map(lambda h, i: h*(i**2), h, range(256))
    ) / (float(im1.size[0]) * im1.size[1]))

def euclidean_distance_diff(img1, img2) -> float:
    """Euclidean distance diff:
    Sources:
    https://www.analyticsvidhya.com/blog/2021/03/a-beginners-guide-to-image-similarity-using-python/

    Args:
        img1 (image): First image for compare
        img2 (image): Second image for compare

    Returns:
        float: Euclidean distance between two points in n-dimensional space
    """
    # Generate histogram for img 1
    im1 = Image.open(img1)
    im1_arr = np.asarray(im1)

    one_dimensional_arr = im1_arr.flatten()
    rh1 = Counter(one_dimensional_arr)

    H1 = []
    for i in range(256):
        if i in rh1.keys():
            H1.append(rh1[i])
        else:
            H1.append(0)

    # Generate histogram for img 2
    im2 = Image.open(img2)
    im2_arr = np.asarray(im2)

    one_dimensional_arr = im2_arr.flatten()
    rh2 = Counter(one_dimensional_arr)

    H2 = []
    for i in range(256):
        if i in rh2.keys():
            H2.append(rh2[i])
        else:
            H2.append(0)

    # Euclidian distance func
    distance = 0
    for i in range(len(H1)):
        distance += np.square(H1[i] - H2[i])

    return np.sqrt(distance)
