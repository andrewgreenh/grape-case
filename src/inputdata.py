from skimage import color
import numpy as np
from scipy import ndimage

from data import load_grape_data, Augmentor


def blur_annotation(annotation):
    return ndimage.gaussian_filter(annotation, sigma=10, mode="constant")


def to_grayscale(img):
    return color.rgb2grey(img)


def full_pictures_with_counts():
    def transform(pair):
        x, y = pair
        return (to_grayscale(x), np.sum(y))
    return load_grape_data(transform)


def full_pictures_with_density_map():
    def transform(pair):
        x, y = pair
        return (to_grayscale(x), blur_annotation(y))
    return load_grape_data(transform)


def to_blocks(img, count=5):
    size = int(img.shape[0] / count)
    blocks = []
    for y in range(count):
        for x in range(count):
            blocks.append(img[(y*size): (y*size) + size,
                              (x*size): (x*size) + size])
    return blocks


def split_pictures_with_counts():
    def transform(pair):
        x, y = pair
        return (to_grayscale(x), y)

    X, Y = load_grape_data(transform)
    new_X = np.array([block for img in X for block in to_blocks(img)])
    new_Y = np.array([np.sum(block) for img in Y for block in to_blocks(img)])
    return new_X, new_Y


def split_pictures_with_density_map():
    def transform(pair):
        x, y = pair
        return (to_grayscale(x), blur_annotation(y))

    X, Y = load_grape_data(transform)
    new_X = np.array([block for img in X for block in to_blocks(img)])
    new_Y = np.array([block for img in Y for block in to_blocks(img)])
    return new_X, new_Y
