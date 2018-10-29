import time
from pathlib import Path
import math
import imageio
import numpy as np
from scipy import ndimage
from skimage import color, exposure
from skimage.transform import resize

_data_pairs_cache = None

data_directory = (Path(__file__).parent / '../data/A').resolve()


def load_grape_data(sub_img_count=1, target_img_size=300):
    """Load the basic annotated images and their respective annotations
    Returns a tuple (images, density, counts) where images and density are ndarrays with shape (100, 1080, 1080) and
    counts is an ndarray with shape (100).
    """

    def to_blocks(img):
        if sub_img_count == 1:
            return [img]
        count = int(math.sqrt(sub_img_count))
        size = int(img.shape[0] / count)
        blocks = []
        for y in range(count):
            for x in range(count):
                blocks.append(img[(y*size): (y*size) + size,
                                  (x*size): (x*size) + size])
        return blocks

    def load_image(path):
        # green bar at the bottom of the files needs to be removed.
        without_green = imageio.imread(path)[:-10, :, :]
        grey_scale = color.rgb2grey(without_green)
        img_cummulated_distribution_function, bin_centers = exposure.cumulative_distribution(
            grey_scale)
        normalized = np.interp(grey_scale, bin_centers,
                               img_cummulated_distribution_function)
        return normalized

    def load_annotations(path, img):
        file = open(str(path))
        annotations = np.empty(img.shape[:2])
        for line in file:
            x, y = line.split(' ')
            annotations[int(y), int(x)] = 1
        return annotations

    def load_files(files):
        img = load_image(files[0])
        annotations = load_annotations(files[1], img)
        return [img, annotations]

    def image_name_to_annotation(img):
        return data_directory / (img.stem + '.annotations.txt')

    def load_data_pairs():
        global _data_pairs_cache
        if _data_pairs_cache is None:
            # Generator of pairs of img_filename and annotation_filename
            file_pairs = ([img, image_name_to_annotation(img)] for img in data_directory.glob('*.jpg')
                          if image_name_to_annotation(img).exists())
            _data_pairs_cache = list(map(load_files, file_pairs))
        return _data_pairs_cache

    data_pairs = load_data_pairs()

    images, locations = map(np.array, zip(*data_pairs))

    # Crop the center square of the images and annotations
    unused_count, height, width, *_ = images.shape
    offset = int((width - height) / 2)
    images = images[:, :, offset:-offset]
    locations = locations[:, :, offset:-offset]

    def scale(img): return resize(img, (target_img_size, target_img_size),
                                  anti_aliasing=True, mode='constant')

    def scale_annotation(annotation):
        factor = target_img_size / len(annotation)
        new_annotation = np.zeros((target_img_size, target_img_size))
        for y, x in zip(*np.nonzero(annotation)):
            new_annotation[int(y * factor)][int(x * factor)
                                            ] += annotation[y, x]
        return new_annotation

    # Split images into equal sized blocks
    images = np.array([scale(block)
                       for img in images for block in to_blocks(img)])

    locations = np.array(
        [scale_annotation(block) for loc in locations for block in to_blocks(loc)])
    density = np.array([density_map(loc) for loc in locations])
    counts = np.sum(locations, axis=(1, 2))

    return images, density, counts


def density_map(locations):
    return ndimage.gaussian_filter(locations, sigma=5, mode="constant")


if __name__ == "__main__":
    X, Y_locations, Y_counts = load_grape_data(16)
    print(X.shape)
    print(Y_locations.shape)
    print(Y_counts.shape)
