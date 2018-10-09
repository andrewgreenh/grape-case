from pathlib import Path

import imageio
import numpy as np
import time
from scipy import ndimage
from skimage.transform import resize

_data_pairs_cache = None

data_directory = (Path(__file__).parent / '../data/A').resolve()


def load_grape_data(transform_data=None):
    """Load the basic annotated images and their respective annotations
    Returns a tuple (images, annotations) where images is an ndarray with shape (1080, 1080, 3) and
    annotations is an ndarray with shape (1080, 1080) where each pixel that is the center of a grape
    has a one, while the rest has a 0.
    """

    def load_image(path):
        # green bar at the bottom of the files needs to be removed.
        return imageio.imread(path)[:-10, :, :]

    def load_annotations(path, img):
        file = open(path)
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

    transformed_data_pairs = list(data_pairs) if transform_data is None else list(
        map(transform_data, data_pairs))

    images = np.array(list(map(lambda pair: pair[0], transformed_data_pairs)))
    annotations = np.array(
        list(map(lambda pair: pair[1], transformed_data_pairs)))

    unused_count, height, width, *_ = images.shape
    offset = int((width - height) / 2)

    # Crop the center square of the images and annotations
    images = images[:, :, offset:-offset, ...]
    annotations = annotations[:, :, offset:-offset]

    return images, annotations


class Augmentor:
    def __init__(self, base_data=None):
        if base_data is None:
            base_data = load_grape_data()
        self.rotation_transforms = [self.rotate(deg) for deg in range(-10, 11)]
        self.flip_transforms = [self.flip(True), self.flip(False)]
        self.zoom_and_crop_transforms = []
        for zoom_percentage in range(0, 41, 10):
            for x_crop_location_percentage in range(0, 101, 25):
                for y_crop_location_percentage in range(0, 101, 25):
                    self.zoom_and_crop_transforms.append(
                        self.zoom_and_crop(1 + zoom_percentage / 100, x_crop_location_percentage / 100, y_crop_location_percentage / 100))

        self.transform_count = len(self.rotation_transforms) * \
            len(self.flip_transforms) * len(self.zoom_and_crop_transforms)

        self.annotation_showcase = [
            [self.flip(True), self.flip(False)],
            [self.rotate(-10), self.rotate(10)],
            [self.zoom_and_crop(1.4, 0, 0), self.zoom_and_crop(1.4, 1, 1)],
        ]
        self.base_data = base_data
        self.base_data_count = len(self.base_data[0])
        self.augmented_count = self.base_data_count * self.transform_count

    def rotate(self, deg):
        return lambda img: ndimage.rotate(img, deg, reshape=False)

    def flip(self, should_flip):
        return lambda img: img if should_flip else np.fliplr(img)

    def zoom_and_crop(self, zoom, xoffset, yoffset):
        def apply_zoom_and_crop(img):
            zoom_levels = [zoom, zoom, 1] if len(
                img.shape) == 3 else [zoom, zoom]
            zoomed = ndimage.zoom(img, zoom_levels)
            height = zoomed.shape[0]
            initial_height = img.shape[0]
            width = zoomed.shape[1]
            initial_width = img.shape[1]
            xoffset_px = int((width - initial_width) * xoffset)
            yoffset_px = int((height - initial_height) * yoffset)
            if len(zoomed.shape) == 3:
                return zoomed[yoffset_px:yoffset_px + initial_height, xoffset_px: xoffset_px + initial_width, :]
            else:
                return zoomed[yoffset_px:yoffset_px + initial_height, xoffset_px: xoffset_px + initial_width]
        return apply_zoom_and_crop

    def get_transformation(self, index):
        flip_index = index % len(self.flip_transforms)
        rotation_index = int(index / len(self.flip_transforms)
                             ) % len(self.rotation_transforms)
        zoom_index = int(index / (len(self.flip_transforms) *
                                  len(self.rotation_transforms))) % len(self.zoom_and_crop_transforms)
        flip_tranform = self.flip_transforms[flip_index]
        rotation_transform = self.rotation_transforms[rotation_index]
        zoom_transform = self.zoom_and_crop_transforms[zoom_index]

        return lambda img: flip_tranform(rotation_transform(zoom_transform(img)))

    def get_data_point(self, index):
        img_index = int(index / self.transform_count)
        t_index = index % self.transform_count
        x = self.base_data[0][img_index]
        y = self.base_data[1][img_index]
        t = self.get_transformation(t_index)

        return t(x), t(y)
