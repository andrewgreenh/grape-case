import numpy as np
import math
from scipy import ndimage
from data import density_map
import keras

import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')


class Augmentor:
    def __init__(self, base_data):
        self.base_data = base_data
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
        img_index = index % self.base_data_count
        t_index = int(index / self.base_data_count)
        image = self.base_data[0][img_index]
        locations = self.base_data[3][img_index]

        t = self.get_transformation(t_index)

        t_image = t(image)
        t_locations = t(locations)
        t_density = density_map([t_locations])[0]
        t_count = np.sum(t_locations)

        return t_image, t_density, t_count, t_locations

    def augmentation_sequence(self, batch_size, get_x, get_y):
        return Augmentation_Sequence(self.base_data, batch_size, get_x, get_y)


class Augmentation_Sequence(keras.utils.Sequence):
    def __init__(self, base_data, batch_size, get_x, get_y):
        self.augmentor = None
        self.base_data = base_data
        self.batch_size = batch_size
        self.get_x = get_x
        self.get_y = get_y

    def get_augmentor(self):
        if self.augmentor is None:
            self.augmentor = Augmentor(self.base_data)

        return self.augmentor

    def __len__(self):
        return math.ceil(self.get_augmentor().augmented_count / self.batch_size)

    def __getitem__(self, index):
        start_index_of_batch = index * self.batch_size
        current_batch_x = []
        current_batch_y = []
        for index_in_batch in range(self.batch_size):
            img_index = start_index_of_batch + index_in_batch
            if img_index > self.get_augmentor().augmented_count - 1:
                break
            point = self.get_augmentor().get_data_point(img_index)

            current_batch_x.append(self.get_x(point))
            current_batch_y.append(self.get_y(point))

        X_array = np.array(current_batch_x)
        X_array = X_array.reshape((*X_array.shape, 1))
        Y_array = np.array(current_batch_y)
        return (X_array, Y_array)
