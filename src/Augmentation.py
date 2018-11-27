import numpy as np
import math
from scipy import ndimage
from data import density_map
import keras

import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

# Each array contains 7 levels of augmentation strength. When a lower factor is needed
# earlier step counts are taken from the arrays.
rotation_steps = np.array([3, 5, 9, 13, 15, 17, 21])
zoom_steps = np.array([1, 3, 4, 5, 4, 5, 5])
x_crop_steps = np.array([1, 2, 4, 4, 5, 5, 5])
y_crop_steps = np.array([1, 2, 4, 4, 5, 5, 5])

factors = rotation_steps * zoom_steps * x_crop_steps * y_crop_steps * 2


def get_transformation_intensity_index(factor):
    global factors
    if factor > factors[-1]:
        return len(factors) - 1
    return np.argmax(factors > factor)


# This class implements the data augmentation. Simply hand over the base data and a target image count
# and this class will be setup enhance the training data with slightly changed images
class Augmentor:
    def __init__(self, base_data, target_image_count):
        global rotation_steps, zoom_steps, x_crop_steps, y_crop_steps

        self.base_data = base_data
        self.base_data_count = len(base_data[0])
        self.target_image_count = target_image_count

        print('target_image_count', target_image_count)

        factor = target_image_count / self.base_data_count

        print('desired augmentation factor', factor)

        transform_indensity_index = get_transformation_intensity_index(factor)

        self.rotation_transforms = [self.rotate(
            deg) for deg in np.linspace(-10, 10, rotation_steps[transform_indensity_index])]
        self.flip_transforms = [self.flip(True), self.flip(False)]
        self.zoom_and_crop_transforms = []
        for zoom_percentage in np.linspace(0, 40, zoom_steps[transform_indensity_index]):
            for x_crop_location_percentage in np.linspace(0, 100, x_crop_steps[transform_indensity_index]):
                for y_crop_location_percentage in np.linspace(0, 100, x_crop_steps[transform_indensity_index]):
                    self.zoom_and_crop_transforms.append(
                        self.zoom_and_crop(1 + zoom_percentage / 100, x_crop_location_percentage / 100, y_crop_location_percentage / 100))

        self.transform_count = len(self.rotation_transforms) * \
            len(self.flip_transforms) * len(self.zoom_and_crop_transforms)

        self.annotation_showcase = [
            [self.flip(True), self.flip(False)],
            [self.rotate(-10), self.rotate(10)],
            [self.zoom_and_crop(1.4, 0, 0), self.zoom_and_crop(1.4, 1, 1)],
        ]

        self.augmented_count = self.base_data_count * self.transform_count

    # Factory of a transformation function that will rotate an image
    def rotate(self, deg):
        return lambda img: ndimage.rotate(img, deg, reshape=False)

    # Factory of a transformation function that will flip an image
    def flip(self, should_flip):
        return lambda img: img if should_flip else np.fliplr(img)

    # Factory of a transformation function that will zoom and crop an image
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

    # Get the transformation that is used for the image at an index
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

    # Get the image, the density map, the count and the annotation locations of an index
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

    # Return a sequence that can be used by keras.model.fit_generator.
    def augmentation_sequence(self, batch_size, get_x, get_y):
        return Augmentation_Sequence(self.base_data, self.target_image_count, batch_size, get_x, get_y)


# Sequence of augmented images that can be used by keras.model.fit_generator
class Augmentation_Sequence(keras.utils.Sequence):
    def __init__(self, base_data, target_image_count, batch_size, get_x, get_y):
        self.augmentor = None
        self.base_data = base_data
        self.target_image_count = target_image_count
        self.batch_size = batch_size
        self.get_x = get_x
        self.get_y = get_y

    def get_augmentor(self):
        if self.augmentor is None:
            self.augmentor = Augmentor(self.base_data, self.target_image_count)

        return self.augmentor

    def __len__(self):
        return math.ceil(min(self.get_augmentor().augmented_count, self.target_image_count) / self.batch_size)

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
