from Trainer import Trainer
import numpy as np
from data import scale_annotation, density_map

from .model import build_model, shape

_image_split = 1
_image_size = shape[0]


def _get_validation_data(data, testing_indices):
    X, _, _, Locations = data
    Y = np.array([_get_output(locations)
                  for locations in Locations[testing_indices]])
    X_test = X[testing_indices]
    return X_test.reshape((*X_test.shape, 1)), Y


def _build_model():
    return build_model()


def _get_x(data):
    return data[0]


def _get_y(data):
    return _get_output(data[3])


def _get_output(locations):
    density = density_map(scale_annotation(locations, int(shape[0] * 0.25)))
    return density.reshape((*density.shape, 1))


def get_trainer(persistence_directory):
    return Trainer(persistence_directory, _image_split, _image_size, _get_validation_data, _build_model, _get_x, _get_y)
