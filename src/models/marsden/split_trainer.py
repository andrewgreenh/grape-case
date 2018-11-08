from Trainer import Trainer
import numpy as np
from data import scale_annotation, density_map

from .model import build_model, shape, batch_size, ms_per_batch

_image_split = 16
_image_size = shape[0]


def _get_data(data):
    X, _, _, Locations = data
    Y = np.array([_get_output(locations)
                  for locations in Locations])
    return X.reshape((*X.shape, 1)), Y


def _build_model():
    return build_model()


def _get_x(data):
    return data[0]


def _get_y(data):
    return _get_output(data[3])


def _get_count_from_y(Y):
    return np.sum(Y, axis=(1, 2, 3))


def _get_output(locations):
    density = density_map(scale_annotation(locations, int(shape[0] * 0.25)))
    return density.reshape((*density.shape, 1))


def get_trainer(persistence_directory):
    return Trainer(persistence_directory, _image_split, _image_size, _get_data, _build_model, batch_size, ms_per_batch, _get_x, _get_y, _get_count_from_y)
