from Trainer import Trainer
import numpy as np

from .model import build_model, shape, custom_objects, batch_size

_image_split = 1
_image_size = shape[0]


def _get_data(data):
    X, _, Y, _ = data
    return X.reshape((*X.shape, 1)), Y.reshape((*Y.shape, 1))


def _build_model():
    return build_model()


def _get_x(data):
    return data[0]


def _get_y(data):
    return data[2]


def _get_count_from_y(Y):
    return np.sum(Y, axis=1)


def get_trainer(persistence_directory):
    return Trainer(persistence_directory, _image_split, _image_size, _get_data, _build_model, batch_size, _get_x, _get_y, _get_count_from_y, custom_objects)
