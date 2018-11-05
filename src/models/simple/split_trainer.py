from Trainer import Trainer

from .model import build_model, shape

_image_split = 16
_image_size = shape[0]


def _get_validation_data(data, testing_indices):
    X, _, Y, _ = data
    X_test = X[testing_indices]
    return X_test.reshape((*X_test.shape, 1)), Y[testing_indices]


def _build_model():
    return build_model()


def _get_x(data):
    return data[0]


def _get_y(data):
    return data[2]


def get_trainer(persistence_directory):
    return Trainer(persistence_directory, _image_split, _image_size, _get_validation_data, _build_model, _get_x, _get_y)
