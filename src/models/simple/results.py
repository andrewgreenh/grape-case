from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from .model import custom_objects


def showcase_results(model, data):
    pass


def results(persistence_directory):
    X, _, Y_true, _ = _get_data(persistence_directory)
    model = _get_model(persistence_directory)

    order = np.argsort(Y_true)
    Y_pred = model.predict(X)

    Y_true_sort = Y_true[order]
    Y_pred_sort = Y_pred[order]

    plt.plot(Y_true_sort)
    plt.plot(Y_pred_sort)
    plt.ylabel('Anzahl Beeren')
    plt.xlabel('Bildindex')

    plt.show()


def _get_data(persistence_directory):
    images_path = str(persistence_directory / 'simple_images.npy')
    densities_path = str(persistence_directory / 'simple_densities.npy')
    counts_path = str(persistence_directory / 'simple_counts.npy')
    locations_path = str(persistence_directory / 'simple_locations.npy')

    images = np.load(images_path)
    densities = np.load(densities_path)
    counts = np.load(counts_path)
    locations = np.load(locations_path)
    return images, densities, counts, locations


def _get_model(persistence_directory):
    return load_model(str(persistence_directory / ('split-%s-weights.h5' % 0)), custom_objects=custom_objects)
