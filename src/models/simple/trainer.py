import pickle
from pathlib import Path

import numpy as np
from helpers import now
from sklearn.model_selection import KFold
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import load_model

from data import load_grape_data
from Augmentation import Augmentor

from .model import build_model, shape

directory = (Path(__file__).parent).resolve()


def train(persistence_directory, stop_at_ms):
    print('Loading data')
    data = get_data(persistence_directory)
    X, _, Y, _ = data

    training_state = get_training_state(persistence_directory)

    last_epoch = max(training_state['finished_epochs']) if len(
        training_state['finished_epochs']) > 0 else -1
    last_split = max(training_state['finished_splits']) if len(
        training_state['finished_splits']) > 0 else -1
    print('Starting at split %s - epoch %s' %
          (last_split + 1, last_epoch + 1))

    print('Data loaded, start training')
    kf = KFold(n_splits=5)
    split = 0
    for train, test in kf.split(X):
        if split in training_state['finished_splits']:
            print('Training split %s already done, skipping' % split)
            split += 1
            continue

        X_test, Y_test = X[test].reshape((*X[test].shape, 1)), Y[test]

        model = get_model(persistence_directory, split)
        logger = CSVLogger(
            str(persistence_directory / ('split-%s-history.txt' % split)), separator=';', append=True)
        last_checkpoint = ModelCheckpoint(
            str(persistence_directory / ('split-%s-weights.h5' % split)))

        augmentor = get_augmentor(data, train)

        for epoch in range(15):
            if epoch in training_state['finished_epochs']:
                print('Training epoch %s already done, skipping' % epoch)
                continue

            generator = augmentor.augmentation_generator(
                32, lambda data: data[0], lambda data: data[2])

            start = now()
            model.fit_generator(generator=generator, steps_per_epoch=int(augmentor.transform_count / 32), epochs=1,
                                callbacks=[logger, last_checkpoint], validation_data=(X_test, Y_test))
            training_state['finished_epochs'].add(epoch)
            persist_training_state(persistence_directory, training_state)

            stop = now()
            duration = stop - start
            remaining_time = stop_at_ms - stop
            how_many_epochs_remaining = int(remaining_time / duration)
            if how_many_epochs_remaining < 5:
                print('No time left, aborting after epoch %s in split %s' %
                      (epoch, split))
                return
            else:
                print('%s epochs time remaining, continuing' %
                      how_many_epochs_remaining)

        training_state['finished_splits'].add(split)
        training_state['finished_epochs'] = set()
        persist_training_state(persistence_directory, training_state)
        split += 1

    print(training_state)


def get_data(persistence_directory):
    images_path = str(persistence_directory / 'simple_images.npy')
    densities_path = str(persistence_directory / 'simple_densities.npy')
    counts_path = str(persistence_directory / 'simple_counts.npy')
    locations_path = str(persistence_directory / 'simple_locations.npy')
    try:
        images = np.load(images_path)
        densities = np.load(densities_path)
        counts = np.load(counts_path)
        locations = np.load(locations_path)
        return images, densities, counts, locations
    except IOError:
        print('persisted data not found, rebuilding...')
        data = load_grape_data(1, shape[0])
        np.save(images_path, data[0])
        np.save(densities_path, data[1])
        np.save(counts_path, data[2])
        np.save(locations_path, data[3])
        return data


def get_training_state(persistence_directory):
    try:
        return np.load(str(persistence_directory / 'training_state.npy')).item()
    except IOError:
        return {'finished_splits': set(), 'finished_epochs': set()}


def persist_training_state(persistence_directory, training_state):
    np.save(str(persistence_directory / 'training_state.npy'), training_state)


def get_model(persistence_directory, split):
    try:
        return load_model(str(persistence_directory / ('split-%s-weights.h5' % split)))
    except IOError:
        return build_model()


def get_augmentor(data, training_indices):
    i = training_indices
    images, density, counts, locations = data
    training_data = images[i], density[i], counts[i], locations[i]
    return Augmentor(training_data)
