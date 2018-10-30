import pickle
from pathlib import Path

import numpy as np
from helpers import now
from sklearn.model_selection import KFold
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import load_model

from data import load_grape_data

from .model import build_model, shape

directory = (Path(__file__).parent).resolve()


def train(persistence_directory, stop_at_ms):
    print('Loading data')
    X, Y = get_data(persistence_directory)

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

        model = get_model(persistence_directory, split)
        logger = CSVLogger(
            str(persistence_directory / ('split-%s-history.txt' % split)), separator=';', append=True)
        last_checkpoint = ModelCheckpoint(
            str(persistence_directory / ('split-%s-weights.h5' % split)))

        for epoch in range(15):
            if epoch in training_state['finished_epochs']:
                print('Training epoch %s already done, skipping' % epoch)
                continue

            start = now()
            model.fit(X[train], Y[train], batch_size=32, epochs=1,
                      validation_data=(X[test], Y[test]), callbacks=[logger, last_checkpoint])
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
    path_to_X = str(persistence_directory / 'simple_X.npy')
    path_to_Y = str(persistence_directory / 'simple_Y.npy')
    try:
        X = np.load(path_to_X)
        Y = np.load(path_to_Y)
        return X, Y
    except IOError:
        print('persisted data not found, rebuilding...')
        X, _, Y, _ = load_grape_data(1, shape[0])
        X.resize((X.shape[0], *shape))
        np.save(path_to_X, X)
        np.save(path_to_Y, Y)
        return X, Y


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
