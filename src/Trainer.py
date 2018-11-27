import math
import numpy as np
import pandas as pd
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import KFold
import multiprocessing
from BetterModelCallback import BetterModelCallback

from Augmentation import Augmentor
from data import load_grape_data
from helpers import now

# Manages training algorithms with injected strategies for model specific tasks.


class Trainer:
    def __init__(self, persistence_directory, image_split, image_size, get_data, build_model, batch_size, ms_per_batch, get_x, get_y, get_count_from_y, custom_objects={}):
        # Attach all strategies to the instance
        self.get_data = get_data
        self.build_model = build_model
        self.batch_size = batch_size
        self.ms_per_batch = ms_per_batch
        self.get_x = get_x
        self.get_y = get_y
        self.get_count_from_y = get_count_from_y
        self.custom_objects = custom_objects

        self.persistence_directory = persistence_directory
        self.data = _get_data(persistence_directory, image_split, image_size)

        # Training state is the persisted state of the training session.
        # This can be used to resume training that was aborted due to time constraints.
        self.training_state = _get_training_state(persistence_directory)

        # Track start time to allow abort or logging of passed time.
        self.session_start = now()
        self.training_time_at_start_of_session = self.training_state['passed_training_time_in_ms']

    # Implementation of the cross validation training process.
    def start_training(self, stop_at_ms, epochs, multi_processing):
        self.print_start_summary()

        kf = KFold(n_splits=3, shuffle=True, random_state=1)
        split = 0
        for train, test in kf.split(self.data[0]):
            if split in self.training_state['finished_splits']:
                print('Training split %s already done, skipping' % split)
                split += 1
                continue

            X, Y = self.get_data(self.data)

            validation_data = X[test], Y[test]

            model = self.get_model(split)

            # Keras callback that persists counts and validation predictions when a better model is found.
            def on_better_modal(model, best_value):
                nonlocal X, Y, train, test, split
                print('\nNew best model in split %s \n' % split)
                self.persist_split_counts(model, X, Y, train, test, split)
                self.persist_split_validation_predictions(
                    model, X, Y, test, split)

            callbacks = self.get_callbacks(split)
            callbacks.append(BetterModelCallback(on_better_modal))
            augmentor = self.get_augmentor(train, epochs)

            for epoch in range(epochs):
                if epoch in self.training_state['finished_epochs']:
                    print('Training epoch %s already done, skipping' % epoch)
                    continue

                sequence = augmentor.augmentation_sequence(
                    self.batch_size, self.get_x, self.get_y)

                start = now()

                print('Training epoch %s in split %s' % (epoch, split))

                model.fit_generator(generator=sequence, epochs=1, steps_per_epoch=len(sequence),
                                    callbacks=callbacks, validation_data=validation_data,
                                    use_multiprocessing=multi_processing, workers=multiprocessing.cpu_count())
                self.training_state['finished_epochs'].add(epoch)
                self.persist_training_state()

                # Check if training should be aborted due to time constraings.
                stop = now()
                duration = stop - start
                remaining_time = stop_at_ms - stop
                how_many_epochs_remaining = int(remaining_time / duration)
                if how_many_epochs_remaining < 3:
                    print('No time left, aborting after epoch %s in split %s' %
                          (epoch, split))
                    print('Training state:', self.training_state)
                    return
                else:
                    print('%s epochs time remaining, continuing' %
                          how_many_epochs_remaining)

            # Update training state after split.
            self.training_state['finished_splits'].add(split)
            self.training_state['finished_epochs'] = set()
            self.persist_training_state()
            split += 1

        print('Done! Training state:', self.training_state)

    def print_start_summary(self):
        last_epoch = max(self.training_state['finished_epochs']) if len(
            self.training_state['finished_epochs']) > 0 else -1
        last_split = max(self.training_state['finished_splits']) if len(
            self.training_state['finished_splits']) > 0 else -1
        print('Starting at split %s - epoch %s' %
              (last_split + 1, last_epoch + 1))

    def get_model(self, split):
        try:
            return load_model(str(self.persistence_directory / ('split-%s-weights.h5' % split)), custom_objects=self.custom_objects)
        except IOError:
            return self.build_model()

    def get_callbacks(self, split):
        logger = CSVLogger(
            str(self.persistence_directory / ('split-%s-history.txt' % split)), separator=';', append=True)
        last_checkpoint = ModelCheckpoint(
            str(self.persistence_directory / ('split-%s-weights.h5' % split)))
        return [logger, last_checkpoint]

    def get_augmentor(self, training_indices, epochs):
        i = training_indices
        images, density, counts, locations = self.data
        training_data = images[i], density[i], counts[i], locations[i]

        batch_size = self.batch_size
        ms_per_batch = self.ms_per_batch

        # 6 hours as target time for each split.
        target_time_to_train_ms = 6 * 60 * 60 * 1000

        target_image_count = target_time_to_train_ms / epochs / ms_per_batch * batch_size

        return Augmentor(base_data=training_data, target_image_count=target_image_count)

    def persist_split_counts(self, model, X, Y, train, test, split):
        print('Persisting split counts')
        start = now()
        Y_pred = model.predict(X)
        end = now()
        print('Inference on %s images took %s seconds' %
              (len(X), (end - start) / 1000))
        pred_count = self.get_count_from_y(Y_pred)
        true_count = self.get_count_from_y(Y)

        table_data = np.array([pred_count, true_count]).transpose()

        data_frame = pd.DataFrame(
            table_data, columns=['pred count', 'true count'])

        data_frame['Is in Training'] = data_frame.index.isin(train)
        data_frame.to_csv(str(self.persistence_directory /
                              ('split-%s-counts.txt' % split)))
        print('Persisting split counts done.')

    def persist_split_validation_predictions(self, model, X, Y, test, split):
        print('Persisting split validation predictions')
        Y_pred = model.predict(X[test])

        data = np.array([Y, Y_pred])

        np.save(str(self.persistence_directory /
                    ('split-%s-validation-predictions.npy' % split)), data)

        print('Persisting split validation predictions done.')

    def persist_training_state(self):
        trained_in_this_session = now() - self.session_start
        self.training_state['passed_training_time_in_ms'] = self.training_time_at_start_of_session + \
            trained_in_this_session
        np.save(str(self.persistence_directory /
                    'training_state.npy'), self.training_state)


def _get_data(persistence_directory, image_split, image_size):
    images_path = str(persistence_directory / 'images.npy')
    densities_path = str(persistence_directory / 'densities.npy')
    counts_path = str(persistence_directory / 'counts.npy')
    locations_path = str(persistence_directory / 'locations.npy')
    try:
        images = np.load(images_path)
        densities = np.load(densities_path)
        counts = np.load(counts_path)
        locations = np.load(locations_path)
        return images, densities, counts, locations
    except IOError:
        print('persisted data not found, rebuilding...')
        data = load_grape_data(image_split, image_size)
        print('Data loaded, persisting for reuse')
        np.save(images_path, data[0])
        np.save(densities_path, data[1])
        np.save(counts_path, data[2])
        np.save(locations_path, data[3])
        return data


def _get_training_state(persistence_directory):
    try:
        return np.load(str(persistence_directory / 'training_state.npy')).item()
    except IOError:
        return {'finished_splits': set(), 'finished_epochs': set(), 'passed_training_time_in_ms': 0}
