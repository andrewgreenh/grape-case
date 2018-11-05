import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import KFold

from Augmentation import Augmentor
from data import load_grape_data
from helpers import now


class Trainer:
    def __init__(self, persistence_directory, image_split, image_size, get_validation_data, build_model, get_x, get_y):
        self.get_validation_data = get_validation_data
        self.build_model = build_model
        self.get_x = get_x
        self.get_y = get_y

        self.persistence_directory = persistence_directory
        self.data = _get_data(persistence_directory, image_split, image_size)
        self.training_state = _get_training_state(persistence_directory)

    def start_training(self, stop_at_ms):
        self.print_start_summary()

        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        split = 0
        for train, test in kf.split(self.data[0]):
            if split in self.training_state['finished_splits']:
                print('Training split %s already done, skipping' % split)
                split += 1
                continue

            validation_data = self.get_validation_data(self.data, test)

            model = self.get_model(split)
            callbacks = self.get_callbacks(split)
            augmentor = self.get_augmentor(train)

            for epoch in range(15):
                if epoch in self.training_state['finished_epochs']:
                    print('Training epoch %s already done, skipping' % epoch)
                    continue

                generator = augmentor.augmentation_generator(
                    32, self.get_x, self.get_y)

                start = now()

                print('Training epoch %s in split %s' % (epoch, split))

                model.fit_generator(generator=generator, steps_per_epoch=int(augmentor.transform_count / 32), epochs=1,
                                    callbacks=callbacks, validation_data=validation_data)
                self.training_state['finished_epochs'].add(epoch)
                self.persist_training_state()

                stop = now()
                duration = stop - start
                remaining_time = stop_at_ms - stop
                how_many_epochs_remaining = int(remaining_time / duration)
                if how_many_epochs_remaining < 3:
                    print('No time left, aborting after epoch %s in split %s' %
                          (epoch, split))
                    return
                else:
                    print('%s epochs time remaining, continuing' %
                          how_many_epochs_remaining)

            self.training_state['finished_splits'].add(split)
            self.training_state['finished_epochs'] = set()
            self.persist_training_state()
            split += 1

    def print_start_summary(self):
        last_epoch = max(self.training_state['finished_epochs']) if len(
            self.training_state['finished_epochs']) > 0 else -1
        last_split = max(self.training_state['finished_splits']) if len(
            self.training_state['finished_splits']) > 0 else -1
        print('Starting at split %s - epoch %s' %
              (last_split + 1, last_epoch + 1))

    def get_model(self, split):
        try:
            return load_model(str(self.persistence_directory / ('split-%s-weights.h5' % split)))
        except IOError:
            return self.build_model()

    def get_callbacks(self, split):
        logger = CSVLogger(
            str(self.persistence_directory / ('split-%s-history.txt' % split)), separator=';', append=True)
        last_checkpoint = ModelCheckpoint(
            str(self.persistence_directory / ('split-%s-weights.h5' % split)))
        return [logger, last_checkpoint]

    def get_augmentor(self, training_indices):
        i = training_indices
        images, density, counts, locations = self.data
        training_data = images[i], density[i], counts[i], locations[i]
        return Augmentor(training_data)

    def persist_training_state(self):
        np.save(str(self.persistence_directory /
                    'training_state.npy'), self.training_state)


def _get_data(persistence_directory, image_split, image_size):
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
        data = load_grape_data(image_split, image_size)
        np.save(images_path, data[0])
        np.save(densities_path, data[1])
        np.save(counts_path, data[2])
        np.save(locations_path, data[3])
        return data


def _get_training_state(persistence_directory):
    try:
        return np.load(str(persistence_directory / 'training_state.npy')).item()
    except IOError:
        return {'finished_splits': set(), 'finished_epochs': set()}
