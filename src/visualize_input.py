import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from inputdata import full_pictures_with_density_map, split_pictures_with_density_map, Augmentor


def showcase_base_data(data):
    X, Y = data

    picture_count = 9
    pictures_per_row = int(picture_count ** 0.5)

    _, subplots = plt.subplots(
        int(picture_count / pictures_per_row), pictures_per_row)
    for i, subplot in enumerate(np.reshape(subplots, (-1))):
        frame_index = int(np.interp(i, (0, picture_count - 1),
                                    (0, len(X) - 1)))
        subplot.axis('off')
        x = X[frame_index]
        y = Y[frame_index]
        subplot.imshow(x, cmap="gray")
        subplot.imshow(y, alpha=0.5, cmap="inferno")

    plt.show()


def showcase_split_data(data):
    X, Y = data

    picture_count = 25
    pictures_per_row = int(picture_count ** 0.5)

    _, subplots = plt.subplots(
        int(picture_count / pictures_per_row), pictures_per_row)
    for i, subplot in enumerate(np.reshape(subplots, (-1))):
        subplot.axis('off')
        x = X[i + 25 * 20]
        y = Y[i + 25 * 20]
        subplot.imshow(x, cmap="gray")
        subplot.imshow(y, alpha=0.5, cmap="inferno")

    plt.show()


def showcase_augmentations(augmentor):
    X, Y = augmentor.base_data
    print(f'{augmentor.transform_count} transformations generate {augmentor.transform_count * len(X)} annotated images.')

    _, subplots = plt.subplots(
        len(augmentor.annotation_showcase), len(augmentor.annotation_showcase[0]))
    showcases = np.reshape(augmentor.annotation_showcase, (-1))
    for i, subplot in enumerate(np.reshape(subplots, (-1))):
        t = showcases[i]
        subplot.axis('off')
        x = t(X[20])
        y = t(Y[20])
        subplot.imshow(x, cmap="gray")
        subplot.imshow(y, alpha=0.5, cmap="inferno")

    plt.show()


def showcase_base_data_counts(data):
    _, Y = data

    print((f'{len(Y)} base images with '
           f'{np.sum(Y)} annotations. '
           f'(ø {int(np.sum(Y) / len(Y))})'
           ))

    counts = Y.sum(axis=(1, 2))
    min = 0
    max = (np.ceil((np.max(counts) + 1) / 50)) * 50 + 50
    plt.subplot(121)
    plt.plot(counts)
    plt.axis([None, None, min, max])
    plt.ylabel('Anzahl Beeren')
    plt.xlabel('Bildindex')

    plt.subplot(122)
    plt.hist(counts, bins=int(np.ceil((max - min) / 50)), range=(min, max),
             histtype='stepfilled', orientation='horizontal')

    plt.axis([None, None, min, max])
    plt.ylabel('Anzahl Beeren')
    plt.xlabel('Anzahl Bilder')
    plt.show()


full_pictures = full_pictures_with_density_map()
full_augmentor = Augmentor(full_pictures)

split_pictures = split_pictures_with_density_map()
split_augmentor = Augmentor(split_pictures)

showcase_base_data(full_pictures)
showcase_base_data_counts(full_pictures)
showcase_augmentations(full_augmentor)

showcase_split_data(split_pictures)
showcase_base_data_counts(split_pictures)
showcase_augmentations(split_augmentor)
