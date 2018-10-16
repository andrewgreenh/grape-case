import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from data import load_grape_data
from Augmentation import Augmentor


def showcase_base_data(data):
    X, Y, _ = data

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
    X, Y, _ = data

    picture_count = 16
    pictures_per_row = int(picture_count ** 0.5)

    _, subplots = plt.subplots(
        int(picture_count / pictures_per_row), pictures_per_row)
    for i, subplot in enumerate(np.reshape(subplots, (-1))):
        subplot.axis('off')
        x = X[i + picture_count * 20]
        y = Y[i + picture_count * 20]
        subplot.imshow(x, cmap="gray")
        subplot.imshow(y, alpha=0.5, cmap="inferno")

    plt.show()


def showcase_augmentations(augmentor):
    X, Y, _ = augmentor.base_data
    print(f'{augmentor.transform_count} transformations generate {augmentor.transform_count * len(X)} annotated images.')

    _, subplots = plt.subplots(len(augmentor.annotation_showcase[0]),
                               len(augmentor.annotation_showcase))
    showcases = np.reshape(augmentor.annotation_showcase, (-1))
    for i, subplot in enumerate(np.reshape(np.transpose(subplots), (-1))):
        t = showcases[i]
        subplot.axis('off')
        x = t(X[20])
        y = t(Y[20])
        subplot.imshow(x, cmap="gray")
        subplot.imshow(y, alpha=0.5, cmap="inferno")

    plt.show()


def showcase_base_data_counts(data):
    _, _, Y_counts = data

    print((f'{len(Y_counts)} base images with '
           f'{np.sum(Y_counts)} annotations. '
           f'(ø {int(np.sum(Y_counts) / len(Y_counts))})'
           ))

    counts = np.sort(Y_counts)
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


print('Loading data...')
print('Loading full pictures...')
full_pictures = load_grape_data()
print(
    f'{full_pictures[0].shape[0]} images with {np.sum(full_pictures[2])} annotations loaded.')
print(full_pictures[2].shape)
print('Loading split pictures...')
split_pictures = load_grape_data(16)
print(
    f'{split_pictures[0].shape[0]} images with {np.sum(split_pictures[2])} annotations loaded.')
print(split_pictures[2].shape)


showcase_base_data(full_pictures)
showcase_base_data_counts(full_pictures)

augmentor = Augmentor(full_pictures)
showcase_augmentations(augmentor)

showcase_split_data(split_pictures)
showcase_base_data_counts(split_pictures)
