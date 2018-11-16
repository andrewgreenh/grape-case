import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from data import load_grape_data
from Augmentation import Augmentor


def showcase_base_data(data):
    X, Y, _, _ = data

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
    X, Y, _, _ = data

    picture_count = 4
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
    X, Y, _, _ = augmentor.base_data
    print('%s transformations generate %s annotated images.' %
          (augmentor.transform_count, augmentor.transform_count * len(X)))

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
    _, _, Y_counts, _ = data

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
print(min(full_pictures[2]), max(full_pictures[2]))
print(
    '%s images with %s annotations loaded. (ø%s)' % (len(full_pictures[0]), np.sum(full_pictures[2]), int(np.sum(full_pictures[2]) / len(full_pictures[0]))))
print(full_pictures[0].shape)
print(full_pictures[1].shape)
print(full_pictures[2].shape)


print('Loading split pictures...')
split_pictures = load_grape_data(4)

print(
    '%s images with %s annotations loaded. (ø%s)' % (len(split_pictures[0]), np.sum(split_pictures[2]), int(np.sum(split_pictures[2]) / len(split_pictures[0]))))
print(split_pictures[0].shape)
print(split_pictures[1].shape)
print(split_pictures[2].shape)


showcase_base_data(full_pictures)
showcase_base_data_counts(full_pictures)

showcase_split_data(split_pictures)
showcase_base_data_counts(split_pictures)

augmentor = Augmentor(full_pictures, 500000)
showcase_augmentations(augmentor)
