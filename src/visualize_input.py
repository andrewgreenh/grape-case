import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from data import load_grape_data, blur_annotation, Augmentor

augmentor = Augmentor()

X, Y_one_pixel = augmentor.base_data

base_annotation_count = int(np.sum(Y_one_pixel))
picture_count = 9
pictures_per_row = int(picture_count ** 0.5)

print((f'{augmentor.base_data_count} base images with '
       f'{base_annotation_count} annotations. '
       f'(ø {int(base_annotation_count / augmentor.base_data_count)})'
       ))
print(f'{augmentor.augmented_count} augmented images with annotations.')


def showcase_base_data():
    _, subplots = plt.subplots(
        int(picture_count / pictures_per_row), pictures_per_row)
    for i, subplot in enumerate(np.reshape(subplots, (-1))):
        frame_index = int(np.interp(i, (0, picture_count - 1),
                                    (0, augmentor.base_data_count - 1)))
        subplot.axis('off')
        x = X[frame_index]
        y_blurred = blur_annotation(Y_one_pixel[frame_index])
        subplot.imshow(x)
        subplot.imshow(y_blurred, alpha=0.5, cmap="inferno")

    plt.show()


def showcase_augmentations():
    _, subplots = plt.subplots(
        len(augmentor.annotation_showcase), len(augmentor.annotation_showcase[0]))
    showcases = np.reshape(augmentor.annotation_showcase, (-1))
    for i, subplot in enumerate(np.reshape(subplots, (-1))):
        t = showcases[i]
        subplot.axis('off')
        x = t(X[10])
        y_blurred = blur_annotation(t(Y_one_pixel[10]))
        subplot.imshow(x)
        subplot.imshow(y_blurred, alpha=0.5, cmap="inferno")

    plt.show()


def showcase_base_data_counts():
    counts = np.sort(Y_one_pixel.sum(axis=(1, 2)))[::-1]
    plt.subplot(121)
    plt.plot(counts)
    plt.axis([None, None, 90, 510])
    plt.ylabel('Anzahl Beeren')
    plt.xlabel('Bildindex')

    plt.subplot(122)
    plt.hist(counts, bins=8, range=(100, 500),
             histtype='stepfilled', orientation='horizontal')

    plt.axis([None, None, 90, 510])
    plt.ylabel('Anzahl Beeren')
    plt.xlabel('Anzahl Bilder')
    plt.show()


showcase_base_data()
showcase_augmentations()
showcase_base_data_counts()
