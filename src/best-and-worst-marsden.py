import numpy as np
from pathlib import Path
from data import load_grape_data
import matplotlib.pyplot as plt
from skimage.transform import resize

interesting_img_indices = (85, 17)
predictions = (879, 315)

validation_ubuntu_indices_in_windows_order = [63, 50, 60, 90, 0, 1, 6, 7, 57, 25, 29, 11, 72,
                                              9, 71, 64, 22, 79, 18, 13, 12, 37, 16, 68, 5, 76, 61, 20, 75, 14, 28, 96, 85]
validation_windows_indices_in_ubuntu_order = [7, 9, 63, 10, 11, 31, 28, 46, 45, 83, 55, 43, 73,
                                              39, 18, 84, 20, 48, 1, 12, 2, 71, 0, 38, 59, 36, 29, 77, 67, 40, 93, 3, 91]
validation_windows_indices = np.sort(
    validation_windows_indices_in_ubuntu_order)

path = Path(
    './remote-results/results/marsden/split-2-validation-predictions.npy')
path = path.resolve()

Y, Y_validation_predictions = np.load(path)
Y = Y.reshape(Y.shape[0:3])
Y_validation_predictions = Y_validation_predictions.reshape(
    Y_validation_predictions.shape[0:3])

print(Y.shape)
print(Y_validation_predictions.shape)

Y_validation_in_winodws_order = Y[validation_ubuntu_indices_in_windows_order]
Y_validation_predictions_in_windows_order = Y_validation_predictions[np.argsort(
    validation_windows_indices_in_ubuntu_order)]

interesting_windows_index = 84
index_in_windows_validation = np.where(
    validation_windows_indices == interesting_windows_index)[0][0]
print(index_in_windows_validation)
images, _, _, _ = load_grape_data(1, 1080)

img = images[interesting_windows_index]
density = Y_validation_in_winodws_order[index_in_windows_validation]
predicted_density = Y_validation_predictions_in_windows_order[index_in_windows_validation]


def r(r, d=1): return round(r, d)


plt.axis('off')
plt.imshow(img[230:380, 300:450], cmap="gray")
plt.imshow(resize(predicted_density, (1080, 1080))[
           230:380, 300:450], alpha=0.5, cmap="inferno")
plt.show()


plt.axis('off')
ax = plt.subplot(1, 3, 1)
ax.axis('off')
plt.imshow(img, cmap="gray")
ax.set_title('Bild mit %s annotierten Beeren' % int(r(np.sum(density), 0)))
ax = plt.subplot(1, 3, 2)
ax.axis('off')
plt.imshow(density, cmap="inferno")
ax.set_title('Generierte Density Map')
ax = plt.subplot(1, 3, 3)
ax.axis('off')
plt.imshow(predicted_density, cmap="inferno")
ax.set_title('Schätzung des Modells: %s (%s%% Abweichung)' % (
    r(np.sum(predicted_density)), r(100*(1 - np.sum(predicted_density) / np.sum(density)))))

plt.show()
