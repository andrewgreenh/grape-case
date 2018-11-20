import numpy as np
from pathlib import Path
from data import load_grape_data
import matplotlib.pyplot as plt
from skimage.transform import resize


validation_ubuntu_indices_in_windows_order = [93,36,19,92,95,56,82,10,51,80,99,67,33,44,32,81,31,38,54,62,35,2,52,97,73,17,94,69,65,46,39,78,27,84]
validation_windows_indices_in_ubuntu_order = [66,24,79,14,96,54,51,37,65,8,57,92,41,90,25,68,58,22,61,89,33,85,76,94,27,53,23,98,16,5,81,17,70,30]
validation_windows_indices = np.sort(
    validation_windows_indices_in_ubuntu_order)

path = Path(
    './remote-results/results/cohen/split-0-validation-predictions.npy')
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

interesting_windows_index = 65
index_in_windows_validation = np.where(
    validation_windows_indices == interesting_windows_index)[0][0]
print(index_in_windows_validation)
images, _, _, _ = load_grape_data(1, 1080)

img = images[interesting_windows_index]
density = Y_validation_in_winodws_order[index_in_windows_validation]
predicted_density = Y_validation_predictions_in_windows_order[index_in_windows_validation]


def r(r, d=1): return round(r, d)

plt.axis('off')
plt.imshow(img, cmap="gray")
plt.imshow(resize(predicted_density, (1080, 1080)), alpha=0.5, cmap="inferno")
plt.show()

plt.axis('off')
ax = plt.subplot(1, 3, 1)
ax.axis('off')
plt.imshow(img, cmap="gray")
ax.set_title('Bild mit %s annotierten Beeren' % int(r(np.sum(density) / 1024, 0)))
ax = plt.subplot(1, 3, 2)
ax.axis('off')
plt.imshow(density, cmap="inferno")
ax.set_title('Generierte Density Map')
ax = plt.subplot(1, 3, 3)
ax.axis('off')
plt.imshow(predicted_density, cmap="inferno")
ax.set_title('Schätzung des Modells: %s (%s%% Abweichung)' % (
    r(np.sum(predicted_density) / 1024), r(100*(1 - (np.sum(predicted_density) / 1024) / (np.sum(density) / 1024)))))

plt.show()
