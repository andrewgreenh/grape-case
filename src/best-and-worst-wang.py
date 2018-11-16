import numpy as np
from data import load_grape_data
import matplotlib.pyplot as plt

interesting_img_indices = (85, 17)
predictions = (879, 315)

images, density, counts, locations = load_grape_data()

for i in range(0, len(interesting_img_indices)):
    image_index = interesting_img_indices[i]
    image = images[image_index]
    true_count = counts[image_index]
    predicted_count = predictions[i]
    ax = plt.subplot(1, 2, i + 1)
    plt.axis('off')
    plt.imshow(image, cmap="gray")
    ax.set_title('Schätzung: %s\nWahrheit: %s' %
                 (predicted_count, true_count))

plt.show()
