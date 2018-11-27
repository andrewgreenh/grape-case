from keras.callbacks import Callback
import math


# Keras callback that performas a variable action when a better model is found during training.
class BetterModelCallback(Callback):
    def __init__(self, on_better, best_value=math.inf):
        self.best_value = best_value
        self.on_better = on_better

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get('val_loss')

        if current is None:
            return
        if current < self.best_value:
            self.best_value = current
            self.on_better(self.model, self.best_value)
