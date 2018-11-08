import numpy as np
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.models import Model

shape = (32, 32, 1)
batch_size = 32
ms_per_batch = 150


def build_model():
    inputs = Input(shape=shape)

    conv1 = Convolution2D(filters=16, kernel_size=3,
                          strides=1, padding='same', activation='relu')(inputs)
    conv1 = MaxPooling2D(pool_size=2, strides=2)(conv1)

    conv2 = Convolution2D(filters=16, kernel_size=3,
                          strides=1, padding='same', activation='relu')(conv1)
    conv2 = MaxPooling2D(pool_size=2, strides=2)(conv2)

    conv3 = Convolution2D(filters=32, kernel_size=3,
                          strides=1, padding='same', activation='relu')(conv2)
    conv3 = MaxPooling2D(pool_size=2, strides=2)(conv3)

    conv4 = Convolution2D(filters=8, kernel_size=3,
                          strides=1, padding='same', activation='relu')(conv3)
    conv4 = MaxPooling2D(pool_size=2, strides=2)(conv4)

    fc1 = Flatten()(conv4)
    fc1 = Dense(units=32, activation='relu')(fc1)
    fc1 = Dropout(0.5)(fc1)

    prediction = Dense(units=1, activation='relu')(fc1)

    model = Model(inputs=inputs, outputs=prediction)

    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=['mae', 'mse', relative_predictions, absolute_precentage_error])

    return model


def relative_predictions(y_true, y_pred):
    return y_pred / (y_true + 1)


def absolute_precentage_error(y_true, y_pred):
    return np.absolute(1 - y_pred / (y_true + 1))


custom_objects = {
    'relative_predictions': relative_predictions,
    'absolute_precentage_error': absolute_precentage_error
}
