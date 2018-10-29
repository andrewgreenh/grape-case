from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

shape = (300, 300, 1)


def build_model(path_to_weights=None):
    inputs = Input(shape=shape)

    conv1 = Convolution2D(filters=32, kernel_size=3,
                          strides=1, padding='same', activation='relu')(inputs)
    conv1 = MaxPooling2D(pool_size=2, strides=2)(conv1)

    conv2 = Convolution2D(filters=32, kernel_size=3,
                          strides=1, padding='same', activation='relu')(conv1)
    conv2 = MaxPooling2D(pool_size=2, strides=2)(conv2)

    conv3 = Convolution2D(filters=64, kernel_size=3,
                          strides=1, padding='same', activation='relu')(conv2)
    conv3 = MaxPooling2D(pool_size=2, strides=2)(conv3)

    fc1 = Flatten()(conv3)
    fc1 = Dense(units=64, activation='relu')(fc1)
    fc1 = Dropout(0.5)(fc1)

    prediction = Dense(units=1, activation='relu')(fc1)

    model = Model(inputs=inputs, outputs=prediction)

    return model
