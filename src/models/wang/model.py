from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

shape = (227, 227, 1)


def build_model():
    inputs = Input(shape=shape)

    conv1 = Convolution2D(filters=96, kernel_size=11,
                          strides=4, padding='valid', activation='relu')(inputs)
    conv1 = MaxPooling2D(pool_size=3, strides=2)(conv1)

    conv2 = ZeroPadding2D(padding=2)(conv1)
    conv2 = Convolution2D(filters=256, kernel_size=5,
                          strides=1, padding='valid', activation='relu')(conv2)
    conv2 = MaxPooling2D(pool_size=3, strides=2)(conv2)

    conv3 = ZeroPadding2D(padding=1)(conv2)
    conv3 = Convolution2D(filters=384, kernel_size=3,
                          strides=1, padding='valid', activation='relu')(conv3)

    conv4 = ZeroPadding2D(padding=1)(conv3)
    conv4 = Convolution2D(filters=384, kernel_size=3,
                          strides=1, padding='valid', activation='relu')(conv4)

    conv5 = ZeroPadding2D(padding=1)(conv4)
    conv5 = Convolution2D(filters=256, kernel_size=3,
                          strides=1, padding='valid', activation='relu')(conv5)
    conv5 = MaxPooling2D(pool_size=3, strides=2)(conv5)

    fc1 = Flatten()(conv5)
    fc1 = Dense(units=4096, activation='relu')(fc1)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(units=4096, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)

    prediction = Dense(units=1, activation='relu')(fc2)

    model = Model(inputs=inputs, outputs=prediction)

    return model
