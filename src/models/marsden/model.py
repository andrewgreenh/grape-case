from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

shape = (300, 300, 1)

batch_size = 32
ms_per_batch = 4000


def build_model():
    inputs = Input(shape=shape)

    conv1 = ZeroPadding2D(padding=4)(inputs)
    conv1 = Convolution2D(filters=36, kernel_size=9,
                          strides=1, padding='valid', activation='relu')(conv1)
    conv1 = MaxPooling2D(pool_size=2, strides=2)(conv1)

    conv2 = ZeroPadding2D(padding=3)(conv1)
    conv2 = Convolution2D(filters=72, kernel_size=7,
                          strides=1, padding='valid', activation='relu')(conv2)
    conv2 = MaxPooling2D(pool_size=2, strides=2)(conv2)

    conv3 = ZeroPadding2D(padding=3)(conv2)
    conv3 = Convolution2D(filters=36, kernel_size=7,
                          strides=1, padding='valid', activation='relu')(conv3)

    conv4 = ZeroPadding2D(padding=3)(conv3)
    conv4 = Convolution2D(filters=24, kernel_size=7,
                          strides=1, padding='valid', activation='relu')(conv4)

    conv5 = ZeroPadding2D(padding=3)(conv4)
    conv5 = Convolution2D(filters=16, kernel_size=7,
                          strides=1, padding='valid', activation='relu')(conv5)

    prediction = Convolution2D(filters=1, kernel_size=1,
                               strides=1, padding='valid', activation='relu')(conv5)

    model = Model(inputs=inputs, outputs=prediction)

    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=['mae', 'mse'])

    return model
