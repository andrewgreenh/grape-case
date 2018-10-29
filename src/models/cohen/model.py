from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, concatenate
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

shape = (320, 320, 1)


def build_model(path_to_weights=None):
    inputs = Input(shape=shape)

    conv1 = Convolution2D(filters=64, kernel_size=3,
                          strides=1, padding='valid', activation='relu')(inputs)

    inc1 = Inception(16, 16)(conv1)

    inc2 = Inception(16, 32)(inc1)

    conv2 = Convolution2D(filters=16, kernel_size=14,
                          strides=1, padding='valid', activation='relu')(inc2)

    inc3 = Inception(112, 48)(conv2)
    inc4 = Inception(64, 32)(inc3)
    inc5 = Inception(40, 40)(inc4)
    inc6 = Inception(32, 96)(inc5)

    conv3 = Convolution2D(filters=32, kernel_size=18,
                          strides=1, padding='valid', activation='relu')(inc6)

    conv4 = Convolution2D(filters=64, kernel_size=1,
                          strides=1, padding='valid', activation='relu')(conv3)
    conv5 = Convolution2D(filters=64, kernel_size=1,
                          strides=1, padding='valid', activation='relu')(conv4)
    prediction = Convolution2D(filters=1, kernel_size=1,
                               strides=1, padding='valid', activation='relu')(conv5)

    model = Model(inputs=inputs, outputs=prediction)

    return model


def Inception(depth_of_size_1_kernels, depth_of_size_3_kernels):
    def add_previous(layer):
        tower1 = Convolution2D(filters=depth_of_size_1_kernels, kernel_size=1,
                               strides=1, padding='valid', activation='relu')(layer)

        tower2 = Convolution2D(filters=depth_of_size_3_kernels, kernel_size=1,
                               strides=1, padding='valid', activation='relu')(layer)

        module = concatenate([tower1, tower2], axis=3)
        return module

    return add_previous
