import tensorflow as tf

conv_args = {
    'strides': (1, 1), 'dilation_rate': (1, 1), 'padding': 'same',
    'use_bias': True, 'bias_initializer': 'zeros', 'kernel_initializer': 'he_normal'
    } # classification specialized

def conv_1x1(ch):
    return tf.keras.layers.Convolution2D(ch, kernel_size=(1, 1), **conv_args)

def conv_3x3(ch):
    return tf.keras.layers.Convolution2D(ch, kernel_size=(3, 3), **conv_args)

def conv_5x5(ch):
    return tf.keras.layers.Convolution2D(ch, kernel_size=(5, 5), **conv_args)

def conv_7x7(ch):
    return tf.keras.layers.Convolution2D(ch, kernel_size=(7, 7), **conv_args)