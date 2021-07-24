def conv_1x1(ch):
    return tf.keras.layers.Convolution2D(ch
                , kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding='same'
                , use_bias=True, bias_initializer='zeros', kernel_initializer='he_normal')

def conv_3x3(ch):
    return tf.keras.layers.Convolution2D(ch
                , kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same'
                , use_bias=True, bias_initializer='zeros', kernel_initializer='he_normal')

def conv_5x5(ch):
    return tf.keras.layers.Convolution2D(ch
                , kernel_size=(5, 5), strides=(1, 1), dilation_rate=(1, 1), padding='same'
                , use_bias=True, bias_initializer='zeros', kernel_initializer='he_normal')

def conv_7x7(ch):
    return tf.keras.layers.Convolution2D(ch
                , kernel_size=(5, 5), strides=(1, 1), dilation_rate=(1, 1), padding='same'
                , use_bias=True, bias_initializer='zeros', kernel_initializer='he_normal')

def tail(chs):
    pass