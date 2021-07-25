import tensorflow as tf

'''classification specialized'''
conv_args = {
    'strides': (1, 1), 'dilation_rate': (1, 1), 'padding': 'same',
    'kernel_initializer': 'he_normal',
    #'kernel_regularizer': None,
    'kernel_constraint': None,
    'use_bias': True, 
    'bias_initializer': 'zeros', 
    #'bias_regularizer': None,
    'bias_constraint': None,
    } 

fc_args = {
    'kernel_initializer': 'he_normal',
    #'kernel_regularizer': None,
    'kernel_constraint': None,
    'use_bias': True,
    'bias_initializer': 'zeros', 
    #'bias_regularizer': None,
    'bias_constraint': None,
    } # classification specialized


def conv_1x1(ch, weight_decay: float):
    if(weight_decay==None):
        return tf.keras.layers.Convolution2D(ch, kernel_size=(1, 1), **conv_args)
    else:
        return tf.keras.layers.Convolution2D(ch, kernel_size=(1, 1), 
                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                    bias_regularizer=tf.keras.regularizers.l2(weight_decay),
                    **conv_args)


def conv_3x3(ch, weight_decay: float):
    if(weight_decay==None):
        return tf.keras.layers.Convolution2D(ch, kernel_size=(3, 3), **conv_args)
    else:
        return tf.keras.layers.Convolution2D(ch, kernel_size=(3, 3), 
                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                    bias_regularizer=tf.keras.regularizers.l2(weight_decay),
                    **conv_args)


def conv_5x5(ch, weight_decay: float):
    if(weight_decay==None):
        return tf.keras.layers.Convolution2D(ch, kernel_size=(5, 5), **conv_args)
    else:
        return tf.keras.layers.Convolution2D(ch, kernel_size=(5, 5), 
                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                    bias_regularizer=tf.keras.regularizers.l2(weight_decay),
                    **conv_args)

def conv_7x7(ch, weight_decay: float):
    if(weight_decay==None):
        return tf.keras.layers.Convolution2D(ch, kernel_size=(7, 7), **conv_args)
    else:
        return tf.keras.layers.Convolution2D(ch, kernel_size=(7, 7), 
                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                    bias_regularizer=tf.keras.regularizers.l2(weight_decay),
                    **conv_args)


def maxpool_2x2():
    return tf.keras.layers.MaxPooling2D(pool_size=(2, 2))


#def gap():
#    return tf.keras.layers.GlobalAveragePooling2D()


def fc(ch, weight_decay: float):
    if(weight_decay==None):
        return tf.keras.layers.Dense(ch, **fc_args)
    else:
        return tf.keras.layers.Dense(ch,  
                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                    bias_regularizer=tf.keras.regularizers.l2(weight_decay),
                    **fc_args)