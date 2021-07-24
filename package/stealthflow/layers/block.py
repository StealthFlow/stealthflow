import tensorflow as tf


class SEBlock(tf.keras.Model):
    def __init__(self, c, ratio=8, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(c//ratio)
        self.dense2 = tf.keras.layers.Dense(c)
        self.reshape = tf.keras.layers.Reshape([1, 1, c])

    def call(self, input_tensor, training=False):
        x = input_tensor
        x = self.gap(x)
        x = self.dense1(x)
        x = tf.nn.relu(x)
        x = self.dense2(x)
        x = tf.math.sigmoid(x)
        x = self.reshape(x)
        return x

class Residual(tf.keras.Model):
    def __init__(self, ch, se: bool, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.se = se
        self.conv1 = tf.keras.layers.Convolution2D(ch
                , kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same'
                , use_bias=True, bias_initializer='zeros', kernel_initializer='he_normal')
        self.conv2 = tf.keras.layers.Convolution2D(ch
                , kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same'
                , use_bias=True, bias_initializer='zeros', kernel_initializer='he_normal')
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.batch2 = tf.keras.layers.BatchNormalization()
        if(self.se==True):
            self.seblock = SEBlock(ch, ratio=8)

    def call(self, input_tensor, training=False):
        x = input_tensor
        x = self.batch1(x)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.batch2(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        if(self.se==True):
            x *= self.seblock(x)
        return x
    
class ConvBatchReLU(tf.keras.Model):
    def __init__(self, ch, **kwargs):
        super(ConvBatchReLU, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Convolution2D(ch
                , kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same'
                , use_bias=True, bias_initializer='zeros', kernel_initializer='he_normal')
        self.batch = tf.keras.layers.BatchNormalization()
    
    def call(self, input_tensor, training=False):
        x = input_tensor
        x = self.conv(x)
        x = self.batch(x)
        x = tf.nn.relu(x)
        return x