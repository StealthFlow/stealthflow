import tensorflow as tf
import stealthflow as sf


class ConvBatchReLU(tf.keras.Model):
    def __init__(self, ch, weight_decay, se=0, **kwargs):
        self.se = se
        super(ConvBatchReLU, self).__init__(**kwargs)
        self.conv = sf.layers.conv_3x3(ch, weight_decay=weight_decay)
        self.batch = tf.keras.layers.BatchNormalization()
        if(self.se!=0):
            self.seblock = SEBlock(ch, ratio=se, weight_decay=weight_decay)
            
    def call(self, input_tensor, training=False):
        x = input_tensor
        x = self.conv(x)
        x = self.batch(x)
        x = tf.nn.relu(x)
        if(self.se==True):
            x *= self.seblock(x)
        return x


class Residual(tf.keras.Model):
    """(↑)-b-r-c-b-r-c-(+)"""
    def __init__(self, ch, weight_decay=None, se=0, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.se = se
        self.conv1 = sf.layers.conv_3x3(ch, weight_decay=weight_decay)
        self.conv2 = sf.layers.conv_3x3(ch, weight_decay=weight_decay)
        self.conv_1x1 = sf.layers.conv_1x1(ch, weight_decay=weight_decay)
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.batch2 = tf.keras.layers.BatchNormalization()
        if(self.se!=0):
            self.seblock = SEBlock(ch, ratio=se, weight_decay=weight_decay)

    def call(self, input_tensor, training=False):
        x = input_tensor
        x = self.batch1(x)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.batch2(x)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Add()([self.conv2(x), self.conv_1x1(input_tensor, self.weight_decay)])
        if(self.se==True):
            x *= self.seblock(x)
        return x
    

class SEBlock(tf.keras.Model):
    def __init__(self, c, ratio=8, weight_decay=None, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = sf.layers.fc(c//ratio, weight_decay)
        self.dense2 = sf.layers.fc(c, weight_decay)
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

