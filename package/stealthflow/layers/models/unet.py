import tensorflow as tf


class UNet(tf.keras.Model):
    def __init__(self, ch, depth, **kwargs):
        super(UNet, self).__init__(**kwargs)
        self.depth = depth - 1
        self.ch = ch
        if(depth != 0):
            self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
            self.unpool = tf.keras.layers.UpSampling2D((2, 2))
            self.concat = tf.keras.layers.Concatenate(axis=-1)
                    
        self.convs = [tf.keras.layers.Convolution2D(ch
                , kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same'
                , use_bias=True, bias_initializer='zeros', kernel_initializer='he_normal') for i in range(4)]
        self.bns = [tf.keras.layers.BatchNormalization() for i in range(4)]

        
    def call(self, input_tensor, training=False):
        x = input_tensor
        
        x = self.convs[0](x)
        x = self.bns[0](x)
        x = self.convs[1](x)
        x = self.bns[1](x)
        
        if(self.depth != 0):
            path = x
            x = self.pool(x)
            x = UNet(ch=self.ch*2, depth=self.depth)(x)
            x = self.unpool(x)
            x = self.concat([x, path])
            
        x = self.convs[2](x)
        x = self.bns[2](x)
        x = self.convs[3](x)
        x = self.bns[3](x)
        return x