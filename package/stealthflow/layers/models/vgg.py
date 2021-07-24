import tensorflow as tf


class VGG:
    def __init__(self, H, W, opt, loss, strategy=tf.distribute.MirroredStrategy(), plot_model=False):
        self.H, self.W = H, W

        with strategy.scope():
            self.model = self.define_network()
            self.model.compile(
                optimizer=opt,
                loss=loss,
                metrics=tf.keras.metrics.RootMeanSquaredError(),
                )
        
        if(plot_model):
            tf.keras.utils.plot_model(self.model, to_file='./model.png', show_shapes=True, show_layer_names=True)

        
    def define_network(self):
        x = x_in = tf.keras.layers.Input(shape=[self.H, self.W, 3])
        
        layers = [64, 64, "p", 128, 128, "p", 256, 256, 256]#, "p", 512, 512, 512]
        for l in layers:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x) if l=="p" else sf.layers.ConvBatchReLU(x, l)

        x = tf.keras.layers.Convolution2D(1, kernel_size=(1, 1))(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        y = x
        return tf.keras.Model(inputs=x_in, outputs=y)

