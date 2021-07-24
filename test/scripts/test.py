import numpy as np
import tensorflow as tf
import stealthflow as sf

print(f"tf.version:{tf.version}")

x = np.zeros(shape=[3000, 48, 48, 3])
y = np.zeros(shape=[3000, 1])

opt = tf.keras.optimizers.SGD(0.1)
loss = "mse"

class VGGSawedOff:
    def __init__(self, H, W, strategy=tf.distribute.MirroredStrategy(), plot_model=False):
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


model = VGGSawedOff(H=48, W=48)

model.fit()

x_train = x[:2000]
x_val = x[2000:]
y_train = y[:2000]
y_val = y[2000:]

batch_size = 32
epochs = 100


history = self.network.fit(
    (x_train, y_train),
    validation_data =  (x_val, y_val),
    epochs = epochs,
    steps_per_epoch = train[0].shape[0]//batch_size,
    batch_size = batch_size,
    verbose = 1)