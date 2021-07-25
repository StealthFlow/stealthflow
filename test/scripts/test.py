import numpy as np
import silence_tensorflow.auto
import tensorflow as tf
import tensorflow_addons as tfa

import sys
sys.path.append('../sf/package/')
import stealthflow as sf

# ---

print(dir(sf))
print(dir(sf.layers))
print(dir(sf.layers.mono))

print(f"tf.version:{tf.version}")

# ---

class NeuralDebugger:
    def __init__(self, network):
        self.network = network

    def debug_mnist(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train[:, :, :, np.newaxis].astype(np.float32)/255.0
        x_test = x_test[:, :, :, np.newaxis].astype(np.float32)/255.0
        print(x_train.shape, x_test.shape)

        mnist_params = {
            'batch_size': 128,
            'epochs': 100,
        }

        history = self.network.fit(
            x_train, y_train,
            validation_data =  (x_test, y_test),
            verbose = 1
            **mnist_params)


def train():
    x = x_in = tf.keras.Input(shape=[28, 28, 1])
    x = sf.models.VGGSawedOff(input_shape=[28, 28, 1], length=4, weight_decay=None)()(x)
    y = sf.models.Muzzle(input_shape=x.shape[1:],
            output_dims=10, last_activation='softmax', weight_decay=None).suppressor()(x)
    model = tf.keras.Model(inputs=x_in, outputs=y)

    opt = tfa.optimizers.AdamW(learning_rate=0.0001, weight_decay=5e-4)

    model.compile(
        optimizer=opt,#,'SGD',#tf.keras.optimizers.SGD(0.1),
        loss='sparse_categorical_crossentropy',
        metrics='accuracy',
    )
    print(model.summary())

    tf.keras.utils.plot_model(
        model, to_file='./model.png', show_shapes=True, 
        show_layer_names=True
    )

train()