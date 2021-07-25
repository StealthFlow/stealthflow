
#import stealthflow as sf
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf
import tensorflow_addons as tfa

import sys
sys.path.append('/work/stealthflow')
import stealthflow as sf
# ---

print(dir(sf))
print(dir(sf.layers))
print(dir(sf.layers.mono))

print(f"tf.version:{tf.version}")

# ---

from easydict import EasyDict

class NeuralDebugger:
    def __init__(self, network):
        self.network = network
        self.data = EasyDict({'x_train': None, 'x_test': None, 'y_train': None, 'y_test': None})
        self.params = {'batch_size': 128, 'epochs': 100, }

    def show_shapes(self):
        print(self.data.x_train.shape, self.data.x_test.shape)
        print(self.data.x_train.max(), self.data.x_test.min())


    def train(self):
        self.show_shapes()
        history = self.network.fit(
            self.data.x_train, self.data.y_train,
            validation_data=(self.data.x_test, self.data.y_test),
            verbose=1,
            **self.params)

    def debug_mnist(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.data.x_train = x_train[:, :, :, np.newaxis].astype(np.float32)/255.0
        self.data.x_test = x_test[:, :, :, np.newaxis].astype(np.float32)/255.0
        self.data.y_train = y_train
        self.data.y_test = y_test
        self.train()

    def debug_cifar10(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        print(x_train.shape, y_train.shape)
        self.data.x_train = x_train.astype(np.float32)/255.0
        self.data.x_test = x_test.astype(np.float32)/255.0
        self.data.y_train = y_train
        self.data.y_test = y_test
        self.train()

    def debug_cifar100(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        self.data.x_train = x_train.astype(np.float32)/255.0
        self.data.x_test = x_test.astype(np.float32)/255.0
        self.data.y_train = y_train
        self.data.y_test = y_test
        self.train()


def create_model(input_shape=[32, 32, 3], output_dims=100):
    x = x_in = tf.keras.Input(shape=input_shape)
    x = sf.models.VGGSawedOff(input_shape=input_shape,
                              length=4, weight_decay=0.001)()(x)
    mz = sf.models.Muzzle(input_shape=x.shape[1:],
                         output_dims=output_dims, last_activation='softmax', weight_decay=0.001)
    muzzle = [mz.compensator, mz.flash_hider, mz.suppressor, mz.outer_barrel, mz.outer_barrel_residual][4]
    y = muzzle()(x)
    model = tf.keras.Model(inputs=x_in, outputs=y)
    return model


def train():
    model = create_model()
    opt = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=5e-4)
    model.compile(
        optimizer=opt,  # ,'SGD',#tf.keras.optimizers.SGD(0.1),
        loss='sparse_categorical_crossentropy',
        metrics='accuracy',
    )

    tf.keras.utils.plot_model(
        model, to_file='./model.png', show_shapes=True,
        show_layer_names=True
    )

    NeuralDebugger(model).debug_cifar100()


train()
