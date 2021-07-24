import numpy as np
import tensorflow as tf

import sys
sys.path.append('../sf/package/')
import stealthflow as sf

# ---

print(dir(sf))
print(dir(sf.layers))
print(dir(sf.layers.mono))

print(f"tf.version:{tf.version}")

# ---

def define_model(H, W):
    x = x_in = tf.keras.layers.Input([H, W, 1])
    #x = sf.layers.conv_3x3(32)(x)
    x = tf.keras.layers.Convolution2D(32, (3, 3))(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Convolution2D(10, (3, 3))(x)
    # x = sf.layers.conv_1x1(10)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    #x = sf.layers.gap()(x)
    x = tf.keras.layers.Activation('softmax')(x)
    y = x
    return tf.keras.Model(inputs=x_in, outputs=y)


def train():
    model = define_model(H=28, W=28)
    #opt = tf.keras.optimizers.SGD(0.1)
    #loss = "mse"
    model.compile(
        optimizer='SGD',#tf.keras.optimizers.SGD(0.1),
        loss='sparse_categorical_crossentropy',
        metrics='accuracy',
    )
    print(model.summary())

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train[:, :, :, np.newaxis].astype(np.float32)/255.0
    x_test = x_test[:, :, :, np.newaxis].astype(np.float32)/255.0

    batch_size = 32
    epochs = 100

    print(x_train.shape, x_test.shape)

    history = model.fit(
        x_train, y_train,
        validation_data =  (x_test, y_test),
        epochs = epochs,
        #steps_per_epoch = x_train[0].shape[0]//batch_size,
        batch_size = batch_size,
        verbose = 1)

train()