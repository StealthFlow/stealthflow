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

def train():
    x = x_in = tf.keras.Input(shape=[28, 28, 1])
    x = sf.models.VGGSawedOff(input_shape=[28, 28, 1], length=2)()(x)
    print(x.shape)
    y = sf.models.Muzzle(input_shape=x.shape[1:],
             output_dims=10, last_activation='softmax').outer_barrel_residual()(x)
    model = tf.keras.Model(inputs=x_in, outputs=y)

    model.compile(
        optimizer='SGD',#tf.keras.optimizers.SGD(0.1),
        loss='sparse_categorical_crossentropy',
        metrics='accuracy',
    )
    print(model.summary())

    tf.keras.utils.plot_model(
        model, to_file='./model.png', show_shapes=True, 
        show_layer_names=True
    )

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