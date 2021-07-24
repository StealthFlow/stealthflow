import tensorflow as tf
import tensorflow_addons as tfa
import stealthflow as sf


class Muzzle:
    """
    use case
    - compensator: fast converge, stability (conv->GAP)
    - flash_hider: fast conerge, stability (GAP->MLP)
    - suppressor: long training, robustness
    - outer_barrel: more feature extraction
    """
    def __init__(self, input_shape, output_dims, last_activation='linear'):
        self.input_shape = input_shape
        self.output_dims = output_dims
        self.last_activation = last_activation

    def compensator(self):
        x = x_in = tf.keras.Input(shape=self.input_shape)
        x = sf.layers.conv_1x1(self.output_dims)(x)
        x = sf.layers.gap()(x)
        y = tf.keras.layers.Activation(self.last_activation)(x)
        return tf.keras.Model(inputs=x_in, outputs=y, name='compensator')

    def flash_hider(self, dropout=0.2):
        x = x_in = tf.keras.Input(shape=self.input_shape)
        x = sf.layers.gap()(x)
        x = tf.keras.layers.Dense(self.output_dims)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        y = tf.keras.layers.Activation(self.last_activation)(x)
        return tf.keras.Model(inputs=x_in, outputs=y, name='flash_hider')

    def suppressor(self, gdropout=0.3):
        x = x_in = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.GaussianDropout(gdropout)(x)
        x = tf.keras.layers.Dense(self.output_dims, kernel_regularizer=tf.keras.regularizers.L2(0.01))(x)
        y = tf.keras.layers.Activation(self.last_activation)(x)
        return tf.keras.Model(inputs=x_in, outputs=y, name='supressor')

    def outer_barrel(self, width=1024, activation=tf.nn.relu, dropout=0.5):
        x = x_in = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(width)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = activation(x)
        x = tf.keras.layers.Dense(width)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = activation(x)
        x = tf.keras.layers.Dense(self.output_dims)(x)
        y = tf.keras.layers.Activation(self.last_activation)(x)
        print(y.shape)
        return tf.keras.Model(inputs=x_in, outputs=y, name='outer_barrel')

    def outer_barrel_residual(self, width=1024, activation=tfa.activations.gelu, norm='ln'):
        x = x_in = tf.keras.Input(shape=self.input_shape)
        x = x_res = tf.keras.layers.Flatten()(x)

        for _ in range(2):
            x = tf.keras.layers.Dense(width)(x)
            if(norm=='bn'): x = tf.keras.layers.BatchNormalization()(x)
            if(norm=='ln'): x = tf.keras.layers.LayerNormalization()(x)
            if(norm=='gn'): x = tfa.layers.GroupNormalization(groups=8, axis=-1)(x)
            if(norm=='in'): x = tfa.layers.InstanceNormalization(axis=-1)(x)
            x = activation(x)

        x = x_res = tf.keras.layers.Add()([x, tf.keras.layers.Dense(width)(x_res)])
        
        for _ in range(2):
            x = tf.keras.layers.Dense(width)(x)
            if(norm=='bn'): x = tf.keras.layers.BatchNormalization()(x)
            if(norm=='ln'): x = tf.keras.layers.LayerNormalization()(x)
            if(norm=='gn'): x = tfa.layers.GroupNormalization(groups=8, axis=-1)(x)
            if(norm=='in'): x = tfa.layers.InstanceNormalization(axis=-1)(x)
            x = activation(x)

        x = tf.keras.layers.Add()([x, x_res])
        x = tf.keras.layers.Dense(self.output_dims)(x)
        y = tf.keras.layers.Activation(self.last_activation)(x)
        return tf.keras.Model(inputs=x_in, outputs=y, name='outer_barrel_residual')