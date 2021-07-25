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
    def __init__(self, input_shape, output_dims, last_activation='linear', weight_decay=None):
        self.input_shape = input_shape
        self.output_dims = output_dims
        self.last_activation = last_activation
        self.weight_decay = weight_decay # weight_decay


    @staticmethod
    def normalization(x, norm):
        if(norm=='bn'): # Batch Normalization
            return tf.keras.layers.BatchNormalization()(x)
        elif(norm=='ln'): # Layer Normalization
            return tf.keras.layers.LayerNormalization()(x)
        elif(norm=='gn'): # Group Normalization
            return tfa.layers.GroupNormalization(groups=8, axis=-1)(x)
        elif(norm=='in'): # Instance Normalization
            return tfa.layers.InstanceNormalization(axis=-1)(x)
        elif(norm==None):
            return x
        else:
            raise NotImplementedError('unknown normalization method')


    def compensator(self):
        x = x_in = tf.keras.Input(shape=self.input_shape)
        x = sf.layers.conv_1x1(self.output_dims, self.weight_decay)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        y = tf.keras.layers.Activation(self.last_activation)(x)
        return tf.keras.Model(inputs=x_in, outputs=y, name='compensator')

    def flash_hider(self, dropout=0.2):
        x = x_in = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = sf.layers.fc(self.output_dims, self.weight_decay)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        y = tf.keras.layers.Activation(self.last_activation)(x)
        return tf.keras.Model(inputs=x_in, outputs=y, name='flash_hider')

    def suppressor(self, gdropout=0.3):
        x = x_in = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.GaussianDropout(gdropout)(x)
        x = sf.layers.fc(self.output_dims, self.weight_decay)(x)
        y = tf.keras.layers.Activation(self.last_activation)(x)
        return tf.keras.Model(inputs=x_in, outputs=y, name='supressor')

    def outer_barrel(self, width=1024, activation=tf.nn.relu, dropout=0.5, norm='bn'):
        x = x_in = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Flatten()(x)
        x = sf.layers.fc(width, self.weight_decay)(x)
        x = self.normalization(x, norm)
        x = tf.keras.layers.Dropout(dropout, self.weight_decay)(x)
        x = activation(x)
        x = sf.layers.fc(width, self.weight_decay)(x)
        x = self.normalization(x, norm)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = activation(x)
        x = sf.layers.fc(self.output_dims, self.weight_decay)(x)
        y = tf.keras.layers.Activation(self.last_activation)(x)
        return tf.keras.Model(inputs=x_in, outputs=y, name='outer_barrel')

    def outer_barrel_residual(self, width=1024, activation=tfa.activations.gelu, norm='bn'):
        x = x_in = tf.keras.Input(shape=self.input_shape)
        x = x_res = tf.keras.layers.Flatten()(x)

        for _ in range(2):
            x = sf.layers.fc(width, self.weight_decay)(x)
            x = self.normalization(x, norm)
            x = activation(x)

        x = x_res = tf.keras.layers.Add()([x, sf.layers.fc(width, self.weight_decay)(x_res)])
        
        for _ in range(2):
            x = sf.layers.fc(width, self.weight_decay)(x)
            x = self.normalization(x, norm)
            x = activation(x)

        x = tf.keras.layers.Add()([x, x_res])
        x = sf.layers.fc(self.output_dims, self.weight_decay)(x)

        y = tf.keras.layers.Activation(self.last_activation)(x)
        return tf.keras.Model(inputs=x_in, outputs=y, name='outer_barrel_residual')