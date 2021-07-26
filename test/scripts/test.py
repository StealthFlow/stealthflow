
#import stealthflow as sf
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf
import tensorflow_addons as tfa

import sys
sys.path.append('/stealthflow')
import stealthflow as sf
# ---

print(dir(sf))
print(dir(sf.layers))
print(dir(sf.layers.mono))

print(f"tf.version:{tf.version}")

# ---

from easydict import EasyDict

class NeuralDebuggerClassifier:
    def __init__(self, network, params):
        self.network = network
        self.data = EasyDict({'x_train': None, 'x_test': None, 'y_train': None, 'y_test': None})
        self.params = params

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


class DecisiveDecayWithWarmStart(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, initial_learning_rate, decay_steps=[100000, 200000], decay_rate=0.2, warm_start_steps=0):
    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps # without warmstart step
    self.decay_rate = decay_rate
    self.warm_start_steps = warm_start_steps

  def __call__(self, step):
    print(step)
    if(step <= self.warm_start_steps):
        print(step)
        return self.initial_learning_rate * (step+1) / self.warm_start_steps
    elif(step - self.warm_start_steps < decay_steps[0]):
        return self.initial_learning_rate * self.decay_rate ^ 1
    elif(step - self.warm_start_steps < decay_steps[1]):
        return self.initial_learning_rate * self.decay_rate ^ 2
    else:
        return self.initial_learning_rate * self.decay_rate ^ 3


class ExponentialDecayWithWarmStart(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, initial_learning_rate, decay_steps=100000, decay_rate=0.96, warm_start_steps=0):
    self.initial_learning_rate = initial_learning_rate
    self.warm_start_steps = warm_start_steps
    self.decay_steps = decay_steps # without warmstart step
    self.decay_rate = decay_rate

  def __call__(self, step):
    if(step <= self.warm_start_steps):
        return self.initial_learning_rate * (step+1) / self.warm_start_steps
    else:
        _step = step - self.warm_start_steps
        return self.initial_learning_rate * self.decay_rate ^ (_step/self.decay_steps)


class CosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0, warm_start_steps=0):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.warm_start_steps = warm_start_steps

    def __call__(self, step):
        step = min(step, self.decay_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return initial_learning_rate * decayed

class WarmStart(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, final_learning_rate, warm_start_steps=2000):
        self.final_learning_rate = final_learning_rate
        self.warm_start_steps = warm_start_steps

    def __call__(self, step):
        lr = tf.math.minimum(self.final_learning_rate, self.final_learning_rate * ((step+1) / self.warm_start_steps))
        print(lr)
        return lr


"""
class OptimizerHanger:
    def __init__(self, power):
        '''combination
        - learning rate : 
            'strong' / 'middle' / 'weak'
        - decay:
            DecisiveDecay
            ExponentialDecay
            CosineDecay
            CosineAnnelaing
        - warm_start
            True/False
        '''
        self.power = power
        self.warm_start = False
        self.decay_method = ['cosine', 'exponential']
        self.warm_restart = False
        self.epochs = 200
        n_data, batch_size = 60000, 128
        self.steps_per_epoch = n_data//batch_size

        '''optimizer params'''
        lr_sgd = {'strong': 0.1, 'middle': 0.01, 'weak': 0.001,}

        lr_adam = {'strong': 0.01, 'middle': 0.001, 'weak': 0.0001, 'gan': 2e-4}
        beta1_adam = {'strong': 0.9, 'middle': 0.9, 'weak': 0.9, 'gan': 0.5}

        lr_adamw = {'strong': 0.001, 'middle': 0.001, 'weak': 0.001,}
        weight_decay_adamw = {'strong': 5e-3, 'middle': 5e-4, 'weak': 5e-5,}

        '''optimizers'''
        # tf.keras.optimizers.Adam(base_lr*10, beta_1=0.9)
        # tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=5e-3)
        # tf.keras.optimizers.SGD(base_lr*10, momentum=0.9, nesterov=True)

    def define_scheduler(self):
        '''scheduler'''
        if(self.decay_method=='cosine'):
            return tf.keras.experimental.CosineDecay(initial_learning_rate=initial_lr,
                                        decay_steps=decay_steps, alpha=0.0, name=None)
        elif(self.decay_method=='exponential'):
            return tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_lr,
                                        decay_steps=decay_steps, decay_rate=0.9)
"""                        



def create_model(input_shape=[32, 32, 3], output_dims=100):
    x = x_in = tf.keras.Input(shape=input_shape)
    #x = sf.models.VGGSawedOff(input_shape=input_shape, length=10, weight_decay=0.001)()(x)
    x = sf.models.VGGDoubleBarrel(input_shape=input_shape, length=10, weight_decay=None)()(x)
    
    mz = sf.models.Muzzle(input_shape=x.shape[1:],
                         output_dims=output_dims, last_activation='softmax', weight_decay=None)
    muzzle = {
        'compensator': mz.compensator,
        'flash_hider': mz.flash_hider,
        'suppressor': mz.suppressor,
        'outer_barrel': mz.outer_barrel,
        'outer_barrel_residual': mz.outer_barrel_residual,
        }['compensator']

    y = muzzle()(x)
    model = tf.keras.Model(inputs=x_in, outputs=y)
    return model


def train():
    model = create_model()
    #lr_scheduler = DecisiveDecayWithWarmStart(initial_learning_rate=0.01, decay_steps=[10000, 20000], decay_rate=0.2, warm_start_steps=2000)
    #lr_scheduler = CosineDecay(initial_learning_rate=0.01, decay_steps=20000, alpha=0.0,)
    #opt = tf.keras.optimizers.SGD(0.001, momentum=0.9, nesterov=True)
    params = {'batch_size': 128, 'epochs': 10, }

    lr_scheduler = WarmStart(final_learning_rate=0.01, warm_start_steps=2000)
    opt = tf.keras.optimizers.SGD(lr_scheduler, momentum=0.9, nesterov=True)
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics='accuracy')
    tf.keras.utils.plot_model(model, to_file='./model.png', show_shapes=True, show_layer_names=True)

    NeuralDebuggerClassifier(model, params).debug_cifar100()

    params = {'batch_size': 128, 'epochs': 500, }
    lr_scheduler = tf.keras.experimental.CosineDecay(initial_learning_rate=0.01, decay_steps = 100000, alpha=0.0, name=None)
    opt = tf.keras.optimizers.SGD(lr_scheduler, momentum=0.9, nesterov=True)
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics='accuracy')
    NeuralDebuggerClassifier(model, params).debug_cifar100()



if(__name__=='__main__'):
    train()


"""
class Generator:
    def __init__(self):
        pass

class DA:
    def __init__(self):
        self.SEED = 42
        H, W = 192, 192
        self.augments = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomCrop(height=H, width=W, seed=self.SEED),
        ])

class CV:
    def __init__(self, n_cv):
        self.n_cv = n_cv

class TTA:
    def __init__(self):
        pass
"""