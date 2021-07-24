import tensorflow as tf
import tensorflow_addons as tfa


class Scheduler:
    def __init__(self, lr, epochs, warm_start):
        self.lr = lr
        self.epochs = epochs
        self.warm_start = warm_start

    def cosine_decay(self):
        if(self.warm_start==True):
            raise NotImplementedError('')
        else:
            opt = tf.keras.experimental.CosineDecay(
                            initial_learning_rate=self.lr, decay_steps=self.epochs, 
                            alpha=0.0, name=None)
        return opt


    def halves(self):
        if(self.warm_start==True)

class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, initial_learning_rate):
    self.initial_learning_rate = initial_learning_rate

  def __call__(self, step):
     return self.initial_learning_rate / (step + 1)

optimizer = tf.keras.optimizers.SGD(learning_rate=MyLRSchedule(0.1))




class SGD:
    def __init__(self, lr):
        self.optimizer = tf.keras.optimizers.SGD(lr, momentum=0.9, nesterov=True)
        strategy = tf.distribute.MirroredStrategy()


class Adam:
    def __init__(self):
    def __call__(self):
        return tf.keras.optimizers.Adam(lr_decayed_fn, beta_1=0.9),


class SGDW:
    def __init__

class SGDWR:

class AdamW:

class RAdam: