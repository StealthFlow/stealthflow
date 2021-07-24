import tensorflow as tf
import stealthflow as sf


class VGGSawedOff:
    def __init__(self, input_shape, length: int):
        self.input_shape = input_shape
        self.barrel = self.create_barrel(length)

    @staticmethod
    def create_barrel(length):
        if(length == 2):
            barrel = [64, "p", 128]  
        if(length == 4):
            barrel = [64, 64, "p", 128, 128]
        elif(length == 7):
            barrel = [64, 64, "p", 128, 128, "p", 256, 256, 256]
        elif(length == 10):
            barrel = [64, 64, "p", 128, 128, "p", 256, 256, 256, "p", 512, 512, 512]
        elif(length == 13):
            barrel = [64, 64, "p", 128, 128, "p", 256, 256, 256, "p", 512, 512, 512, "p", 512, 512, 512]
        return barrel

    def __call__(self):
        x = x_in = tf.keras.layers.Input(shape=self.input_shape)
        for b in self.barrel:
            x = sf.layers.maxpool_2x2()(x) if b=="p" else sf.layers.ConvBatchReLU(ch=b)(x)
        y = x
        model_name = "".join([f"-{str(_)}" for _ in self.barrel])
        return tf.keras.Model(inputs=x_in, outputs=y, name=f"VGG-Sawedoff{model_name}")


class VGGDoubleBarrel:
    def __init__(self, length, input_shape: int):
        self.input_shape = input_shape

    def create_model(self):
        x = x_in = tf.keras.layers.Input(shape=self.input_shape)
        x = sf.layers.ConvBatchReLU(ch=64)(x)
        x = sf.layers.ConvBatchReLU(ch=64)(x)
        x = sf.layers.maxpool_2x2()(x)
        x = sf.layers.ConvBatchReLU(ch=128)(x)
        x = sf.layers.ConvBatchReLU(ch=128)(x)
        x = sf.layers.maxpool_2x2()(x)
        x = sf.layers.Residual(ch=256)(x)
        x = sf.layers.Residual(ch=256)(x)
        x = sf.layers.Residual(ch=256)(x)
        x = sf.layers.maxpool_2x2()(x)
        x = sf.layers.Residual(ch=512)(x)
        x = sf.layers.Residual(ch=512)(x)
        x = sf.layers.Residual(ch=512)(x)
        return tf.keras.Model(inputs=x_in, outputs=x)