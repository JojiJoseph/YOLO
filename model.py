import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

class Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = tf.keras.applications.InceptionV3(include_top=False)
        self.backbone.trainable = False
        self.conn = tf.keras.layers.Dense(4096, activation="leaky_relu")
        self.final = tf.keras.layers.Dense(5*5*3)

    def call(self, x):
        y = self.backbone(x)
        y = tf.reshape(y, (x.shape[0],-1))
        y = self.conn(y)
        y = self.final(y)
        y = tf.reshape(y, (x.shape[0], 5, 5, 3))
        return y


if __name__ == "__main__":
    model = Model()
    print(model(np.ones((2,224,224,3))).shape)