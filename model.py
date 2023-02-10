import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

class Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.i3 = tf.keras.applications.InceptionV3(include_top=False)
        self.i3.trainable = False
        # self.cv1 = tf.keras.layers.Conv2D(1024,5,activation="relu", strides=1)
        self.conn = tf.keras.layers.Dense(4096, activation="leaky_relu")
        self.final = tf.keras.layers.Dense(5*5*3)

    def call(self, x):
        y = self.i3(x)
        # print(y.shape)
        # y = self.cv1(y)
        y = tf.reshape(y, (x.shape[0],-1))
        y = self.conn(y)
        y = self.final(y)
        # print(y.shape)
        y = tf.reshape(y, (x.shape[0], 5, 5, 3))
        return y


if __name__ == "__main__":
    model = Model()
    print(model(np.ones((2,224,224,3))).shape)