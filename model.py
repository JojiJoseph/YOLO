import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

class Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = tf.keras.applications.InceptionV3(include_top=False)
        self.backbone.trainable = True
        for layer in self.backbone.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                print(layer.name)
                layer.trainable = False

        self.conn = tf.keras.layers.Dense(4096, activation="leaky_relu")
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.final = tf.keras.layers.Dense(5*5*25)

    def call(self, x):
        y = self.backbone(x)
        y = tf.reshape(y, (x.shape[0],-1))
        y = self.conn(y)
        y = self.dropout(y)
        y = self.final(y)
        y = tf.reshape(y, (x.shape[0], 5, 5, 25))
        return y


if __name__ == "__main__":
    model = Model()
    print(model(np.ones((2,224,224,3))).shape)