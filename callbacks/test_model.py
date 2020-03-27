import tensorflow as tf

class TestModel(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.s1 = tf.keras.layers.Flatten()
        self.s2 = tf.keras.layers.Dense(10)

    def call(self, x):
        x = self.s1(x)
        x = self.s2(x)
        return x