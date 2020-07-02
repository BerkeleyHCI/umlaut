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

class TestModelAddNode(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.s1 = tf.keras.layers.Flatten()
        self.s2 = tf.keras.layers.Dense(10)

    def call(self, x):
        x = self.s1(x)
        for i in range(100):
            x = tf.keras.layers.Dense(2048)(x)
        x = tf.keras.layers.Dense(10)(x)
        print('Trying. ')
        return x