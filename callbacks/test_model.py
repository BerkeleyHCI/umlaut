import tensorflow as tf

class TestModel(tf.keras.models.Model):
    def __init__(self, cb):
        super().__init__()
        self.s1 = tf.keras.layers.Flatten()
        self.s2 = tf.keras.layers.Dense(10)
        self.cb = cb

    def call(self, x):
        x = tf.assign(self.cb.input_node, x, validate_shape=False)
        x = self.s1(x)
        x = self.s2(x)

        assign_op = tf.assign(self.cb.output_node, x, validate_shape=False)
        with tf.control_dependencies([assign_op]):
            x = tf.identity(x)
        return x