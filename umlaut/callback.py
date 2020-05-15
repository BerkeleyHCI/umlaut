import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import types
import json

from umlaut.client import UmlautClient


class UmlautCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, session_name=None, host=None):

        self.model = model
        self.host = host
        if not self.host.startswith('http'):
            self.host = 'http://' + self.host
        self.input_node = tf.Variable(
            0.,
            dtype=tf.float32,
            # shape=self.model.input_shape,
            shape=tf.TensorShape(None),
            validate_shape=False,
            trainable=False,
        )
        self.output_node = tf.Variable(
            0.,
            dtype=tf.float32,
            shape=tf.TensorShape(None),
            # shape=self.model.input_shape,
            validate_shape=False,
            trainable=False,
        )
        self.register_model(self.model)
        self.umlaut_client = UmlautClient(session_name, host)

    def on_epoch_end(self, batch, logs=None):
        # send_obj = {k: float(v) for k, v in logs.items()}
        # send_obj['input_example'] = K.eval(self.input_node)[0].tolist()
        # send_obj['output_example'] = K.eval(self.output_node)[0].tolist()

        # self.umlaut_client.send_batch_metrics({
        #     'loss': {
        #         'train': [batch, float(logs['loss'])],
        #         'val': [batch, float(logs['val_loss'])],
        #     },
        #     'acc': {
        #         'train': [batch, float(logs['acc'])],
        #         'val': [batch, float(logs['val_acc'])],
        #     },
        # })
        print(logs)

    def register_model(self, model):
        if not isinstance(model, tf.keras.models.Model):
            raise NotImplementedError(
                'Umlaut curently doesn\'t support Non-keras models.'
            )

        current_call = model.call
        outer_input_node = self.input_node
        outer_output_node = self.output_node

        def new_call(wrap_instance, x, *args, **kwargs):  # self here is the model, not the callback
            outer_assign = tf.compat.v1.assign(outer_input_node, tf.cast(x, tf.float32), validate_shape=False)
            out = current_call(outer_assign, *args, **kwargs)
            output_assign = tf.compat.v1.assign(outer_output_node, tf.cast(out, tf.float32), validate_shape=False)
            with tf.control_dependencies([output_assign]):
                out = tf.identity(out)
            return out

        model.call = types.MethodType(new_call, model)
