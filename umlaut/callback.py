import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import types
import json

from umlaut.client import UmlautClient


class UmlautCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, session_name=None, host=None):

        self.host = host
        if not self.host.startswith('http'):
            self.host = 'http://' + self.host
        self.input_node = tf.Variable(0., validate_shape=False)
        self.output_node = tf.Variable(0., validate_shape=False)
        self.register_model(model)
        self.umlaut_client = UmlautClient(session_name, host)

    def on_train_batch_end(self, batch, logs=None):
        send_obj = {k: float(v) for k, v in logs.items()}
        # send_obj['input_example'] = K.eval(self.input_node)[0].tolist()
        # send_obj['output_example'] = K.eval(self.output_node)[0].tolist()

        self.umlaut_client.send_batch_metrics({
            'loss': {'train': [batch, logs['loss']]},
        })
        self.send_data_to_server(json.dumps(send_obj))

    def on_test_batch_end(self, batch, logs=None):
        self.umlaut_client.send_batch_metrics({
            'loss': {'val': [batch, logs['loss']]}
        })

    def register_model(self, model):
        if not isinstance(model, tf.keras.models.Model):
            raise NotImplementedError(
                'Umlaut curently doesn\'t support Non-keras models.')

        current_call = model.call
        outer_input_node = self.input_node
        outer_output_node = self.output_node

        def new_call(self, x, *args, **kwargs):
            x = tf.assign(outer_input_node, x, validate_shape=False)
            x = current_call(x, *args, **kwargs)
            assign_op = tf.assign(outer_output_node, x, validate_shape=False)
            with tf.control_dependencies([assign_op]):
                x = tf.identity(x)
            return x

        model.call = types.MethodType(new_call, model)

    def send_data_to_server(self, data):
        print("Debug: Sending Data: ", data)
        pass
