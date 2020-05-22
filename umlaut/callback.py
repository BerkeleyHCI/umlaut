import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import types
import json

from umlaut.client import UmlautClient
from umlaut.heuristics import run_epoch_heuristics

class UmlautCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, session_name=None, host=None, offline=False):

        self.tf_version = int(tf.__version__[0])  # 1 or 2

        # set up model shim
        self.model = model
        self.input_node = tf.Variable(
            0.,
            shape=tf.TensorShape(None),
            validate_shape=False,
            trainable=False,
        )
        self.output_node = tf.Variable(
            0.,
            shape=tf.TensorShape(None),
            validate_shape=False,
            trainable=False,
        )
        self.register_model(self.model)

        # set up umlaut client
        self.umlaut_client = None
        if not offline:
            self.host = host
            if not self.host.startswith('http'):
                self.host = 'http://' + self.host
            self.umlaut_client = UmlautClient(session_name, host)


    def on_train_begin(self, logs=None):
        NotImplemented


    def on_epoch_end(self, batch, logs=None):
        if self.umlaut_client:
            self.umlaut_client.send_batch_metrics({
                'loss': {
                    'train': [batch, float(logs['loss'])],
                    'val': [batch, float(logs['val_loss'])],
                },
                'acc': {  # try 'accuracy' for tf2, fall back on 'acc' for tf1 else raise
                    'train': [batch, float(logs['accuracy'] if self.tf_version == 2 else logs['acc'])],
                    'val': [batch, float(logs['val_accuracy'] if self.tf_version == 2 else logs['val_acc'])],
                },
            })
        print(logs)
        print('Running Umlaut checks...')
        model_input = K.eval(self.input_node)
        errors = run_epoch_heuristics(batch, self.model, logs, model_input)
        print(list(filter(None, errors)) or 'No errors!')
        if errors and self.umlaut_client:
            self.umlaut_client.send_errors(errors)

        # print(K.eval(self.output_node))

    def register_model(self, model):
        if not isinstance(model, tf.keras.models.Model):
            raise NotImplementedError(
                'Umlaut curently doesn\'t support Non-keras models.'
            )

        current_call = model.call

        def new_call(wrap_instance, x, *args, **kwargs):  # self here is the model, not the callback
            input_assign = self.input_node.assign(tf.cast(x, K.floatx()))  # pylint: disable=no-member
            with tf.control_dependencies([input_assign]):
                out = current_call(x, *args, **kwargs)
            output_assign = self.output_node.assign(tf.cast(out, K.floatx()))  # pylint: disable=no-member
            with tf.control_dependencies([output_assign]):
                out = tf.identity(out)
            return out

        if tf.__version__.startswith('1'):
            def v1_compat_call(wrap_instance, x, *args, **kwargs):
                input_assign = tf.assign(self.input_node, tf.cast(x, K.floatx()), validate_shape=False)  # pylint: disable=no-member
                with tf.control_dependencies([input_assign]):
                    out = current_call(x, *args, **kwargs)
                output_assign = tf.assign(self.output_node, tf.cast(out, K.floatx()), validate_shape=False)  # pylint: disable=no-member
                with tf.control_dependencies([output_assign]):
                    out = tf.identity(out)
                return out

            new_call = v1_compat_call
            
            model.call = types.MethodType(new_call, model)
            return


        model.call = types.MethodType(new_call, model)