import numpy as np
import json
import tensorflow as tf
import tensorflow.keras.backend as K
import traceback as tb
import types
from termcolor import colored

from umlaut.client import UmlautClient
from umlaut.heuristics import run_epoch_heuristics
from umlaut.heuristics import run_pretrain_heuristics

class UmlautCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, session_name=None, host='localhost', offline=False):

        #TODO need a case where we can't extract the frame, set to None
        self._source_module_path = tb.extract_stack()[-2].filename
        with open(self._source_module_path, 'r') as f:
            self._source_module_contents = f.read().splitlines()

        self.source_module = {
            'path': self._source_module_path,
            'contents': self._source_module_contents,
        }

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
        errors = run_pretrain_heuristics(self.model, self.source_module)
        if errors:
            print(colored('Umlaut results:', 'magenta'))
            print(colored(errors, 'red'))
            if self.umlaut_client:
                self.umlaut_client.send_errors(errors)


    def on_epoch_end(self, batch, logs=None):
        if self.umlaut_client:
            self.umlaut_client.send_logs_to_server(batch, logs)

        model_input = K.eval(self.input_node)
        errors = run_epoch_heuristics(batch, self.model, logs, model_input, self.source_module)
        if errors:
            print()
            print(colored('Umlaut results:', 'magenta'))
            print(colored(errors, 'red'))
            if self.umlaut_client:
                self.umlaut_client.send_errors(errors)


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

        model.call = types.MethodType(new_call, model)