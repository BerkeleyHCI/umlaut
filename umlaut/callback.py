import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import types
import json

# from umlaut.client import UmlautClient
# from umlaut.heuristics import run_epoch_heuristics
# from umlaut.heuristics import run_pretrain_heuristics

class UmlautCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, session_name=None, host='localhost', offline=False):

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
        self.label_node = tf.Variable(
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

    def get_label_metric_fn(self):
        def label_metric(y_true, y_label):
            label_assign = self.label_node.assign(tf.cast(y_true, K.floatx()))  # pylint: disable=no-member
            with tf.control_dependencies([label_assign]):
                x = tf.constant(0)
            return x
        return label_metric

    def compute_naive_loss(self):
        loss_type = type(self.model.loss)
        random_logits = None
        if loss_type == tf.keras.losses.CategoricalCrossentropy or loss_type == tf.keras.losses.SparseCategoricalCrossentropy:
            random_class = tf.random.uniform(tf.shape(self.output_node)[:-1], minval=0, maxval=tf.shape(self.output_node)[-1], dtype=tf.int32)
            random_one_hot = tf.one_hot(random_class, depth=tf.shape(self.output_node)[-1])
            random_logits = tf.cast(random_one_hot, tf.float32) * tf.math.log(0.999) + tf.cast(1 - random_one_hot, tf.float32) * tf.math.log(tf.constant(0.001)/tf.cast(tf.shape(self.output_node)[-1], tf.float32))

        if loss_type == tf.keras.losses.BinaryCrossentropy:
            random_class = tf.random.uniform(tf.shape(self.output_node), minval=0, maxval=2, dtype=tf.int32)
            random_logits = tf.cast(random_class, tf.float32) * tf.math.log(0.999) + tf.cast(1 - random_class. tf.float32) * tf.math.log(0.001)
        
        if loss_type == tf.keras.losses.MeanAbsoluteError or loss_type == tf.keras.losses.MeanSquaredError:
            random_logits = tf.ones_like(self.output_node) * tf.reduce_mean(self.label_node, dtype=tf.float32)
        
        if random_logits is not None:
            n = K.eval(tf.reduce_sum(tf.nn.softmax(self.output_node) * tf.one_hot(tf.squeeze(tf.cast(self.label_node, tf.int32)), 10), axis=-1))
            return K.eval(self.model.loss(self.label_node, random_logits))
        else:
            return None
    def on_train_begin(self, logs=None):
        #errors = run_pretrain_heuristics(self.model)
        #print(list(filter(None, errors)) or 'No pretrain errors!')
        errors = None
        if errors and self.umlaut_client:
            self.umlaut_client.send_errors(errors)


    def on_epoch_end(self, batch, logs=None):
        if self.umlaut_client:
            self.umlaut_client.send_logs_to_server(batch, logs)

        print('Running Umlaut checks...')
        model_input = K.eval(self.input_node)
        print('Computing Naive Loss...')
        print(self.compute_naive_loss())
        labels = K.eval(self.label_node)
        print('Labels...')
        print(labels)
        #errors = run_epoch_heuristics(batch, self.model, logs, model_input)
        #print(list(filter(None, errors)) or 'No errors!')
        errors = None
        if errors and self.umlaut_client:
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

        current_compile = model.compile
        def new_compile(wrap_instance, *args, **kwargs):
            if len(args) >= 4:
                metrics = args[3]
            else:
                if 'metrics' not in kwargs:
                    kwargs['metrics'] = []
                metrics = kwargs['metrics']
            metrics.append(self.get_label_metric_fn())
            return current_compile(*args, **kwargs)

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
        
        model.compile = types.MethodType(new_compile, model)
        model.call = types.MethodType(new_call, model)
