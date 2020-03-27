import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import types
import json

class UmlautCallback(tf.keras.callbacks.Callback):
  def __init__(self, model):
    self.input_node = tf.Variable(0., validate_shape=False)
    self.output_node = tf.Variable(0., validate_shape=False)
    self.register_model(model)

  def on_batch_end(self, batch, logs=None):
    send_obj = {k: float(v) for k, v in logs.items()}
    send_obj['input_example'] = K.eval(self.input_node)[0].tolist()
    send_obj['output_example'] = K.eval(self.output_node)[0].tolist()

    self.send_data_to_server(json.dumps(send_obj))

  def register_model(self, model):
    if not isinstance(model, tf.keras.models.Model):
      raise NotImplementedError('Umlaut curently doesn\'t support Non-keras models.')
    
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