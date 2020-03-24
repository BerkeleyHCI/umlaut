import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

class TestCallback(tf.keras.callbacks.Callback):
  def __init__(self):
    self.input_node = tf.Variable(0., validate_shape=False)
    self.output_node = tf.Variable(0., validate_shape=False)
    K.get_session().run(tf.global_variables_initializer())

  def on_batch_begin(self, batch, logs=None):
    #print('here?')
    #n = self.model.input_node
    print(np.mean(K.eval(self.input_node)))
    print(np.mean(K.eval(self.output_node)))
    #print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))
