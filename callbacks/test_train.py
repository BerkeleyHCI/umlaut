import tensorflow as tf
from test_model import TestModel
from callbacks import TestCallback

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

cb = TestCallback()
model = TestModel(cb)
model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, callbacks=[cb])
