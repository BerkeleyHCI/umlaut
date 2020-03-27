import tensorflow as tf
from test_model import TestModel
from umlaut_callback import UmlautCallback

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10))

# Also works with subclass of tf.keras.models.Model
# model = TestModel() 

cb = UmlautCallback(model)

model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, callbacks=[cb])
