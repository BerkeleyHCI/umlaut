import tensorflow as tf
from test_model import TestModel
from umlaut import UmlautCallback

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

model = tf.keras.Sequential()
# model.add(tf.keras.layers.Input())
# model.add(tf.keras.layers.Reshape((28, 28, 1)))
# model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10))

# Also works with subclass of tf.keras.models.Model
# model = TestModel() 

cb = UmlautCallback(
    model,
    session_name='test_update_metrics_epoch',
    host='localhost',
)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy'],
)

model.fit(
    train_images / 256.,
    train_labels / 256.,
    epochs=10,
    callbacks=[cb],
    validation_split=0.2,  # add validation for val metrics
)
