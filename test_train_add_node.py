import tensorflow as tf
from test_model import TestModelAddNode

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images #/ 255.
test_images = test_images #/ 255.

model = TestModelAddNode()


class TestCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, _, logs=None):
        print(len(tf.trainable_variables()))

test_cb = TestCallback()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy'],
)

model.fit(
    train_images,
    train_labels,
    callbacks=[test_cb],
    epochs=30,
    batch_size=256
)
