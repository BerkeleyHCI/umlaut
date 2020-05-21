import tensorflow as tf
from test_model import TestModel
from umlaut import UmlautCallback

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images
test_images = test_images

print(train_images.dtype)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10),
])

cb = UmlautCallback(
    model,
    session_name='heuristic_test_floating',
    host='localhost',
    offline=True,
)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

model.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=256,
    callbacks=[cb],
    validation_split=0.2,  # add validation for val metrics
)
