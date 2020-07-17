import tensorflow as tf
from umlaut import UmlautCallback

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10),
])

cb = UmlautCallback(
    model,
    session_name='pilot',
)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# train the model
model.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=256,
    callbacks=[cb],
    validation_data=(train_images[:100], train_labels[:100])
)

results = model.evaluate(test_images, test_labels, batch_size=256)
print('test loss, test acc: ', results)