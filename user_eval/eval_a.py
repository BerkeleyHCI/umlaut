import tensorflow as tf
from umlaut import UmlautCallback

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3), dtype=tf.float32),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.Dense(10),
])

cb = UmlautCallback(
    model,
    session_name='ea',
    # offline=True,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy'],
)

# train the model
model.fit(
    train_images,
    train_labels,
    epochs=10,
    batch_size=128,
    callbacks=[cb],
    validation_split=0.2,
)

results = model.evaluate(test_images, test_labels, batch_size=4096)
print('test loss, test acc: ', results)