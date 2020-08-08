import tensorflow as tf
from umlaut import UmlautCallback

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images = train_images / 255.
test_images = test_images / 255.

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
    # tf.keras.layers.Softmax(),
])

cb = UmlautCallback(
    model,
    session_name='cifar',
    offline=True,
)

model.compile(
    optimizer='adam',
    # optimizer=tf.keras.optimizers.SGD(learning_rate=-1e5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy'],
)

# train the model
model.fit(
    train_images,
    train_labels,
    epochs=10,
    batch_size=128,
    callbacks=[cb],
    # validation_data=(train_images[:100], train_labels[:100])  # cross train/val data
    validation_split=0.2,  # add validation for val metrics
)

results = model.evaluate(test_images, test_labels, batch_size=4096)
print('test loss, test acc: ', results)