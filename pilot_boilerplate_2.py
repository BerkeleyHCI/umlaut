import tensorflow as tf
from umlaut import UmlautCallback

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(10),
])

cb = UmlautCallback(
    model,
    session_name='pilot_2',
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=-1e5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# train the model
model.fit(
    train_images,
    train_labels,
    epochs=15,
    batch_size=256,
    callbacks=[cb],
    validation_split=0.2,  # add validation for val metrics
)

results = model.evaluate(test_images, test_labels, batch_size=256)
print('test loss, test acc: ', results)