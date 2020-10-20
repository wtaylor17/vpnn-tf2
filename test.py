from tensorflow.python.keras.engine.input_layer import Input
from vpnn import layers
from tensorflow import keras
from vpnn.models import vpnn
import tensorflow as tf
# from tensorflow.python.ops.gen_math_ops import mod
# import tensorflow_datasets as tfds
# from vpnn import VpnnLayer

# (ds_train, ds_test), ds_info = tfds.load(
#     'mnist',
#     split=['train', 'test'],
#     shuffle_files=True,
#     as_supervised=True,
#     with_info=True,
# )


# def normalize_img(image, label):
#     """Normalizes images: `uint8` -> `float32`."""
#     return tf.cast(image, tf.float32) / 255., label


# ds_train = ds_train.map(
#     normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# ds_train = ds_train.cache()
# ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
# ds_train = ds_train.batch(128)
# ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)


# ds_test = ds_test.map(
#     normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# ds_test = ds_test.batch(128)
# ds_test = ds_test.cache()
# ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28) / 255
x_test = x_test.reshape(-1, 28*28) / 255
# y_train = tf.keras.utils.to_categorical(y_train)
# y_test = tf.keras.utils.to_categorical(y_test)

model = tf.keras.models.Sequential([
    # tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    # VpnnLayer(n_layers=3, n_rotations=10),
    tf.keras.layers.Input((28*28,)),
    vpnn(n_layers=3, n_rotations=10),
    tf.keras.layers.Dense(10, activation='softmax')
])

# model = vpnn(input_dim=28*28, n_layers=4, n_rotations=4)
model.summary()
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

model.fit(
    x=x_train,
    y=y_train,
    epochs=30,
    validation_data=(x_test, y_test)
)
