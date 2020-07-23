import tensorflow as tf

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--model_path', type=str)

args = parser.parse_args()
model = tf.saved_model.load(args.model_path)
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.reshape(-1, 28 * 28) / 255
y_test = tf.keras.utils.to_categorical(y_test)

print(f'******************metrics****************')
metrics = [tf.losses.categorical_crossentropy,
           tf.metrics.categorical_accuracy]
loaded = tf.saved_model.load(args.model_path)
infer = loaded.signatures['serving_default']
pred = infer(tf.constant(x_test, dtype=tf.float32))
h = pred[list(pred.keys())[0]]
for metric in metrics:
    val = tf.reduce_mean(metric(y_test, h))
    print(f'{metric.__name__}: {val}')
print('***************************************\n\n')
