import tensorflow as tf


def chebyshev(x, M=1.3, eps=1e-5):
    xs = x[..., ::2]
    ys = x[..., 1::2]
    r = tf.sqrt(tf.square(xs) + tf.square(ys)) + eps
    M_theta = M * tf.acos(tf.clip_by_value(xs / r, -1, 1))
    cos, sin = tf.cos(M_theta), tf.sin(M_theta)
    rm = r / tf.sqrt(M)
    evens = rm * cos
    odds = rm * sin * tf.sign(ys)
    return tf.reshape(tf.stack([evens, odds], axis=-1), tf.shape(x))


class Chebyshev(tf.keras.layers.Layer):
    def __init__(self, M_initializer='ones', M_init=1.3, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.trainable = trainable
        self.M_initializer, self.M = M_initializer, M_init

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        if input_shape[-1] % 2 != 0:
            raise ValueError('input dimension must be even')
        if self.trainable:
            self.M = self.add_weight(name='M',
                                     initializer=self.M_initializer,
                                     shape=(input_shape[-1]//2,))
        super().build(input_shape)

    def call(self, inputs, eps=1e-5, **kwargs):
        return chebyshev(inputs, M=self.M, eps=eps)


def get(activation, **kwargs):
    if isinstance(activation, tf.keras.layers.Layer):
        return activation
    elif activation == 'chebyshev':
        return Chebyshev(**kwargs)
    return tf.keras.activations.get(activation)
