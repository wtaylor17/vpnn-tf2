import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from vpnn import vpnn
from vpnn.layers import KernelWrapper


x = np.random.normal(size=(1000, 4))
y = x * x

v = vpnn(4, use_bias=False, n_rotations=4)
k = KernelWrapper(v, clip_args=(-.8, .8))

i = tf.keras.Input((4,))
m = tf.keras.Model(i, k(i))
m.compile(optimizer='sgd', loss='mse')
h = m.fit(x, y, epochs=100).history
plt.plot(h['loss'])
plt.title('loss')
plt.show()

# compare kernels
kernel_1 = v(tf.eye(4))
kernel_2 = m(tf.eye(4))
kernel_1 = tf.clip_by_value(kernel_1, -.8, .8)

print('L2 NORM BETWEEN KERNELS (should be v small): ', tf.norm(kernel_1 - kernel_2))
