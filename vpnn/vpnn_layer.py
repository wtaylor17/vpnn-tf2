import tensorflow as tf
from tensorflow import keras
from .models import vpnn
from typing import List, Callable, Union


# class VpnnLayer(keras.layers.Layer):
#     def __init__(self,
#                  n_layers: int = 1,
#                  n_rotations: int = 1,
#                  theta_initializer: str = 'uniform',
#                  t_initializer: str = 'uniform',
#                  bias_initializer: str = 'uniform',
#                  output_dim: int = None,
#                  use_bias: bool = True,
#                  use_permutations: bool = True,
#                  use_diagonals: bool = True,
#                  diagonal_fn: Callable = None,
#                  output_activation: Union[str, Callable] = 'linear',
#                  hidden_activation: Union[str, Callable] = 'relu',
#                  M_initializer='ones',
#                  M_init=1.3,
#                  trainable_M=False):
#         super(VpnnLayer, self).__init__()
#         self.n_layers = n_layers
#         self.n_rotations = n_rotations
#         self.theta_initializer = theta_initializer
#         self.t_initializer =

#     def build(self, input_shape):
#         if input_dim % 2 != 0:
#             raise ValueError('input dimension must be even')

#         input_tensor = tf.keras.Input(shape=(input_dim,))
#         current_output = input_tensor

#         for k in range(n_layers):
#             for j in range(n_rotations):
#                 if use_permutations:
#                     current_output = Permutation()(current_output)
#                 current_output = Rotation(
#                     theta_initializer=theta_initializer)(current_output)

#             if use_diagonals:
#                 current_output = Diagonal(
#                     t_initializer=t_initializer, function=diagonal_fn)(current_output)

#             for j in range(n_rotations):
#                 if use_permutations:
#                     current_output = Permutation()(current_output)
#                 current_output = Rotation(
#                     theta_initializer=theta_initializer)(current_output)

#             if use_bias:
#                 current_output = Bias(
#                     bias_initializer=bias_initializer)(current_output)
#             if k != n_layers - 1:
#                 current_output = activations.get(hidden_activation,
#                                                  trainable=trainable_M,
#                                                  M_init=M_init,
#                                                  M_initializer=M_initializer)(current_output)

#         if output_dim:
#             current_output = SVDDownsize(output_dim)(current_output)
#         current_output = activations.get(output_activation,
#                                          trainable=trainable_M,
#                                          M_init=M_init,
#                                          M_initializer=M_initializer)(current_output)
#         return tf.keras.Model(input_tensor, current_output)
#     # print(self.n_layers)
#     # print(self.n_rotations)
#     # self.vpnn = vpnn(
#     #     input_dim=input_shape[-1],
#     #     n_rotations=10,
#     #     n_layers=3
#     # )

#     def call(self, input):
#         return self.vpnn(input)


class VpnnLayer(keras.layers.Layer):
    def __init__(self,
                 n_layers: int = 1,
                 n_rotations: int = 1,
                 theta_initializer: str = 'uniform',
                 t_initializer: str = 'uniform',
                 bias_initializer: str = 'uniform',
                 output_dim: int = None,
                 use_bias: bool = True,
                 use_permutations: bool = True,
                 use_diagonals: bool = True,
                 #  diagonal_fn: Callable = None,
                 #  output_activation: Union[str, Callable] = 'linear',
                 #  hidden_activation: Union[str, Callable] = 'relu',
                 M_initializer='ones',
                 M_init=1.3,
                 trainable_M=False):
        super(VpnnLayer, self).__init__()
        # self.n_layers = n_layers
        # self.n_rotations = n_rotations
        self.vpnn = vpnn(
            input_dim=input_shape[-1],
            n_layers=3,
            n_rotations=10
        )

    def build(self, input_shape):
        super(VpnnLayer, self).build(input_shape)
        # print(self.n_layers)
        # print(self.n_rotations)

    def call(self, input):
        return self.vpnn(input)


# class VpnnLayer(keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(VpnnLayer, self).__init__()
#         self.vpnn_config = kwargs
#         self.vpnn = None

#     def build(self, input_shape):
#         print(input_shape)
#         print(self.vpnn_config)
#         # if 'input_dim' not in self.vpnn_config:
#         # self.vpnn_config['input_dim'] = input_shape[-1]
#         self.vpnn = vpnn(input_dim=input_shape[-1], **self.vpnn_config)

#     def call(self, input):
#         return self.vpnn(input)
