import tensorflow as tf
import numpy as np
from typing import List, Callable, Union, Tuple

from .layers import Rotation, Permutation, Diagonal, Bias, SVDDownsize


def vpnn(input_dim: int = 2,
         n_layers: int = 1,
         n_rotations: int = 1,
         seeds: List[int] = None,
         theta_initializer: str = 'uniform',
         t_initializer: str = 'uniform',
         bias_initializer: str = 'uniform',
         output_dim: int = None,
         use_bias: bool = True,
         use_permutations: bool = True,
         use_diagonals: bool = True,
         diagonal_fn: Callable = None,
         output_activation: Union[str, Callable] = 'linear',
         hidden_activation: Union[str, Callable] = 'relu'):

    if input_dim % 2 != 0:
        raise ValueError('input dimension must be even')

    if not seeds:
        seeds = np.random.randint(0, high=10000, size=(2 * n_rotations,)).tolist()

    input_tensor = tf.keras.Input(shape=(input_dim,))
    current_output = input_tensor

    for k in range(n_layers):
        for j in range(n_rotations):
            if use_permutations:
                current_output = Permutation(seed=seeds[j])(current_output)
            current_output = Rotation(theta_initializer=theta_initializer)(current_output)

        if use_diagonals:
            current_output = Diagonal(t_initializer=t_initializer, function=diagonal_fn)(current_output)

        for j in range(n_rotations):
            if use_permutations:
                current_output = Permutation(seed=seeds[j + n_rotations])(current_output)
            current_output = Rotation(theta_initializer=theta_initializer)(current_output)

        if use_bias:
            current_output = Bias(bias_initializer=bias_initializer)(current_output)
        if k != n_layers - 1:
            current_output = tf.keras.activations.get(hidden_activation)(current_output)

    if output_dim:
        current_output = SVDDownsize(output_dim)(current_output)
    current_output = tf.keras.activations.get(output_activation)(current_output)
    return tf.keras.Model(input_tensor, current_output)
