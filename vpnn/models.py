import tensorflow as tf
from typing import Callable, Union

from .layers import Rotation, Permutation, Diagonal, Bias, SVDDownsize
from . import activations


def vpnn(input_dim: int = None,
         n_layers: int = 1,
         n_rotations: int = 1,
         theta_initializer: str = 'uniform',
         t_initializer: str = 'uniform',
         bias_initializer: str = 'uniform',
         output_dim: int = None,
         use_bias: bool = True,
         use_permutations: bool = True,
         use_diagonals: bool = True,
         diagonal_fn: Callable = None,
         output_activation: Union[str, Callable] = 'linear',
         hidden_activation: Union[str, Callable] = 'relu',
         M_initializer='ones',
         M_init=1.3,
         trainable_M=False):
    """
    builds a VPNN model (just volume preserving kernels)
    :param M_initializer: passed to `Chebyshev` constructors
    :param M_init: passed to `Chebyshev` constructors
    :param trainable_M: passed as `trainable` to `Chebyshev` constructors
    :param input_dim: the input dimension to the model
    :param n_layers: the number of hidden layers of the model
    :param n_rotations: the number of rotations to use (k/2 if you read the paper)
    :param theta_initializer: initializer for angles of rotations
    :param t_initializer: initializer for t parameter of diagonals
    :param bias_initializer: initializer for bias vectors
    :param output_dim: if not None, the output dimension for an SVDDownsize
    :param use_bias: if False, no bias vectors are used
    :param use_permutations: if False, no permutations are used
    :param use_diagonals: if False, no diagonals are used
    :param diagonal_fn: a callable for the diagonal
    :param output_activation: activation for the output layer (the SVD if applicable)
    :param hidden_activation: activation for hidden layers (all but the last if no SVD)
    :return: a tf.keras.Model
    """
    model = tf.keras.Sequential()
    if input_dim:
        model.add(tf.keras.layers.Input((input_dim,)))

    for k in range(n_layers):
        for j in range(n_rotations):
            if use_permutations:
                model.add(Permutation())
            model.add(Rotation(theta_initializer=theta_initializer))

        if use_diagonals:
            model.add(Diagonal(t_initializer=t_initializer, function=diagonal_fn))

        for j in range(n_rotations):
            if use_permutations:
                model.add(Permutation())
            model.add(Rotation(theta_initializer=theta_initializer))

        if use_bias:
            model.add(Bias(bias_initializer=bias_initializer))
        if k != n_layers - 1:
            model.add(tf.keras.layers.Lambda(activations.get(hidden_activation,
                                                             trainable=trainable_M,
                                                             M_init=M_init,
                                                             M_initializer=M_initializer)))

    if output_dim:
        model.add(SVDDownsize(output_dim))
    model.add(tf.keras.layers.Lambda(activations.get(output_activation,
                                                     trainable=trainable_M,
                                                     M_init=M_init,
                                                     M_initializer=M_initializer)))
    return model
