"""
Learning methods
"""

import numpy as np

import global_params

time_steps = global_params.time_steps


def learn_weights(weights, target_weights, learning_rate):
    """
    Takes a weight array with the aim of getting to a target weight array.
    The advancement towards target is governed by a learning rate.
    A learning rate of 1 allows to reach target instantaneously.
    A learning rate of 0 means target is never reached.
    The relationship between leaning rate and time to converge to target is
    not linear.

    Preliminary analysis: a learning rate of 0.337 makes weights converge to
    target weights (reach 95% of target value) at 10 time steps.

    :param weights:
        Network weight matrices with shape [time_step, neuron_i, neuron_j]
    :param target_weights: array-like
        Weight matrix to reach with shape [neuron_i, neuron_j]
    :param learning_rate: float
        Between 0 and 1
    :return: array-like
        [time_step, neuron_i, neuron_j]
    """

    assert 0 <= learning_rate <= 1

    for time_step in range(1, time_steps):
        weights[time_step] += np.subtract(target_weights[-1], weights[
            time_step - 1]) * learning_rate + weights[time_step - 1]

    return weights
