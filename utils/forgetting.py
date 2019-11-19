"""
Forgetting methods
"""

import numpy as np

import global_params
import utils.noise

time_steps = global_params.time_steps
num_neurons = global_params.num_neurons


def forget_weights(weights, forgetting_rate):
    """
    Factor allows the noise to shift the values of the network according to
    the number of patterns stored.
    Otherwise, a network with 1 pattern would forget faster than another with 3

    :param weights:
    :param forgetting_rate:
    :return:
    """

    assert 0 <= forgetting_rate <= 1

    factor = np.amax(weights)

    for time_step in range(1, time_steps):
        target_weights = utils.noise.weights_noise() * factor
        weights[time_step] += np.subtract(target_weights, weights[
            time_step - 1]) * forgetting_rate + weights[time_step - 1]

    return weights
