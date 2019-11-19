"""
Network initialization functions
"""

import numpy as np

import components.noise
import global_params
import utils.tools

np.random.seed(global_params.seed)

time_steps = global_params.time_steps
num_neurons = global_params.num_neurons
sparsity = global_params.sparsity


def randomize_weights(fake_num_pattern):
    """
    Makes a symmetrical, square matrix, with all 0 in te main diagonal, and
    all elements are chosen randomly to serve as control.
    E.g. possible random values for different combinations of patterns:
        1 pattern: [-1, 0, 1]
        2 patterns: [-2, -1, 0, 1, 2]
        ...

    :param fake_num_pattern: int > 0
        Number of patterns the would theoretically make the weights
    :return: array_like
        Weight matrix
    """

    assert fake_num_pattern > 0

    weights = np.zeros((num_neurons, num_neurons))
    choices = np.arange(-fake_num_pattern, fake_num_pattern + 1, 1)

    for i in range(num_neurons):
        for j in range(num_neurons):
            if j >= i:
                break
            weights[i, j] = np.random.choice(choices)
    weights += weights.T

    return weights


def initialize_weights(fake_num_patterns):
    weights = np.zeros((time_steps, num_neurons, num_neurons))
    weights[0] += randomize_weights(fake_num_patterns)
    return weights


# random_weights = randomize_weights(1)


def randomize_activations():
    """ Initial activations from noise through binary step function """
    initial = np.vstack([utils.tools.heaviside_activation(val) for val in
                         components.noise.noise[:, 0]])
    rest = np.zeros((num_neurons, time_steps - 1))
    return np.hstack((initial, rest)).astype(int)


random_activations = randomize_activations()

# def randomize_activations(idx):
#     """
#     Sets the network currents to random values according to sparsity
#     This avoids that the first currents update is driven only by noise.
#     """
#
#     currents[0] = np.random.choice([0, 1], p=[1 - , f],
#                                         size=num_neurons)
#
#     print("\nInitial currents:\n", currents[0])
#
#
# activations = randomize_activations()
