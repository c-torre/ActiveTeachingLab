"""
Network initialization functions
"""

import numpy as np

import components.noise
import global_params
import utils.tools

time_steps = global_params.time_steps
num_neurons = global_params.num_neurons
sparsity = global_params.sparsity


def randomize_weights():
    weights = np.random.choice([-1, 0, 1], size=(num_neurons, num_neurons))
    for i in range(num_neurons):
        weights[i, i] = 0

    return weights


random_weights = randomize_weights()


def randomize_activations():
    """ Initial activations from noise through binary step function """
    initial = np.vstack([utils.tools.heaviside_activation(val) for val in
                         components.noise.noise[:, 0]])
    rest = np.zeros((num_neurons, time_steps - 1))
    return np.hstack((initial, rest))


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
