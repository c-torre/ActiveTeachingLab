"""
Neuron dynamics
"""

import matplotlib.pyplot as plt
import numpy as np

import global_params
from components import patterns, noise
from utils import init, tools

time_steps = global_params.time_steps
num_neurons = global_params.num_neurons
num_patterns = global_params.num_patterns

# Init
patterns = patterns.patterns
weights = init.random_weights
activations = init.random_activations
noise = noise.noise
plt.imshow(weights)


# Main loop
# pattern_similarity = metrics.compute_pattern_similarity()  # args!


def _update_activations_time_step(time_step, noise_values):
    """ Calculate activation for all neurons in a given time step """
    # Rnd order needed?

    if noise_values is None:
        noise_values = np.zeros_like(activations)

    dot_product = np.array(
        [np.dot(weights[:, num_neuron], activations[:, time_step - 1]) for
         num_neuron in range(num_neurons)])

    activations[:, time_step] += np.array([tools.heaviside_activation(
        dot_product[num_neuron] + noise_values[num_neuron, time_step]) for
        num_neuron in range(num_neurons)])


def compute_all_activations(noise_values):
    """ Compute all neuron activations for all time steps """

    for t_step in range(time_steps):
        if t_step == 0:
            continue
        _update_activations_time_step(t_step, noise_values)


compute_all_activations(noise)
