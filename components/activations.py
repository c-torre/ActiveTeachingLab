"""
Neuron activations
"""

import numpy as np

import global_params
from components import noise
from utils import init, tools

time_steps = global_params.time_steps
num_neurons = global_params.num_neurons

activations = init.randomize_activations()
noise = noise.noise


def _update_activations_time_step(weights, time_step, noise_values):
    """ Calculate activation for all neurons in a given time step """
    # Rnd order needed?

    if noise_values is None:
        noise_values = np.zeros_like(activations)

    dot_product = np.array([np.dot(weights[time_step, :, num_neuron],
                                   activations[:, time_step - 1]) for
                            num_neuron in range(num_neurons)])

    activations[:, time_step] += np.array([tools.heaviside_activation(
        dot_product[num_neuron] + noise_values[num_neuron, time_step]) for
        num_neuron in range(num_neurons)])


def compute_all_activations(weights):
    """ Compute all neuron activations for all time steps """

    for t_step in range(time_steps):
        if t_step == 0:
            continue
        _update_activations_time_step(weights, t_step, noise)

    return activations

# compute_all_activations(noise)
