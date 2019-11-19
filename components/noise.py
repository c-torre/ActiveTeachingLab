import numpy as np

import global_params
import utils.noise

time_steps = global_params.time_steps
num_neurons = global_params.num_neurons

noise_variance = global_params.noise_variance
noise_modulation = global_params.noise_modulation


def compute_all_noise():
    """ Get all noise for all neurons and time steps"""
    noise_values = np.zeros((num_neurons, time_steps))
    with np.nditer(noise_values, op_flags=["readwrite"]) as array:
        for i in array:
            i[...] = utils.noise.modulated_gaussian_noise(noise_variance,
                                                          noise_modulation)

    return noise_values


noise = compute_all_noise()
