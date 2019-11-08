import numpy as np

import global_params

time_steps = global_params.time_steps
num_neurons = global_params.num_neurons

noise_variance = global_params.noise_variance
noise_modulation = global_params.noise_modulation


def modulated_gaussian_noise(variance, multiplier):
    """
    Amplitude-modulated Gaussian noise.

    :return: int, noise_value
    """

    return np.random.normal(loc=0, scale=variance ** 0.5) * multiplier


def compute_all_noise():
    noise_values = np.zeros((num_neurons, time_steps))
    with np.nditer(noise_values, op_flags=["readwrite"]) as array:
        for i in array:
            i[...] = modulated_gaussian_noise(noise_variance, noise_modulation)

    return noise_values


noise = compute_all_noise()
