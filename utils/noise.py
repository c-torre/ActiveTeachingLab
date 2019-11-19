import numpy as np

import components.patterns
import global_params
import utils.noise

np.random.seed(global_params.seed)

num_neurons = global_params.num_neurons
num_patterns = components.patterns.num_patterns
noise_variance = global_params.noise_variance
noise_modulation = global_params.noise_modulation


def modulated_gaussian_noise(variance, multiplier):
    """
    Amplitude-modulated Gaussian noise.

    :return: int
    """

    return np.random.normal(loc=0, scale=variance ** 0.5) * multiplier


def weights_noise():
    """
    Note: it seems correct to assume that the main diagonal is still 0.
    :return:
    """
    weights = np.zeros((num_neurons, num_neurons))

    for i in range(num_neurons):
        for j in range(num_neurons):
            if j >= i:
                break
            weights[i, j] = modulated_gaussian_noise(
                noise_variance, noise_modulation) * num_patterns
    weights += weights.T

    return weights
