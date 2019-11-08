"""
Binary patterns representing memories
"""

import numpy as np

import global_params

sparsity = global_params.sparsity
num_patterns = global_params.num_patterns
num_neurons = global_params.num_neurons

np.random.seed(global_params.seed)


def compute_patterns():
    return np.random.choice([0, 1], p=[1 - sparsity, sparsity],
                            size=(num_patterns, num_neurons))


patterns = compute_patterns()

# if __name__ == '__main__':
#     compute_patterns()
