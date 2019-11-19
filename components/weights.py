"""
Weights calculations
"""

import numpy as np

import components.patterns
import global_params

num_neurons = global_params.num_neurons
num_patterns = components.patterns.num_patterns
time_steps = global_params.time_steps
patterns = components.patterns.patterns







def compute_all_target_weights():
    """ Computes the target weights array for each of the patterns """

    # print("Computing target weights for all patterns...")
    t_weights = []

    for pattern in patterns:
        w_matrix = np.zeros((num_neurons, num_neurons))
        for i in range(num_neurons):
            for j in range(num_neurons):
                if j >= i:
                    break
                w_matrix[i, j] += \
                    (2 * pattern[i] - 1) \
                    * (2 * pattern[j] - 1)

        w_matrix += w_matrix.T
        t_weights.append(np.array(w_matrix))

    return t_weights


# target_weights = compute_all_target_weights()


def combine_target_weights():
    """
    Calculates the combined target weights by adding all the target weights
    """
    arrays = compute_all_target_weights()

    t_weights = []

    for i, array in enumerate(arrays):
        if i == 0:
            result = np.copy(array)
        else:
            result = np.add(np.copy(array), t_weights[-1])

        t_weights.append(result)

    return t_weights


# target_weights = combine_target_weights(target_weights)


def main():
    """ Main """
    weights = compute_all_target_weights()
    combine_target_weights(weights)


if __name__ == '__main__':
    main()
