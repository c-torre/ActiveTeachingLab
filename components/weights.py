"""

"""

import numpy as np

import components.patterns
import global_params

num_neurons = global_params.num_neurons
num_patterns = components.patterns.num_patterns

patterns = components.patterns.patterns


def compute_all_target_weights():
    """
    Computes the target weights array for each of the patterns.
    """

    print("Computing target weights for all patterns...")
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
        t_weights.append(w_matrix)

    return t_weights


target_weights = compute_all_target_weights()


def combine_target_weights(arrays):
    """
    Calculates the combined target weights by adding all the target weights
    """

    # print("Adding all target weights together...")
    t_weights = []

    for i, array in enumerate(arrays):
        if i == 0:
            result = array.copy()
        else:
            result = array.copy() + arrays[i-1].copy()

        t_weights.append(result)

    # assert np.amax(t_weights) == num_patterns and np.amin(
    #     t_weights) == -num_patterns
    # TODO idx 2 not correct values, correct this first

    return t_weights


# target_weights = combine_target_weights(target_weights)


# def main():
#     compute_all_target_weights()
#     combine_target_weights()
#
#
# if __name__ == '__main__':
#     main()
#