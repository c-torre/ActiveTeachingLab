import numpy as np


def binarize_item(item, num_neurons):
    """
    Item number to binary and append zeros according to network size.

    :param item: int, item index
    :param num_neurons: int, network number of neurons
    :return: array_like, dtype=int (bin)
    """
    question_array = np.array([item])
    bin_item = ((question_array[:, None]
                 & (1 << np.arange(8))) > 0).astype(int)
    bin_item = np.append(bin_item, np.zeros(num_neurons
                                            - bin_item.size))

    print("Item given as pattern:", bin_item)

    return bin_item


def distort_pattern(pattern, proportion):
    """
    Inverts array values in random positions proportionally to a parameter.

    :param pattern: array_like, binary vector to distort
    :param proportion: float, 0 to 1, 1 being full array inversion
    :return pattern: array_like, dtype=int (bin)
    """

    num_inversions = int(pattern.size * proportion)
    assert proportion != 1
    idx_reassignment = np.random.choice(pattern.size, num_inversions,
                                        replace=False)
    pattern[idx_reassignment] = np.invert(pattern[idx_reassignment] - 2)
    print("\nDistorted pattern (i.e. initial currents)...\n", pattern,
          "\n ...in positions\n", idx_reassignment)
    return pattern


def heaviside_activation(x):
    """Unit step function"""
    return int(x >= 0)


def compute_pattern_similarity(pattern_0, pattern_1):
    """
    Returns the squared proportion of bits that match in value and position
    in both given patterns with respect to the total number of neurons.

    :param pattern_0: array_like, dtype=bool
    :param pattern_1: array_like, dtype=bool
    :return: float, similarity_score
    """
    assert pattern_0.size == pattern_1.size
    match = np.sum(pattern_0 == pattern_1)
    return (match / pattern_0.size)**2


# def present_pattern(self, item):
#     kanji = item["kanji"]
#     meaning = item["meaning"]
#
#     self.patterns.append(np.concatenate((kanji, meaning), axis=None))

# flower = {"kanji": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
#           "meaning": np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1])}
#
# leg = {"kanji": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
#        "meaning": np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 1])}
#
# eye = {"kanji": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
#        "meaning": np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])}
