"""
Model metrics tools
"""

import numpy as np


def compute_pattern_similarity(pattern_0, pattern_1):
    """
    Returns the proportion of bits that match in value and position
    in both given patterns with respect to the total number of neurons.

    :param pattern_0: array_like, dtype=bool
    :param pattern_1: array_like, dtype=bool
    :return: float
    """

    assert pattern_0.size == pattern_1.size
    match = np.sum(pattern_0 == pattern_1)
    return match / pattern_0.size
