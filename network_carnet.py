"""
Control network, does not learn.

Honoring someone who once thought his calculator was broken as "1+1" displayed
"2" instead of "11"
"""

import global_params
from components import patterns, noise, weights
from utils import init, learning

learning_rate = 0

time_steps = global_params.time_steps
patterns = patterns.patterns
weights_ = init.initialize_weights(1)
target_weights = weights.combine_target_weights()
noise = noise.noise

weights_two = learning.learn_weights(weights_, target_weights, learning_rate)
