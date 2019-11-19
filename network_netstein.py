"""
Control network, learns in one time step.
"""

import global_params
from components import patterns, noise, weights
from utils import init, learning

learning_rate = 1  # Keep at 1 for the control net

time_steps = global_params.time_steps
patterns = patterns.patterns
weights_ = init.initialize_weights(1)
target_weights = weights.combine_target_weights()
noise = noise.noise

weights_two = learning.learn_weights(weights_, target_weights, learning_rate)
