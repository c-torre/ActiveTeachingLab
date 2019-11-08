"""

"""

import components.patterns
import global_params
from utils import init

time_steps = global_params.time_steps
num_neurons = global_params.num_neurons
num_patterns = global_params.num_patterns

# Init
patterns = components.patterns.patterns
weights = init.random_weights
activations = init.random_activations
