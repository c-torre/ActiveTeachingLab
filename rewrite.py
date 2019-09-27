"""
"""

import os
import pickle

import numpy as np

import tools.functions as tools
import tools.plot as plot


class Hopfield:
    """
    :param num_iterations: int
        Total number of iterations the network will be updated for.
    :param num_neurons: int, optional
        Number of neurons in the network. As in Hopfield networks, they are
        fully connected, and at the same time serve as input and output.
        If not given or 0, it will be set within the memory capacity limits for
        the number of memories.
    :param learning_rate: float, proportion of the theoretical weights learn
        per time step
    :param forgetting_rate: float, proportion of the theoretical weights
        forgotten per time step
    """

    version = 3.0
    bounds = ('learning_rate', 10**-7, 0.99), \
             ('forgetting_rate', 10**-7, 0.99),

    def __init__(self, num_iterations, num_neurons=0, p=16, f=0.1,
                 noise_variance=65, noise_modulation=0.05, first_p=0,
                 learning_rate=0.3, forgetting_rate=0.1, **kwargs):

        # super().__init__(**kwargs)

        # Basic parameters
        self.num_iterations = num_iterations
        self.n_iteration = 0

        self.num_neurons = num_neurons
        self.p = p

        if self.num_neurons == 0:
            self._auto_num_neurons()
            print(f"Set number of neurons automatically to {self.num_neurons}")
        elif self.num_neurons is int:
            pass
        else:
            raise ValueError("num_neurons must be int or 'auto'")

        self.p = p
        self.f = f
        self.first_p = first_p

        # Noise
        self.noise_variance = noise_variance
        self.noise_modulation = noise_modulation
        self.noise = np.zeros((self.num_neurons, self.num_iterations))

        # Memory parameters
        self.learning_rate = learning_rate
        self.forgetting_rate = forgetting_rate
        assert (self.learning_rate and self.forgetting_rate) <= 1

        # Patterns
        self.patterns = np.random.choice([0, 1], p=[1 - self.f, self.f],
                                         size=(self.p, self.num_neurons))
        self.pattern_similarity = np.zeros((self.p, self.num_iterations))
        print("\nPatterns:\n", self.patterns)

        # Weights
        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        self.next_weights = np.zeros_like(self.weights)
        self.weights_history = np.zeros((self.num_iterations, self.num_neurons,
                                         self.num_neurons))
        self.weights_mean = np.zeros(self.num_iterations)

        # Target weights
        self.all_target_weights = np.zeros((self.p, self.num_neurons,
                                            self.num_neurons), dtype=int)
        self.combined_target_weights = np.zeros_like(self.weights)

        # Currents
        self.currents = np.zeros((self.num_iterations, self.num_neurons),
                                 dtype=int)

    def _auto_num_neurons(self):
        """
        Updates number of neurons within memory capacity as of:

        Wulfram Gerstner, Werner M. Kistler, Richard Naud and Liam Paninski
        "Neuronal dynamics", ch 17.2.4, eq 17.22
        https://neuronaldynamics.epfl.ch/
        """

        neurons_per_p = 9  # Approximated as np.ceil(8.33)
        self.num_neurons = int(neurons_per_p * self.p)

    def _randomize_weights(self):
        """
        Randomizes weights as int -1, 0 or 1.
        When i == j, element is left at 0.
        """

        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if j == i:
                    continue
                self.weights[i, j] = np.random.choice([-1, 0, 1])

    def _compute_all_target_weights(self):
        """
        Computes the weights array for each of the patterns, updating the
        (index, i, j) target weights array.
        """

        for p in range(self.p):
            for i in range(self.num_neurons):
                for j in range(self.num_neurons):
                    if j >= i:
                        break
                    self.all_target_weights[p, i, j] +=\
                        (2 * self.patterns[p, i] - 1) \
                        * (2 * self.patterns[p, j] - 1)

            self.all_target_weights[p] += self.all_target_weights[p].T

    def _combine_target_weights(self):
        """
        Calculates the combined target weights by adding all the target weights
        """

        self.combined_target_weights = np.sum(self.all_target_weights, 0)

        assert np.amax(self.combined_target_weights) == self.p and \
            np.amin(self.combined_target_weights) == -self.p

    def _compute_noise(self):
        """
        Computes the Gaussian noise value for every neuron and iteration.
        """
        if self.n_iteration > 50 and self.num_neurons > 100:
            print("Computing modulated Gaussian noise...")

        for i in np.nditer(self.noise, op_flags=["readwrite"]):
            i += tools.modulated_gaussian_noise(
                    self.noise_variance, self.noise_modulation)

    def _update_pattern_similarity(self):
        """
        Compute the last pattern similarity and append it to the network
        history. NONOOONONONOONOONONONONONONONONONONONOONONONONONONONONO

        The problem pattern is either computed from binarizing the given int
        or taken from the stored p patterns of the network.
        """

        for p in range(self.p):
            self.pattern_similarity[p, self.n_iteration] = \
                tools.compute_pattern_similarity(
                    self.currents[self.n_iteration], self.patterns[p])

    def _update_current(self, neuron):
        """
        If you are updating one node of a Hopfield hopfield_network, then the
        values of
        all the other nodes are input values, and the weights from those nodes
        to the updated node as the weights.
        In other words, first you do a weighted sum of the inputs from the
        other nodes, then if that value is greater than or equal to 0, you
        output 1. Otherwise, you output 0
        :param neuron: int neuron number
        """
        dot_product = np.dot(self.weights[neuron],
                             self.currents[self.n_iteration - 1])

        self.currents[self.n_iteration, neuron] = \
            tools.heaviside_activation(
                dot_product + self.noise[neuron, self.n_iteration])

    def update_all_currents(self):
        """
        Neurons are updated update in random order as described by Hopfield.
        The full hopfield_network should be updated before the same node gets
        updated again.
        """

        values = np.arange(0, self.num_neurons, 1)
        neuron_update_order = np.random.choice(values,
                                               self.num_neurons,
                                               replace=False)

        for neuron in neuron_update_order:
            self._update_current(neuron)

        self._update_pattern_similarity()

    def initialize(self):
        """

        """

        self._randomize_weights()
        self._compute_all_target_weights()
        self._combine_target_weights()
        self._compute_noise()


def main(force=False):
    """
    Instantiate Init(), Teacher() and Hopfield(), perform operations and plot.
    """

    bkp_file = f"bkp/hopfield.p"

    os.makedirs(os.path.dirname(bkp_file), exist_ok=True)

    if not os.path.exists(bkp_file) or force:

        np.random.seed(12345)

        network = Hopfield(
            num_iterations=50,
            num_neurons=0,
            p=2,
            f=0.51,
            first_p=0,
            learning_rate=0.1,
            forgetting_rate=0.5
        )

        network.initialize()

        pickle.dump(network, open(bkp_file, "wb"))

    else:
        print("Loading from pickle file...")
        network = pickle.load(open(bkp_file, "rb"))

    # plot.mean_weights(network)
    # plot.pattern_similarity(network)
    # plot.currents(network)
    # plot.present_weights(network)
    # tools.noise(network)
    # tools.energy(network)
    # plot.array_element_change(network.weights_history)
    # tools.array_element_change(network.theoretical_weights)
    # for i in range(len(network.theoretical_weights)-1):
    #     plot.array_history_index(network.theoretical_weights,
    #                              index=i+1, title="theoretical", contour=False)
    # for i in range(len(network.weights_history)-1):

    # Testing plots
    plot.array(
        array_like=network.weights,
        title="weights")

    plot.array_history_index(
        array_history=network.all_target_weights,
        title="all_target_weights",
        color_bar=False,
        contour=False)

    plot.array(
               array_like=network.combined_target_weights,
               title="combined_target_weights",
               contour=False)

    plot.multi_line(
        array_like=network.noise,
        title="noise")


if __name__ == '__main__':
    main(force=True)
