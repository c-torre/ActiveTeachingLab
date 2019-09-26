import os
import pickle

import numpy as np

import tools.functions as tools
import tools.plot as plot


class Hopfield:
    """
    Pattern consists of multiple binary vectors representing both the item and
    its different characteristics that can be recalled.

    :param learning_rate: float, proportion of the theoretical weights learn
        per time step
    :param forgetting_rate: float, proportion of the theoretical weights
        forgotten per time step
    """

    version = 3.0
    bounds = ('learning_rate', 10**-7, 0.99), \
             ('forgetting_rate', 10**-7, 0.99),

    def __init__(self, num_iterations, num_neurons=1000, p=16, f=0.1,
                 inverted_fraction=0.3, noise_variance=65,
                 noise_modulation=0.05, first_p=0, learning_rate=0.3,
                 forgetting_rate=0.1, **kwargs):

        super().__init__(**kwargs)

        self.num_iterations = num_iterations
        self.n_iteration = 0

        self.num_neurons = num_neurons
        self.p = p
        self.f = f
        self.first_p = first_p
        self.inverted_fraction = inverted_fraction

        self.noise_variance = noise_variance
        self.noise_modulation = noise_modulation
        self.noise = np.zeros((num_neurons, num_iterations))

        self.learning_rate = learning_rate
        self.forgetting_rate = forgetting_rate
        assert (self.learning_rate and self.forgetting_rate) <= 1

        self.patterns = \
            np.random.choice([0, 1], p=[1 - self.f, self.f],
                             size=(self.p, self.num_neurons))
        self.pattern_similarity = np.zeros((
            self.p, self.num_iterations))
        print("\nPatterns:\n", self.patterns)

        self.weights = np.random.random((self.num_neurons, self.num_neurons))
        self.next_weights = np.zeros_like(self.weights)
        self.weights_history = np.zeros((self.num_iterations, self.num_neurons,
                                         self.num_neurons))
        self.weights_mean = np.zeros(self.num_iterations)

        self.next_theoretical_weights = np.zeros_like(self.weights)
        self.theoretical_weights = np.zeros((self.p, self.num_neurons,
                                             self.num_neurons), dtype=int)

        self.currents = np.zeros((self.num_iterations, self.num_neurons),
                                 dtype=int)

    ###################
    # NETWORK METHODS #
    ###################

    def update_weights_history(self):
        self.weights_history[self.n_iteration] = np.copy(self.weights)

    def update_theoretical_weights_history(self, p):
        """
        Adds the current theoretical weights the the current ones and appends
        to the history list.
        """
        self.next_theoretical_weights += self.theoretical_weights[-1]

        self.theoretical_weights[p] =\
            np.copy(self.next_theoretical_weights)

    def calculate_next_theoretical_weights(self, pattern, p):
        """
        Calculate the weights after the presentation of a new pattern but does
        not change the current weights of the network.
        """
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if j >= i:
                    break

                self.next_theoretical_weights[i, j] += (2 * pattern[i] - 1) \
                    * (2 * pattern[j] - 1)

        self.next_theoretical_weights += self.next_theoretical_weights.T
        self.update_theoretical_weights_history(p)

    def compute_all_theoretical_weights(self):
        """
        For every pattern, calculate its theoretical weights and add them to
        the history.
        """
        for p in range(len(self.patterns)):
            self.calculate_next_theoretical_weights(self.patterns[p], p)

    def update_weights(self, weights):
        self.weights += weights

    def compute_noise(self):
        if self.n_iteration > 50 and self.num_neurons > 100:
            print("Computing modulated Gaussian noise...")

        for i in range(self.num_neurons):
            for j in range(self.num_iterations):
                self.noise[i, j] = tools.modulated_gaussian_noise(
                    self.noise_variance, self.noise_modulation
                )

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

        self.currents[self.n_iteration, neuron] =\
            tools.heaviside_activation(
                dot_product + self.noise[neuron, self.n_iteration])

    def _update_pattern_similarity(self):
        """
        Compute the last pattern similarity and append it to the network
        history.

        The problem pattern is either computed from binarizing the given int
        or taken from the stored p patterns of the network.

        :param item: int, item id
        :param n_pattern: array_like
        """

        for p in range(self.p):
            self.pattern_similarity[p, self.n_iteration] = \
                tools.compute_pattern_similarity(
                    self.currents[self.n_iteration],
                    self.patterns[p])

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

    def update_n_iteration(self):
        self.n_iteration += 1

    ###########################
    # ACTIVE TEACHING METHODS #
    ###########################

    def p_recall(self, item, time=None):
        """Expected return from specific learner: p_r"""
        p_r = np.random.random() * self.p / self.p  # making PEP8 happy; static
        return p_r

    def decide(self, item, possible_replies, time=None, time_index=None):
        p_r = self.p_recall(item,
                            time=time)
        r = np.random.random()

        if p_r > r:
            reply = item
        else:
            reply = np.random.choice(possible_replies)

        # if self.verbose:
        #     print(f't={self.t}: question {item}, reply {reply}')
        return reply

    def learn(self, index, item=None, time=None):
        """
        The normalized difference of means calculated at every time step gives
        a logarithmic emergent behavior as the weights get closer to the
        theoretical ones.

        :param item:
        :param time:
        """

        self.next_weights = (self.theoretical_weights[index]
                             - self.weights) * self.learning_rate

        self.update_weights(self.next_weights)

        self.weights_mean[self.self.n_iteration] = \
            np.mean(self.weights) - np.mean(self.next_theoretical_weights)

        self.update_weights_history()

    def unlearn(self):
        pass

    def forget(self):
        noise = np.zeros_like(self.weights)
        for i in np.nditer(noise, op_flags=["readwrite"]):
            i += tools.modulated_gaussian_noise(self.noise_variance,
                                                self.noise_modulation)

        self.next_weights = - np.copy(self.weights) * self.forgetting_rate \
            + noise

        self.update_weights(self.next_weights)

        self.weights_mean[self.n_iteration] =\
            np.mean(self.weights) - np.mean(self.next_theoretical_weights)

        self.update_weights_history()
        # print(self.weights)

    def learn_from_naive(self):
        print("DEBUG: I'm learning for theoretical weights index -1!")
        self.compute_all_theoretical_weights()
        self.compute_noise()
        self.update_all_currents()
        for i in range(self.num_iterations):
            self.learn(-1)
            self.update_all_currents()
            self.update_n_iteration()

    def learn_more_patterns(self):
        print("DEBUG: I'm learning pattern -1 after learning pattern -2")
        assert self.p > 1

        self.compute_all_theoretical_weights()
        self.compute_noise()
        self.update_weights(self.theoretical_weights[-2])
        self.update_all_currents()
        for i in range(self.num_iterations):
            # self.learn(-1)
            self.update_all_currents()
            self.update_n_iteration()

    def test_forgetting(self):
        self.compute_all_theoretical_weights()
        self.compute_noise()
        self.update_weights(self.theoretical_weights[-1])
        self.update_all_currents()
        plot.present_weights(self)
        # for p in range(self.p, 0, -1):
        #     self.update_weights(self.theoretical_weights[-p])
        #     self.update_all_currents()
        # self.update_weights(self.theoretical_weights[-1])
        # self.update_all_currents()
        for i in range(self.num_iterations):
            self.forget()
            self.update_all_currents()
            self.update_n_iteration()


def main(force=False):

    bkp_file = f"bkp/hopfield.p"

    os.makedirs(os.path.dirname(bkp_file), exist_ok=True)

    if not os.path.exists(bkp_file) or force:

        np.random.seed(12345)

        network = Hopfield(
            num_iterations=50,
            num_neurons=10,
            f=0.51,
            p=2,
            first_p=0,
            inverted_fraction=0.51,
            learning_rate=0.1,
            forgetting_rate=0.5
        )

        network.test_forgetting()
        print(network.theoretical_weights)

        pickle.dump(network, open(bkp_file, "wb"))

    else:
        print("Loading from pickle file...")
        network = pickle.load(open(bkp_file, "rb"))

    plot.mean_weights(network)
    # plot.pattern_similarity(network)
    # plot.currents(network)
    plot.present_weights(network)
    # tools.noise(network)
    # tools.energy(network)
    plot.array_element_change(network.weights_history)
    # tools.array_element_change(network.theoretical_weights)
    # for i in range(len(network.theoretical_weights)-1):
    #     plot.array_history_index(network.theoretical_weights,
    #                              index=i+1, title="theoretical", contour=False)
    # for i in range(len(network.weights_history)-1):
    plot.array_history_index(
        network.weights_history,
        title="evolution", contour=False)


if __name__ == "__main__":
    main(force=True)
