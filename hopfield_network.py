import os
import pickle

# from tqdm import tqdm
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import numpy as np

import hopfield_tools as tools
import plot.plot_tools as plot


class Hopfield:
    """
    Pattern consists of multiple binary vectors representing both the item and
    its different characteristics that can be recalled.

    :param learning_rate: float, proportion of the theoretical weights learn
        per time step
    :param forgetting_rate: float, proportion of the theoretical weights
        forgotten per time step
    """

    version = 1.0
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

        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        self.next_theoretical_weights = np.zeros_like(self.weights)
        self.next_weights = np.zeros_like(self.weights)
        self.weights_history = [np.zeros_like(self.weights)]
        self.theoretical_weights_history = [np.zeros_like(self.weights)]
        self.weights_mean = []

        self.pattern_similarity = np.zeros((
            self.p, self.num_iterations))

        self.active_fraction = f

        self.initial_currents = np.zeros(self.num_neurons)

        self.patterns = \
            np.random.choice([0, 1], p=[1 - self.f, self.f],
                             size=(self.p, self.num_neurons))
        print("\nPatterns:\n", self.patterns)

        self.currents = np.zeros((1, self.num_neurons), dtype=int)
        self.patterns_evolution = None

        self.question_pattern = np.zeros(self.num_neurons)

    ###################
    # NETWORK METHODS #
    ###################

    def _initialize_currents(self):
        """Initial currents are set to the first distorted pattern."""

        self.currents = np.copy(tools.distort_pattern(
            self.patterns[self.first_p],
            self.inverted_fraction)
        )

        # print("\nInitial currents:\n", self.currents)

    def update_weights_history(self):
        self.weights_history.append(np.copy(self.weights))

    def update_theoretical_weights_history(self):
        """
        Adds the current theoretical weights the the current ones and appends
        to the history list.
        """
        self.next_theoretical_weights += self.theoretical_weights_history[-1]

        self.theoretical_weights_history.append(
            np.copy(self.next_theoretical_weights)
        )

    def calculate_next_theoretical_weights(self, pattern):
        """
        Calculate the weights after the presentation of a new pattern but does
        not change the current weights of the network.
        """
        for i in (range(self.num_neurons)):  # tqdm
            for j in range(self.num_neurons):
                if j >= i:
                    break

                self.next_theoretical_weights[i, j] += (2 * pattern[i] - 1) \
                    * (2 * pattern[j] - 1) \

        self.next_theoretical_weights += self.next_theoretical_weights.T
        self.update_theoretical_weights_history()

    def compute_all_theoretical_weights(self):
        """
        For every pattern, calculate its theoretical weights and add them to
        the history.
        """
        for p in range(len(self.patterns)):
            self.calculate_next_theoretical_weights(self.patterns[p])

    def update_weights(self, weights):
        self.weights += weights

    def compute_weights_all_patterns(self):
        """Testing method"""
        print(f"\n...Computing weights for all patterns...\n")

        for p in (range(len(self.patterns))):  # tqdm
            self.calculate_next_theoretical_weights(self.patterns[p])
            self.update_weights(self.next_theoretical_weights)

        print("Done!")

    def compute_noise(self):
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
        dot_product = np.dot(self.weights[neuron], self.currents[-2])

        self.currents[-1, neuron] =\
            tools.heaviside_activation(
                dot_product
                + self.noise[neuron, self.n_iteration])

    def update_all_neurons(self):
        """
        Neurons are updated update in random order as described by Hopfield.
        The full hopfield_network should be updated before the same node gets
        updated again.
        """
        # self.last_currents = self.currents

        values = np.arange(0, self.num_neurons, 1)
        update_order = np.random.choice(values,
                                        self.num_neurons,
                                        replace=False)

        self.currents = np.vstack((self.currents, np.zeros(self.num_neurons)))

        for i in update_order:
            self._update_current(i)

        # self.currents_history = np.vstack((self.currents_history,
        #                                    self.currents))

    def _update_current_learning(self, neuron):
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
        random_currents = \
            np.random.choice([0, 1], p=[1 - self.f, self.f],
                             size=self.num_neurons)
        dot_product = np.dot(self.weights[neuron], random_currents)

        self.currents[-1, neuron] =\
            tools.heaviside_activation(
                dot_product
                + tools.modulated_gaussian_noise(self.noise_variance,
                                                 self.noise_modulation)
            )

    def update_all_neurons_learning(self):
        """
        Neurons are updated update in random order as described by Hopfield.
        The full hopfield_network should be updated before the same node gets
        updated again.
        """
        # self.last_currents = self.currents

        values = np.arange(0, self.num_neurons, 1)
        neuron_update_order = np.random.choice(values,
                                               self.num_neurons,
                                               replace=False)

        self.currents = np.vstack((self.currents, np.zeros(self.num_neurons)))

        for neuron in neuron_update_order:
            self._update_current(neuron)

        # self.currents_history = np.vstack((self.currents_history,
        #                                    self.currents))

    def _compute_patterns_evolution(self):

        for p in range(self.p):
            similarity = np.sum(self.currents[-1] == self.patterns[p])
            self.patterns_evolution = \
                np.vstack((self.patterns_evolution, similarity))

        self.patterns_evolution = self.patterns_evolution.T
        self.patterns_evolution = self.patterns_evolution[0, 1:]

    def _find_attractor(self):
        """
        If an update does not change any of the node values, the
        hopfield_network rests at an attractor and updating stops.
        """
        tot = 1

        # np.sum(self.currents - self.last_currents) != 0:
        while (self.currents[-1, :] != self.currents[-2, :]).all() or tot < 2:
            self.update_all_neurons()
            self._compute_patterns_evolution()
            tot += 1
            print(f"\nUpdate {tot} finished.\n")

        attractor = np.int_(np.copy(self.currents[-1]))

        print(f"\nFinished as attractor {attractor} after {tot} "
              f"node value updates.\n")

    def update_pattern_similarity(self, item=None, n_pattern=None):
        """
        Compute the last pattern similarity and append it to the network
        history.

        The problem pattern is either computed from binarizing the given int
        or taken from the stored p patterns of the network.

        :param item: int, item id
        :param n_pattern: array_like
        """
        assert (item is not None and n_pattern is None) \
            or (n_pattern is not None and item is None)
        if item is not None:
            bin_item = tools.binarize_item(item, self.num_neurons)
        # elif n_pattern is not None:
        #     bin_item = self.patterns[n_pattern]
        elif n_pattern is not None:
            bin_item = self.patterns
        else:
            raise Exception("Item or n_pattern should be given")

        for p in range(self.p):
            self.pattern_similarity[p, self.n_iteration] =\
                tools.compute_pattern_similarity(self.currents[-1], bin_item[p])

        # similarity = tools.compute_pattern_similarity(
        #     self.currents[-1],
        #     bin_item
        # )
        #
        # self.pattern_similarity.append(similarity)

    def simulate(self):
        # assert self.patterns
        # assert self.num_neurons == self.patterns[0].size

        # self._initialize()
        self.compute_weights_all_patterns()
        self._initialize_currents()
        self.update_all_neurons()
        self._find_attractor()

    def update_n_iteration(self):
        self.n_iteration += 1

    ###########################
    # ACTIVE TEACHING METHODS #
    ###########################

    def p_recall(self, item, time=None):
        """Expected return from specific learner: p_r"""
        p_r = np.random.random() * self.p / self.p  # making PEP8 happy
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

    def learn(self, item=None, time=None):
        """
        The normalized difference of means calculated at every time step gives
        a logarithmic emergent behavior as the weights get closer to the
        theoretical ones.

        :param item:
        :param time:
        """

        self.next_weights = (self.next_theoretical_weights - self.weights) \
            * self.learning_rate

        self.update_weights(self.next_weights)

        self.weights_mean.append(-np.mean(self.next_theoretical_weights)
                                 + np.mean(self.weights))

        self.weights_history.append(self.weights)

    def unlearn(self):
        pass

    def fully_learn(self):
        # tot = 1
        #
        # while (self.weights[-1] != self.next_theoretical_weights).all():
        #     self.learn()
        #     tot += 1
        #
        # print(f"\nFinished learning after {tot} "
        #       f"node weight updates.\n")
        pass

    def forget(self, constant=10000000):
        self.next_weights = \
            (self.weights + tools.modulated_gaussian_noise(
                self.noise_variance, self.noise_modulation) * constant) \
            * self.forgetting_rate

        self.update_weights(self.next_weights)

        self.weights_mean.append(-np.mean(self.next_theoretical_weights)
                                 + np.mean(self.weights))

    def simulate_learning(self, iterations, recalled_pattern):

        if self.p == 1:
            self.calculate_next_theoretical_weights(self.patterns[self.first_p])
            self.update_all_neurons()
        else:
            # for p in range(self.p - 2):
            for p in range(len(self.patterns - 1)):
                self.calculate_next_theoretical_weights(self.patterns[p])
                self.update_weights(self.next_theoretical_weights)

            # self.currents = np.vstack((self.currents,
                                       # np.zeros(self.num_neurons)))
            self.calculate_next_theoretical_weights(self.patterns[self.p - 1])
            self.update_all_neurons()
        # self.update_all_neurons()

        self.update_pattern_similarity(n_pattern=recalled_pattern)

        for i in range(iterations):
            self.learn()
            self.update_all_neurons()
            self.update_pattern_similarity(n_pattern=recalled_pattern)


def main(force=False):

    bkp_file = f"bkp/hopfield.p"

    os.makedirs(os.path.dirname(bkp_file), exist_ok=True)

    if not os.path.exists(bkp_file) or force:

        np.random.seed(123)

        network = Hopfield(
            num_iterations=3,
            num_neurons=3,
            f=0.45,
            p=3,
            first_p=0,
            inverted_fraction=0.51,
            learning_rate=0.001,
            forgetting_rate=1
        )

        ##########################
        # #### TESTING AREA #### #
        ##########################

        network.compute_all_theoretical_weights()
        network.compute_noise()
        # network.update_all_neurons()

        for i in range(network.num_iterations):
            network.learn()
            network.update_all_neurons_learning()
            network.update_pattern_similarity(n_pattern=0)
            network.update_n_iteration()
        # print(network.weights_history)
        print(network.pattern_similarity)

        # # learning loop
        # network.calculate_next_weights(hopfield_network.patterns[0])
        # network.update_weights(hopfield_network.next_theoretical_weights)
        # network.calculate_next_weights(hopfield_network.patterns[0])
        # network.update_all_neurons()
        #
        # network.update_pattern_similarity(n_pattern=0)
        #
        # for i in range(20):
        #     network.learn()
        #     network.update_all_neurons_learning()
        #     network.update_pattern_similarity(n_pattern=0)

        # network.compute_noise()
        # network.calculate_next_weights(network.patterns[0])
        # print(network.theoretical_weights_history)
        # network.calculate_next_weights(network.patterns[1])
        # for j in range(len(network.patterns)):
        #     network.calculate_next_theoretical_weights(network.patterns[j])
        # print(network.theoretical_weights_history)
        # network.update_weights(network.next_theoretical_weights)
        # network.calculate_next_weights(network.patterns[1])
        # network.update_weights(network.next_theoretical_weights)
        # print('-' * 10)
        # print(network.theoretical_weights_history[0])
        # print(network.theoretical_weights_history[1])
        # print(network.theoretical_weights_history[2])

        # network.update_pattern_similarity(n_pattern=1)
        #
        # for i in range(network.num_iterations):
        #     network.learn()
        #     network.update_all_neurons_learning()
        #     network.update_pattern_similarity(n_pattern=1)
        # plot.weights(network)

        # hopfield_network.simulate_learning(iterations=100,
        # recalled_pattern=2)

        # network.weights = np.zeros_like(network.next_theoretical_weights)
        # network.next_theoretical_weights = np.zeros_like(network.weights)
        # network.compute_weights_all_patterns()
        # plot.weights(network)

        #################################
        # #### END OF TESTING AREA #### #
        #################################

        pickle.dump(network, open(bkp_file, "wb"))

    else:
        print("Loading from pickle file...")
        network = pickle.load(open(bkp_file, "rb"))

    plot.mean_weights(network)
    plot.pattern_similarity(network)
    plot.currents(network)
    plot.present_weights(network)
    plot.noise(network)
    plot.energy(network)
    plot.array_element_change(network.weights_history)
    plot.array_element_change(network.theoretical_weights_history)
    for i in range(len(network.theoretical_weights_history)-1):
        plot.array_history_index(network.theoretical_weights_history,
                                 index=i+1, contour=False)
    for i in range(len(network.weights_history)-1):
        plot.array_history_index(network.weights_history,
                                 index=i+1, contour=False)


if __name__ == '__main__':
    main(force=True)
