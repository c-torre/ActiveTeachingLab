import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

FIG_FOLDER = "fig"
os.makedirs(FIG_FOLDER, exist_ok=True)


def currents(network):

    data = network.currents

    fig, ax = plt.subplots()
    ax.imshow(data)
    ax.set_aspect("auto")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")

    ax.set_title("Network currents history")
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Iteration")

    plt.tight_layout()

    plt.savefig(os.path.join(FIG_FOLDER, "currents.pdf"))


def present_weights(network):

    fig, ax = plt.subplots()
    # im = ax.contourf(network.weights)
    im = plt.imshow(network.weights)
    ax.set_aspect("auto")

    ax.set_title("Weights matrix")
    ax.set_xlabel("Neuron $i$")
    ax.set_ylabel("Neuron $j$")

    plt.tight_layout()

    fig.colorbar(im, ax=ax)

    plt.savefig(os.path.join(FIG_FOLDER, "present_weights.pdf"))


def array(array_like, title, color_bar=True, contour=False):

    fig, ax = plt.subplots()

    im = plt.imshow(array_like)
    ax.set_aspect("auto")
    ax.set_title(f"{title}")

    plt.tight_layout()

    if color_bar:
        fig.colorbar(im, ax=ax)
    if contour:
        ax.contourf(array_like)

    plt.savefig(os.path.join(FIG_FOLDER, f"{title}.pdf"))


def array_history_index(array_history, title, color_bar=False, contour=False):

    n_subplot = len(array_history)

    fig, axes = plt.subplots(ncols=n_subplot)

    for index in range(n_subplot):

        ax = axes[index]

        if contour:
            ax.contourf(array_history[index])
        im = ax.imshow(array_history[index])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        if color_bar:
            plt.colorbar(im, ax=ax)

    plt.suptitle(f"{title}")

    plt.savefig(os.path.join(FIG_FOLDER, f"{title}.pdf"))


def array_element_change(array_history, alpha=0.7):

    history = array_history
    diff = np.diff(history, axis=0)

    fig, ax = plt.subplots()

    for j in tqdm(range(diff.shape[1])):
        for k in range(diff.shape[2]):
            a = np.zeros(diff.shape[0])
            for i in range(diff.shape[0]):
                a[i] = diff[i, j, k]
            ax.plot(a, alpha=alpha)

    ax.set_title("Matrix element change")
    ax.set_xlabel("Iteration")
    # ax.set_xticks(np.arange(diff.shape[0]))
    ax.set_ylabel("Point value change")

    plt.savefig(os.path.join(FIG_FOLDER, "array_element_change.pdf"))


def theoretical_weights(network, index):

    fig, ax = plt.subplots()
    # im = ax.contourf(network.theoretical_weights[index])  # fancy
    im = ax.imshow(network.theoretical_weights_history[index])
    ax.set_aspect("auto")

    ax.set_title(f"Theoretical weights (iter={index})")
    ax.set_xlabel("Neuron $i$")
    ax.set_ylabel("Neuron $j$")

    plt.tight_layout()

    fig.colorbar(im, ax=ax)
    plt.savefig(os.path.join(FIG_FOLDER, "theoretical_weights.pdf"))


def mean_weights(network):

    fig, ax = plt.subplots()
    ax.plot(network.weights_mean)

    ax.set_title("Weights learning rule")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Difference of means")

    plt.savefig(os.path.join(FIG_FOLDER, "mean_weights.pdf"))


def pattern_similarity(network):

    fig, ax = plt.subplots()
    # ax.tools(network.pattern_similarity)  # For single line; comment loop

    for p in range(network.p):
        ax.plot(network.pattern_similarity[p])

    ax.set_title("Pattern similarity")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Pattern match$^2$ CHANGEEEEE")
    ax.set_ylim((-0.1, 1.1))

    plt.savefig(os.path.join(FIG_FOLDER, "pattern_similarity.pdf"))


def energy(network):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.arange(0, network.num_neurons, 1)
    y = x
    x, y = np.meshgrid(x, y)
    z = np.copy(network.weights)

    for i in range(network.num_neurons):
        for j in range(network.num_neurons):
            z[i, j] *= network.currents[-1, j]

    ax.plot_surface(x, y, z, alpha=0.9, cmap="viridis", antialiased=True)

    ax.set_title("Energy landscape")
    ax.set_xlabel("Neuron $_i$")
    ax.set_ylabel("Neuron $_j$")
    ax.set_zlabel("Energy")

    plt.savefig(os.path.join(FIG_FOLDER, "energy.pdf"))


def multi_line(array_like, title):

    fig, ax = plt.subplots()

    for n in array_like:
        ax.plot(n, linewidth=0.5, alpha=0.9)

    ax.set_title(f"{title}")

    plt.savefig(os.path.join(FIG_FOLDER, f"{title}.pdf"))
