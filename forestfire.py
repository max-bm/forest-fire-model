#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import matplotlib.animation as animation
from kernels import von_neumann_kernel

def sim_forest(forest, num_generations, f=.5, p=.5, fig_size=(8, 8)):
    """
    Display evolution of the forest on screen (first 2 dimensions if d > 2) for given number of generations. 
    """
    fig, ax = plt.subplots(figsize=fig_size)
    plt.axis("off")
    imgs = []

    for i in range(num_generations + 1):
        imgs.append([plt.imshow(forest[(slice(None), slice(None)) + (0,)*(forest.ndim-2)], animated=True)])
        forest = evolve_forest(forest, f, p)

    ani = animation.ArtistAnimation(fig, imgs, interval=50, repeat_delay=0, blit=True)
    plt.show()

def evolve_forest(forest, f, p):
    """
    Update forest according to forest-fire model defined by Drossel and Schwabl (1992).
    Each cell can be empty (0), occupied by a tree (1), or burning (2). 
    The update rules are:
    1. A burning cell turns into an empty cell.
    2. A tree will burn if at least one neighbour is burning.
    3. A tree ignites with probability f even when no neighbour is burning.
    4. An empty space fills with a tree with probability p.
    """
    d = forest.ndim
    # Work with one-hot encoded forest and empty for update
    oh_forest = (np.arange(3) == forest[...,None]).astype(int) # (L^d, [empty, tree, burning])
    oh_update = np.zeros_like(oh_forest)
    nn_kernel = von_neumann_kernel(d)
    rule_1(oh_forest, oh_update)
    rule_2(oh_forest, oh_update, nn_kernel)
    # Generate probabilities for monte-carlo update
    probs = np.random.rand(*oh_forest[..., 0].shape)
    rule_3(oh_forest, oh_update, probs, f)
    rule_4(oh_forest, oh_update, probs, p)
    # Un-encode one_hot_vector
    updated_forest = np.argmax(oh_update, axis=2)
    return updated_forest

def rule_1(oh_forest, oh_update):
    """
    Forest-fire model rule 1: burning cells become empty cells.
    """
    oh_update[..., 0] += oh_forest[..., 2]

def rule_2(oh_forest, oh_update, nn_kernel):
    """
    Forest-fire model rule 2: trees with burning neighbours start burning.
    """
    # Convolve burning 'layer' with nearest-neighbour kernel to find cells with burning neighbours
    burning_neighbours = ndimage.convolve(oh_forest[..., 2], nn_kernel, mode='wrap')
    new_burning = np.array(oh_forest[..., 1] * burning_neighbours > 0, dtype=int)
    oh_update[..., 2] += new_burning
    # Now ignore these trees for next rules
    oh_forest[..., 1][new_burning] = 0
   
def rule_3(oh_forest, oh_update, probs, f):
    """
    Forest-fire model rule 3: trees spontaneously ignite with probability f.
    """
    new_burning = np.array(oh_forest[..., 1] * probs > (1. - f), dtype=int)
    oh_update[..., 2] += new_burning
    # Add trees which didn't combust into new tree layer
    oh_update[..., 1] += (1 - new_burning) * oh_forest[..., 1]
    
def rule_4(oh_forest, oh_update, probs, p):
    """
    Forest-fire model rule 4: trees spontaneously grow with probability p.
    """
    new_trees = np.array(oh_forest[..., 0] * probs > (1. - p), dtype=int)
    oh_update[..., 1] += new_trees
    # Put old empties back into empty layer
    oh_update[..., 0] += (1 - new_trees) * oh_forest[..., 0]


if __name__ == "__main__":
    print("Forest-fire model simulation.")
    grid_size = input("Enter grid size:")
    d = input("Enter the dimensionality of the system:")
    f = input("Enter tree ignition probability, f:")
    p = input("Enter tree growth probability, p:")
    generations = input("Enter number of generations:")
    forest = np.random.randint(0, 3, (int(grid_size), ) * int(d)) 
    sim_forest(forest, int(generations), float(f), float(p)) 
