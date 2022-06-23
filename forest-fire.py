#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import matplotlib.animation as animation

def sim_forest(board, num_generations, fig_size=(8, 8)):
    """
    Display evolution of the forest on screen (first 2 dimensions if d > 2) for given number of generations. 
    """
    fig, ax = plt.subplots(figsize=fig_size)
    plt.axis("off")
    imgs = []

    for i in range(num_generations + 1):
        imgs.append([plt.imshow(board, animated=True)])
        board = evolve_forest(board)

    ani = animation.ArtistAnimation(fig, imgs, interval=50, repeat_delay=0, blit=True)
    plt.show()


def evolve_forest(forest, f=.5, p=0.):
    """
    Update forest according to forest-fire model defined by Drossel and Schwabl (1992).
    Each cell can be empty (0), occupied by a tree (1), or burning (2). 
    The update rules are:
    1. A burning cell turns into an empty cell.
    2. A tree will burn if at least one neighbour is burning.
    3. A tree ignites with probability f even when no neighbour is burning.
    4. An empty space fills with a tree with probability p.
    """
    # Work with one-hot encoded forest and empty for update
    one_hot_forest = (np.arange(3) == forest[...,None]).astype(int) # (L^d, [empty, tree, burning])
    new_one_hot_forest = np.zeros_like(one_hot_forest)
    # Previously burning cells now empty - RULE 1.
    new_one_hot_forest[..., 0] += one_hot_forest[..., 2]
    # Convolve burning 'layer' with nearest-neighbour kernel to find cells with burning neighbours
    nn_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) # To be made d-dimensional and user-specified.
    burning_neighbours = ndimage.convolve(one_hot_forest[..., 2], nn_kernel, mode='wrap')
    # And now set trees with burning neighbours to be burning - RULE 2.
    new_burning = np.array(one_hot_forest[..., 1] * burning_neighbours > 0, dtype=int)
    new_one_hot_forest[..., 2] += new_burning
    # Now ignore these trees for next rules
    one_hot_forest[..., 1][new_burning] = 0
    # Generate probabilities for monte-carlo update
    probs = np.random.rand(*one_hot_forest[..., 0].shape)
    new_burning = np.array(one_hot_forest[..., 1] * probs > (1. - f), dtype=int)
    # Add new burning trees to new burning layer - RULE 3.
    new_one_hot_forest[..., 2] += new_burning
    # Add trees which didn't combust into new tree layer
    new_one_hot_forest[..., 1] += (1 - new_burning) * one_hot_forest[..., 1]
    # Add new sprouted trees to new tree layer - RULE 4.
    new_trees = np.array(one_hot_forest[..., 0] * probs > (1. - p), dtype=int)
    new_one_hot_forest[..., 1] += new_trees
    # Put old empties back into empty layer
    new_one_hot_forest[..., 0] += (1 - new_trees) * one_hot_forest[..., 0]
    # Un-encode one_hot_vector
    updated_forest = np.argmax(new_one_hot_forest, axis=2)
    return updated_forest


if __name__ == "__main__":
    forest = np.random.randint(0, 3, (100,100)) # Make grid size user input, same with probs
    sim_forest(forest, 100) 

