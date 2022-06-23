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

    ani = animation.ArtistAnimation(fig, imgs, interval=250, repeat_delay=0, blit=True)
    plt.show()


def evolve_forest(forest, f=1., p=1.):
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
    one_hot_forest = (np.arange(forest.max()+1) == forest[...,None]).astype(int) # (L^d, [empty, tree, burning])
    new_one_hot_forest = np.zeros_like(one_hot_forest)
    # Previously burning cells now empty - RULE 1.
    new_one_hot_forest[..., 0] = one_hot_forest[..., 2]
    # Convolve burning 'layer' with nearest-neighbour kernel to find cells with burning neighbours
    nn_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) # To be made d-dimensional and user-specified.
    burning_neighbours = ndimage.convolve(one_hot_forest[..., -1], nn_kernel, mode='wrap')
    # And now set trees with burning neighbours to be burning - RULE 2.
    new_one_hot_forest[..., 2] = np.array(one_hot_forest[..., 1] * burning_neighbours > 0, dtype=int)


if __name__ == "__main__":
    forest = np.random.randint(0, 3, (5,5))
#    sim_forest(forest, 100)    
    one_hot_forest = evolve_forest(forest)
    print(forest.shape)
    print(one_hot_forest.shape)
    print(forest)
    print()
    print(one_hot_forest[:, :, -1])

