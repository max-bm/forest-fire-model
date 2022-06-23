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


def evolve_forest(board):
    """
    Update forest according to forest-fire model defined by Drossel and Schwabl (1992).
    Each cell can be empty (0), occupied by a tree (1), or burning (2). 
    The update rules are:
    1. A burning cell turns into an empty cell.
    2. A tree will burn if at least one neighbour is burning.
    3. A tree ignites with probability f even when no neighbour is burning.
    4. An empty space fills with a tree with probability p.
    """
    return board


if __name__ == "__main__":
    forest = np.random.randint(0, 2, (20,20))
    sim_forest(forest, 100)    
