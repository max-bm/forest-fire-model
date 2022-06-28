#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import matplotlib.animation as animation
from kernels import von_neumann_kernel


class forest():
    """
    Forest object with update methods for simulating the forest-fire model rules.
    """    
    def __init__(self, grid_size=100, n_dim=2, burn_p=0.1, grow_p=0.1):
        self.n_dim = n_dim
        self.state = np.random.randint(0, 3, (int(grid_size), ) * int(self.n_dim)) 
        self.nn_kernel = von_neumann_kernel(self.n_dim)
        self.burn_p = burn_p
        self.grow_p = grow_p
        self.probs = np.random.rand(*self.state.shape)
    
    def _forest_to_one_hot(self, forest):
        oh_forest = (np.arange(3) == forest[...,None]).astype(int) # (L^d, [empty, tree, burning])
        return oh_forest

    def _one_hot_to_forest(self, oh_forest):
        return np.argmax(oh_forest, axis=-1)

    def _rule_1(self, oh_forest, oh_update):
        """
        Forest-fire model rule 1: burning cells become empty cells.
        """
        oh_update[..., 0] += oh_forest[..., 2]

    def _rule_2(self, oh_forest, oh_update):
        """
        Forest-fire model rule 2: trees with burning neighbours start burning.
        """
        # Convolve burning 'layer' with nearest-neighbour kernel to find cells with burning neighbours
        burning_neighbours = ndimage.convolve(oh_forest[..., 2], self.nn_kernel, mode='wrap')
        new_burning = np.array(oh_forest[..., 1] * burning_neighbours > 0, dtype=int)
        oh_update[..., 2] += new_burning
        # Now ignore these trees for next rules
        oh_forest[..., 1][new_burning] = 0
           
    def _rule_3(self, oh_forest, oh_update):
        """
        Forest-fire model rule 3: trees spontaneously ignite with probability f.
        """
        new_burning = np.array(oh_forest[..., 1] * self.probs > (1. - self.burn_p), dtype=int)
        oh_update[..., 2] += new_burning
        # Add trees which didn't combust into new tree layer
        oh_update[..., 1] += (1 - new_burning) * oh_forest[..., 1]
        
    def _rule_4(self, oh_forest, oh_update):
        """
        Forest-fire model rule 4: trees spontaneously grow with probability p.
        """
        new_trees = np.array(oh_forest[..., 0] * self.probs > (1. - self.grow_p), dtype=int)
        oh_update[..., 1] += new_trees
        # Put old empties back into empty layer
        oh_update[..., 0] += (1 - new_trees) * oh_forest[..., 0]

    def update(self):
        """
        Function to execute forest update accordnig to rules, using one-hot encoded forest for 
        acceleration of rule 2.
        """
        # Work with one-hot encoded forest and empty for update
        oh_forest = self._forest_to_one_hot(self.state)
        oh_update = np.zeros_like(oh_forest)
        self._rule_1(oh_forest, oh_update)
        self._rule_2(oh_forest, oh_update)
        self._rule_3(oh_forest, oh_update)
        self._rule_4(oh_forest, oh_update)
        # Un-encode one_hot_vector
        self.state = self._one_hot_to_forest(oh_update)


def sim_forest(forest, n_gen, fig_size=(8, 8)):
    """
    Display evolution of the forest on screen (first 2 dimensions if d > 2) for given number of generations. 
    """
    fig, ax = plt.subplots(figsize=fig_size)
    plt.axis("off")
    imgs = []

    for i in range(n_gen + 1):
        imgs.append([plt.imshow(forest.state[(slice(None), slice(None)) + (0,)*(forest.n_dim-2)], animated=True)])
        forest.update()

    ani = animation.ArtistAnimation(fig, imgs, interval=100, repeat_delay=0, blit=True)
    plt.show()


if __name__ == "__main__":
    print("Forest-fire model simulation.")
    grid_size = int(input("Enter grid size: "))
    n_dim = int(input("Enter the dimensionality of the system: "))
    burn_p = float(input("Enter tree ignition probability, f: "))
    grow_p = float(input("Enter tree growth probability, p: "))
    n_gen = int(input("Enter number of generations: "))
    forest = forest(grid_size, n_dim, burn_p, grow_p)
    sim_forest(forest, n_gen)

