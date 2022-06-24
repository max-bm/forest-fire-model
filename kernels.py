import numpy as np

def von_neumann_kernel(d):
    """
    Function generates a von Neumann neighbourhood kernel in d dimensions.
    """
    kernel = np.zeros((3,) * d)
    for i in range(d):
        kernel[(1,) * i + (slice(None),) + (1,) * (d-i-1)] = [1, 0, 1]
    return kernel

