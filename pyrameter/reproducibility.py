import numpy as np


class GlobalRNG():
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed=seed)
        self.seed = seed

    def set_seed(self, seed=None):
        """Restart the RNG with a new seed in place."""
        self.rng = np.random.RandomState(seed=seed)
        self.seed = seed


RNG = GlobalRNG()
