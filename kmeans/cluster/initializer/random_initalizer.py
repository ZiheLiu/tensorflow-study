import numpy as np

from .initializer import Initializer


class RandomInitializer(Initializer):
    def fit(self, source_data):
        return np.random.sample(source_data, self.n_clusters)
