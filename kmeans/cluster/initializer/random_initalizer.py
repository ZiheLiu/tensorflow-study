import numpy as np

from .initializer import Initializer


class RandomInitializer(Initializer):
    def fit(self, source_data):
        return source_data[np.random.randint(source_data.shape[0], size=self.n_clusters), :]
