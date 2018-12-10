from random import random

import numpy as np

from .initializer import Initializer


class KMeansppInitializer(Initializer):
    def fit(self, source_data):
        centers = list(source_data[np.random.randint(source_data.shape[0], size=1), :])
        for _ in range(1, self.n_clusters):
            distances = np.array([np.mean(np.power(source_data - center, 2), axis=1) for center in centers])
            distances = distances.min(axis=0)

            distances_sum = distances.sum()
            distances_sum *= random()

            for idx, dis in enumerate(distances):
                distances_sum -= dis
                if distances_sum <= 0:
                    centers.append(source_data[idx, ])
                    break

        return centers
