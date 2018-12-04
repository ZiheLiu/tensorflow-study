import numpy as np

from abc import ABCMeta, abstractmethod


class Initializer(object):
    __metaclass__ = ABCMeta

    def __init__(self, n_clusters: int):
        """Constructor of Initializer.

        Args:
            n_clusters: total number of clusters.
        """
        self.n_clusters = n_clusters

    @abstractmethod
    def fit(self, source_data: np.ndarray) -> np.ndarray:
        """Generate initial center positions from source_data.

        Args:
            source_data: shape is (n_instances, ?), positions of all data.
        Return:
            shape is (self.n_clusters, ?).
        """
        pass
