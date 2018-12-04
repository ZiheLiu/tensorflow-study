import numpy as np

from .initializer import RandomInitializer


class KMeans(object):
    def __init__(self, n_clusters, init):
        self.n_clusters = n_clusters
        self.initializer = self._get_initializer(init)

        self.centers = None

    def _get_initializer(self, init):
        if init == 'random':
            return RandomInitializer(self.n_clusters)
        elif init == 'kmeans++':
            pass
        raise ValueError('Argument <init> must be "random" or "kmeans++", but get: {}'.format(init))

    def fit(self, source_data: np.ndarray) -> np.ndarray:
        """Cluster source_data and return predicted labels.

        Args:
            source_data: shape is (n_instances, ?), each row is a instance.

        Return:
            shape is (n_instances,), label for each instance.
        """
        self.centers = self.initializer.fit(source_data)

        changed = True
        labels = None

        while changed:
            distances = np.array([np.mean(np.power(source_data - center, 2), axis=1) for center in self.centers])
            new_labels = np.argmin(distances, axis=0)

            if labels is not None and (new_labels == labels).all():
                labels = new_labels
                changed = False
            else:
                labels = new_labels
                for label_idx in range(self.n_clusters):
                    self.centers[label_idx] = np.mean(source_data[labels == label_idx], axis=0)

        return labels

