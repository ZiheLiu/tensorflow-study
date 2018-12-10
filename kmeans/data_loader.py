import numpy as np

import os

import constants
from utils.shell_args import SHELL_ARGS


class DataLoader(object):
    def __init__(self, prefix):
        self.prefix = prefix
        self.source_data = self._read_source_data()
        self.target_data = self._read_target_data()
        self.n_clusters = self._get_n_clusters()

    def _normalize(self, data: np.ndarray):
        min_bound = data.min(axis=0)
        max_bound = data.max(axis=0)
        return (data - min_bound) / (max_bound - min_bound)

    def _read_source_data(self):
        source_data_path = os.path.join(constants.INPUT_DIR,
                                        self.prefix,
                                        '{}.{}'.format(self.prefix, constants.SUFFIX_SOURCE_DATA))
        source_data = np.loadtxt(source_data_path, dtype=np.float, delimiter=',')
        return self._normalize(np.array(source_data))

    def _read_target_data(self):
        target_data_path = os.path.join(constants.INPUT_DIR,
                                        self.prefix,
                                        '{}.{}'.format(self.prefix, constants.SUFFIX_TARGET_DATA))
        target_data = np.loadtxt(target_data_path, dtype=np.float, delimiter=',')
        return np.array(target_data)

    def _get_n_clusters(self):
        label_set = {label: True for label in self.target_data}
        return len(label_set)


if __name__ == '__main__':
    data_loader = DataLoader(SHELL_ARGS.prefix)
    print(data_loader.n_clusters)
