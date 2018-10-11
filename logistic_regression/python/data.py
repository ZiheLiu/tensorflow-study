import numpy as np

import constants
from utils.shell_args import SHELL_ARGS


class Data(object):
    def __init__(self):
        self.source_data, self.target_data = self.get_data()

        self.source_data = np.array(self.source_data)
        self.target_data = np.array(self.target_data)

        self.source_max_bound = np.max(self.source_data, axis=0)
        self.source_min_bound = np.min(self.source_data, axis=0)

    def _is_contained_labels(self, str_label):
        return str_label == constants.LABEL_0 or str_label == constants.LABEL_1

    def _get_numeric_label(self, str_label):
        return 0 if str_label == constants.LABEL_0 else 1

    def _get_features(self, columns):
        return float(columns[SHELL_ARGS.feature_0]), float(columns[SHELL_ARGS.feature_1])

    def get_data(self):
        """从csv文件中获取source_data和target_data.

        只保留LABEL_0和LABEL_1两个类别数据.
        只保留FEATURE_0和FEATURE_1两个特征.

        Return:
            source_data: array.
                shape: [-1, 2], dtype: float.
                The features for each row.
                eg. (('5.1', '1.4'), ('4.9', '1.4'))
            target_data: array.
                shape: [-1], dtype: float(1 or 0).
                The label for each row.
                eg. (1., 0.)
        """
        source_data = list()
        target_data = list()
        with open(constants.INPUT_IRIS_FILENAME, 'r') as fin:
            for line in fin.readlines():
                if line == '\n':
                    continue
                columns = line.replace('\n', '').split(',')
                if self._is_contained_labels(columns[4]):
                    source_data.append(self._get_features(columns))
                    target_data.append(self._get_numeric_label(columns[4]))
        return source_data, target_data
