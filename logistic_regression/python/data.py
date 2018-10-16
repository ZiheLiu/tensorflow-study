import numpy as np

import constants
from utils.shell_args import SHELL_ARGS


class Data(object):
    def __init__(self, data_type):
        self.data_type = data_type

        if data_type == 'iris':
            self.features = constants.IRIS_FEATURES
            self.source_data, self.target_data = self.get_data(constants.INPUT_IRIS_FILENAME, 4)
        elif data_type == 'wine':
            self.features = constants.WINE_FEATURES
            self.source_data, self.target_data = self.get_data(constants.INPUT_WINE_FILENAME, 0)
        else:
            raise ValueError('--data_type must be <iris> or <wine>, but: %s' % data_type)

        self.source_data = np.array(self.source_data)
        self.target_data = np.array(self.target_data)

        self.source_max_bound = np.max(self.source_data, axis=0)
        self.source_min_bound = np.min(self.source_data, axis=0)

    def _is_contained_labels(self, str_label):
        return str_label == SHELL_ARGS.label_0 or str_label == SHELL_ARGS.label_1

    def _get_numeric_label(self, str_label):
        return 0 if str_label == SHELL_ARGS.label_0 else 1

    def _get_features(self, columns):
        return float(columns[SHELL_ARGS.feature_0]), float(columns[SHELL_ARGS.feature_1])

    def label_0(self):
        return SHELL_ARGS.label_0

    def label_1(self):
        return SHELL_ARGS.label_1

    def feature_name_0(self):
        return self.features[SHELL_ARGS.feature_0]

    def feature_name_1(self):
        return self.features[SHELL_ARGS.feature_1]

    def source_data_label_0(self):
        return self.source_data[np.reshape(self.target_data, -1) == 0]

    def source_data_label_1(self):
        return self.source_data[np.reshape(self.target_data, -1) == 1]

    def get_data(self, filename, label_index):
        """从csv文件中获取source_data和target_data.

        只保留LABEL_0和LABEL_1两个类别数据.
        只保留FEATURE_0和FEATURE_1两个特征.

        Return:
            source_data: array.
                shape: [-1, 2], dtype: float.
                The features for each row.
                eg. (('5.1', '1.4'), ('4.9', '1.4'))
            target_data: array.
                shape: [-1, 1], dtype: float(1 or 0).
                The label for each row.
                eg. ((1.,), (0.,))
        """
        source_data = list()
        target_data = list()
        with open(filename, 'r') as fin:
            for line in fin.readlines():
                if line == '\n':
                    continue
                columns = line.replace('\n', '').split(',')
                if self._is_contained_labels(columns[label_index]):
                    source_data.append(self._get_features(columns))
                    target_data.append([self._get_numeric_label(columns[label_index])])
        return source_data, target_data
