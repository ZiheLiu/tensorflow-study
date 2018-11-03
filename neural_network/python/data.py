import numpy as np
import os

import constants
from utils import file_utils


class Data(object):
    def __init__(self, data_prefix):
        self.data_prefix = data_prefix
        self.name = os.path.split(data_prefix)[-1]

        self.target_name2id = self._read_target_vocab()

        self.train_source_data, self.train_target_data, \
            self.eval_source_data, self.eval_target_data, \
            self.test_source_data, self.test_target_data = self._read_data_set()

    def input_size(self):
        return self.train_source_data.shape[1]

    def output_size(self):
        return self.train_target_data.shape[1]

    def get_batches(self, source_data, target_data, batch_size):
        for batch_i in range(source_data.shape[0] // batch_size):
            start_id = batch_i * batch_size
            end_id = start_id + batch_size
            yield source_data[start_id: end_id], target_data[start_id: end_id]

    def reshuffle_train_data(self):
        index_list = np.arange(self.train_source_data.shape[0])
        np.random.shuffle(index_list)
        self.train_source_data = self.train_source_data[index_list]
        self.train_target_data = self.train_target_data[index_list]

    def train_batches_sum(self, batch_size):
        return self.train_source_data.shape[0] // batch_size

    def eval_batches_sum(self, batch_size):
        return self.eval_source_data.shape[0] // batch_size

    def test_batches_sum(self, batch_size):
        return self.test_source_data.shape[0] // batch_size

    def train_batches(self, batch_size):
        return self.get_batches(self.train_source_data, self.train_target_data, batch_size)

    def eval_batches(self, batch_size):
        return self.get_batches(self.eval_source_data, self.eval_target_data, batch_size)

    def test_batches(self, batch_size):
        return self.get_batches(self.test_source_data, self.test_target_data, batch_size)

    def _read_target_vocab(self):
        filename = "%s%s" % (self.data_prefix, constants.SUFFIX_TARGET_VOCAB)
        return file_utils.read_json_file_to_dict(filename)

    def _read_source_data(self, filename):
        source_data_path = os.path.join(constants.ROOT_DIR, filename)
        source_data = np.loadtxt(source_data_path, dtype=np.float, delimiter=',')
        return source_data

    def _read_target_data(self, filename):
        target_data_path = os.path.join(constants.ROOT_DIR, filename)
        raw_target_data = np.loadtxt(target_data_path, dtype=np.str)
        # map name to id
        target_data = np.vectorize(self.target_name2id.__getitem__)(raw_target_data)
        # one hot
        target_data = np.eye(len(self.target_name2id))[target_data]
        return target_data

    def _read_data_set(self):
        train_source_data = self._read_source_data('%s%s' % (self.data_prefix, constants.SUFFIX_TRAIN_SOURCE_DATA))
        train_target_data = self._read_target_data('%s%s' % (self.data_prefix, constants.SUFFIX_TRAIN_TARGET_DATA))

        eval_source_data = self._read_source_data('%s%s' % (self.data_prefix, constants.SUFFIX_EVAL_SOURCE_DATA))
        eval_target_data = self._read_target_data('%s%s' % (self.data_prefix, constants.SUFFIX_EVAL_TARGET_DATA))

        test_source_data = self._read_source_data('%s%s' % (self.data_prefix, constants.SUFFIX_TEST_SOURCE_DATA))
        test_target_data = self._read_target_data('%s%s' % (self.data_prefix, constants.SUFFIX_TEST_TARGET_DATA))

        return train_source_data, train_target_data, \
               eval_source_data, eval_target_data, \
               test_source_data, test_target_data
