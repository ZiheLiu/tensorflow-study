import os

import numpy as np
import matplotlib.pyplot as plot

import constants
from data import Data
from model import NeuralNetwork
from utils import file_utils
from utils.log_utils import LOGGER
from utils.shell_args import SHELL_ARGS


class Train(object):
    def __init__(self, hidden_sizes, batch_size, learning_rate, activator):
        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.activator = activator

        self.data = Data(SHELL_ARGS.prefix)
        self.model = NeuralNetwork(self.data.input_size(),
                                   self.hidden_sizes,
                                   self.data.output_size(),
                                   self.batch_size,
                                   self.activator)

    def _calc_accurate(self, batches_func, batches_sum):
        accurate = 0
        for test_source_batch, test_target_batch in batches_func(self.batch_size):
            predictions = self.model.predict(test_source_batch)
            mask = predictions == test_target_batch
            accurate += np.mean(mask)
        accurate /= batches_sum
        return accurate

    def _save_loss_plot(self, output_dir, train_loss_list, eval_loss_list):
        plot.clf()

        plot.xlabel('epoch')
        plot.ylabel('loss')

        data_x = range(1, 1 + len(train_loss_list))
        plot.plot(data_x, train_loss_list, color='blue', label='train_loss')
        plot.plot(data_x, eval_loss_list, color='red', label='eval_loss')

        plot.legend()

        plot.savefig(os.path.join(output_dir, 'loss.png'))

    def train_epoch(self):
        train_loss_value = 0
        for train_source_batch, train_target_batch in self.data.train_batches(self.batch_size):
            loss_value = self.model.train(train_source_batch, train_target_batch, self.learning_rate)
            train_loss_value += loss_value
        train_loss_value /= self.data.train_batches_sum(self.batch_size)
        return train_loss_value

    def eval_epoch(self):
        eval_loss_value = 0
        for eval_source_batch, eval_target_batch in self.data.eval_batches(self.batch_size):
            loss_value = self.model.loss(eval_source_batch, eval_target_batch)
            eval_loss_value += loss_value
        eval_loss_value /= self.data.eval_batches_sum(self.batch_size)
        return eval_loss_value

    def train(self):
        LOGGER.info('activator: %s, batch_size: %d, hidden_sizes: %s, learning: %f' %
                    (self.activator, self.batch_size, str(self.hidden_sizes), self.learning_rate))

        train_batches_sum = self.data.train_batches_sum(self.batch_size)
        eval_batches_sum = self.data.eval_batches_sum(self.batch_size)
        test_batches_sum = self.data.test_batches_sum(self.batch_size)

        train_loss_list = list()
        eval_loss_list = list()

        pre_train_loss_value = 1000
        pre_eval_loss_value = 1000
        increase_times = 0
        for epoch_i in range(constants.EPOCHS):
            # self.data.reshuffle_train_data()
            cur_train_loss_value = self.train_epoch()
            cur_eval_loss_value = self.eval_epoch()

            LOGGER.info('Epoch_i: %d, train loss: %.6f, eval loss: %.6f' %
                            (epoch_i, cur_train_loss_value, cur_eval_loss_value))

            if cur_train_loss_value > pre_train_loss_value or cur_eval_loss_value > pre_eval_loss_value:
                increase_times += 1
                if increase_times >= 5:
                    LOGGER.info('Early stop, increase_times >= 5')
                    break
            pre_train_loss_value = cur_train_loss_value
            pre_eval_loss_value = cur_eval_loss_value
            train_loss_list.append(cur_train_loss_value)
            eval_loss_list.append(cur_eval_loss_value)

        output_dir = os.path.join(constants.OUTPUT_DIR, self.data.name)
        file_utils.safe_mkdirs(output_dir)
        self._save_loss_plot(output_dir, train_loss_list, eval_loss_list)

        train_accurate = self._calc_accurate(self.data.train_batches, train_batches_sum)
        eval_accurate = self._calc_accurate(self.data.eval_batches, eval_batches_sum)
        test_accurate = self._calc_accurate(self.data.test_batches, test_batches_sum)

        LOGGER.info('train_accurate: %.6f, eval_accurate: %.6f, test_accurate: %.6f' % (train_accurate, eval_accurate, test_accurate))


if __name__ == '__main__':
    Train(hidden_sizes=SHELL_ARGS.hidden_sizes,
          batch_size=SHELL_ARGS.batch_size,
          learning_rate=SHELL_ARGS.learning_rate,
          activator=SHELL_ARGS.activator).train()
