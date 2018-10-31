import numpy as np

import constants
from data import Data
from model import NeuralNetwork
from utils.shell_args import SHELL_ARGS


class Train(object):
    def __init__(self, hidden_size, batch_size, learning_rate):
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.data = Data(SHELL_ARGS.prefix)
        self.model = NeuralNetwork(self.data.input_size(),
                                   self.hidden_size,
                                   self.data.output_size(),
                                   self.batch_size)

    def _calc_accurate(self, batches_func, batches_sum):
        accurate = 0
        for test_source_batch, test_target_batch in batches_func(self.batch_size):
            predictions = self.model.predict(test_source_batch)
            mask = predictions == test_target_batch
            accurate += np.mean(mask)
        accurate /= batches_sum
        return accurate

    def train(self):
        train_batches_sum = self.data.train_batches_sum(self.batch_size)
        eval_batches_sum = self.data.eval_batches_sum(self.batch_size)
        test_batches_sum = self.data.test_batches_sum(self.batch_size)

        pre_train_loss_value = 1000
        pre_eval_loss_value = 1000
        increase_times = 0
        for epoch_i in range(constants.EPOCHS):
            # self.data.reshuffle_train_data()
            cur_train_loss_value = 0
            for train_source_batch, train_target_batch in self.data.train_batches(self.batch_size):
                loss_value = self.model.train(train_source_batch, train_target_batch, self.learning_rate)
                cur_train_loss_value += loss_value
            cur_train_loss_value /= train_batches_sum

            cur_eval_loss_value = 0
            for eval_source_batch, eval_target_batch in self.data.eval_batches(self.batch_size):
                loss_value = self.model.loss(eval_source_batch, eval_target_batch)
                cur_eval_loss_value += loss_value
            cur_eval_loss_value /= eval_batches_sum

            print('Epoch_i: %d, train loss: %.6f, eval loss: %.6f' %
                  (epoch_i, cur_train_loss_value, cur_eval_loss_value))

            if cur_train_loss_value > pre_train_loss_value or cur_eval_loss_value > pre_eval_loss_value:
                increase_times += 1
                if increase_times >= 5:
                    print('Early stop, increase_times >= 5')
                    break
            pre_train_loss_value = cur_train_loss_value
            pre_eval_loss_value = cur_eval_loss_value

        train_accurate = self._calc_accurate(self.data.train_batches, train_batches_sum)
        eval_accurate = self._calc_accurate(self.data.eval_batches, eval_batches_sum)
        test_accurate = self._calc_accurate(self.data.test_batches, test_batches_sum)

        print('train_accurate: %.6f, eval_accurate: %.6f, test_accurate: %.6f' % (train_accurate, eval_accurate, test_accurate))


if __name__ == '__main__':
    Train(hidden_size=constants.HIDDEN_SIZE,
          batch_size=constants.BATCH_SIZE,
          learning_rate=constants.LEARNING_RATE).train()
