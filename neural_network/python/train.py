import numpy as np

import constants
from data import Data
from model import NeuralNetwork
from utils.shell_args import SHELL_ARGS


class Train(object):
    def __init__(self):
        self.data = Data(SHELL_ARGS.prefix)
        self.model = NeuralNetwork(self.data.input_size(),
                                   constants.HIDDEN_SIZE,
                                   self.data.output_size(),
                                   constants.BATCH_SIZE)

    def _calc_accurate(self, batches_func, batches_sum):
        accurate = 0
        for test_source_batch, test_target_batch in batches_func(constants.BATCH_SIZE):
            predictions = self.model.predict(test_source_batch)
            mask = predictions == test_target_batch
            accurate += np.mean(mask)
        accurate /= batches_sum
        return accurate

    def train(self):
        pre_train_loss_value = 1000
        pre_eval_loss_value = 1000
        increase_times = 0
        for epoch_i in range(constants.EPOCHS):
            # self.data.reshuffle_train_data()
            cur_train_loss_value = 0
            cur_eval_loss_value = 0

            for train_source_batch, train_target_batch in self.data.train_batches(constants.BATCH_SIZE):
                loss_value = self.model.train(train_source_batch, train_target_batch, constants.LEARNING_RATE)
                cur_train_loss_value += loss_value

            for eval_source_batch, eval_target_batch in self.data.eval_batches(constants.BATCH_SIZE):
                loss_value = self.model.loss(eval_source_batch, eval_target_batch)
                cur_eval_loss_value += loss_value

            cur_train_loss_value /= self.data.train_batches_sum(constants.BATCH_SIZE)
            cur_eval_loss_value /= self.data.eval_batches_sum(constants.BATCH_SIZE)
            print(
                'Epoch_i: %d, train loss: %.6f, eval loss: %.6f' % (epoch_i, cur_train_loss_value, cur_eval_loss_value))

            if cur_train_loss_value > pre_train_loss_value or cur_eval_loss_value > pre_eval_loss_value:
                increase_times += 1
                if increase_times >= 5:
                    print('Early stop, increase_times >= 10')
                    break
            pre_train_loss_value = cur_train_loss_value
            pre_eval_loss_value = cur_eval_loss_value

        train_accurate = self._calc_accurate(self.data.train_batches, self.data.train_batches_sum(constants.BATCH_SIZE))
        eval_accurate = self._calc_accurate(self.data.eval_batches, self.data.eval_batches_sum(constants.BATCH_SIZE))
        test_accurate = self._calc_accurate(self.data.test_batches, self.data.test_batches_sum(constants.BATCH_SIZE))

        print('train_accurate: %.6f, eval_accurate: %.6f, test_accurate: %.6f' % (train_accurate, eval_accurate, test_accurate))


if __name__ == '__main__':
    Train().train()
