import os

import matplotlib.pyplot as plt
import numpy as np

import constants
from data import Data
from model import LogisticRegressionModel


def _write_string_to_file(filename, content):
    with open(filename, 'w') as fout:
        fout.write(content)


def _draw_line(param_a, param_b, min_x, max_x):
    """根据两个端点的横坐标, 直线的一次函数，画出直线.

    f(x) = param_a * x + param_b.
    画出两个端点即可:(min_x, f(min_x)), (max_x, f(max_x))

    Args:
        param_a: float.
        param_b: float.
        min_x: float.
        max_x: float.
    """
    plt.plot((min_x, max_x), (param_a * min_x + param_b, param_a * max_x + param_b))


class Train(object):
    def __init__(self):
        self.data = Data()
        self.model = LogisticRegressionModel()

        self.print_data_list = list()
        self.loss_list = list()

    def train(self):
        """对于线性回归模型, 使用牛顿法进行训练."""

        is_stop = False
        pre_loss = 1.0
        bad_loss_sum = 0
        while not is_stop:
            is_stop, gradient1_norm = self.model.optimize(self.data.source_data,
                                                          self.data.target_data,
                                                          constants.STOP_VALUE)
            loss = self.model.loss(self.data.source_data, self.data.target_data)
            self.loss_list.append(loss)

            print_data = 'gradient1_norm: %.6f, loss: %.6f, weights: %s' % (gradient1_norm, loss, str(self.model.weights))
            self.print_data_list.append(print_data)
            print(print_data)

            if pre_loss <= loss:
                bad_loss_sum += 1
                if bad_loss_sum > 5:
                    print('bad_loss_sum > 5, early stop')
                    break
            pre_loss = loss

        output_dir = os.path.join(constants.OUTPUT_DIR,
                                  '%s_%s_%d_%d' % (constants.LABEL_0, constants.LABEL_1,
                                                   constants.FEATURE_0, constants.FEATURE_1))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        self._save_result_plot(output_dir)
        self._save_train_print_data(output_dir)
        self._save_loss_plot(output_dir)

    def _save_result_plot(self, output_dir):
        plt.xlabel(constants.FEATURES[constants.FEATURE_0])
        plt.ylabel(constants.FEATURES[constants.FEATURE_1])

        colors = np.array(self.data.source_data.shape[0] * ['b'])
        colors[self.data.target_data == 1] = 'r'
        plt.scatter(self.data.source_data[:, 0], self.data.source_data[:, 1], c=colors)

        # w1 * x + w2 * y + b = 0
        # y = -w1/w2 * x - b/w2
        _draw_line(-self.model.weights[0] / self.model.weights[1],
                   -self.model.weights[2] / self.model.weights[1],
                   self.data.source_min_bound[0],
                   self.data.source_max_bound[0])

        plt.savefig(os.path.join(output_dir, 'result.png'))

    def _save_loss_plot(self, output_dir):
        plt.clf()

        plt.xlabel('epoch')
        plt.ylabel('loss')
        data_x = range(1, 1 + len(self.loss_list))
        data_y = [item for item in self.loss_list]
        plt.plot(data_x, data_y)
        plt.savefig(os.path.join(output_dir, 'loss.png'))

    def _save_train_print_data(self, output_dir):
        _write_string_to_file(os.path.join(output_dir, 'result.txt'),
                              '\n'.join(self.print_data_list))


if __name__ == '__main__':
    Train().train()
