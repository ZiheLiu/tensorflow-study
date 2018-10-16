import os

import matplotlib.pyplot as plt

import constants
from data import Data
from model import LogisticRegressionClassificationModel
from utils.shell_args import SHELL_ARGS


def _write_string_to_file(filename, content):
    with open(filename, 'w') as fout:
        fout.write(content)


def _draw_line(param_a, param_b, min_x, max_x):
    """According with x-axis coordinate of 2 end points and parameters linear function,
    draw the line.

    f(x) = param_a * x + param_b.
    Only need to draw the 2 end points: (min_x, f(min_x)), (max_x, f(max_x)).

    Args:
        param_a: float.
        param_b: float.
        min_x: float.
        max_x: float.
    """
    plt.plot((min_x, max_x), (param_a * min_x + param_b, param_a * max_x + param_b), label='decision')


class Train(object):
    def __init__(self):
        self.data = Data(SHELL_ARGS.data_type)
        self.model = LogisticRegressionClassificationModel()

        self.print_data_list = list()
        self.loss_list = list()

    def train(self):
        """Train LogisticRegressionClassificationModel with Newton Method or Gradient Descent."""

        epoch_i = 1
        is_stop = False
        pre_loss = 1.0
        bad_loss_sum = 0
        while not is_stop:
            is_stop, gradient1_norm = self.model.optimize(self.data.source_data,
                                                          self.data.target_data,
                                                          constants.STOP_VALUE)
            loss = self.model.loss(self.data.source_data, self.data.target_data)
            self.loss_list.append(loss)
            accuracy = self.model.accuracy(self.data.source_data, self.data.target_data)

            # print data
            print_data = 'epoch: %d, gradient1_norm: %.6f, loss: %.6f, accuracy: %.6f, weights:\n%s' % \
                         (epoch_i, gradient1_norm, loss, accuracy, str(self.model.weights))
            self.print_data_list.append(print_data)
            print(print_data)

            # early stop check
            # Early stop when loss increases 5 times with Newton Method.
            if SHELL_ARGS.optimizer == 'newton' and pre_loss <= loss:
                bad_loss_sum += 1
                if bad_loss_sum > 5:
                    print('bad_loss_sum > 5, early stop')
                    break

            # early stop check
            # Only iterate 50 times with Gradient Descent.
            if SHELL_ARGS.optimizer == 'gradient_descent' and epoch_i >= 100:
                break

            pre_loss = loss
            epoch_i += 1

        output_dir = os.path.join(constants.OUTPUT_DIR, SHELL_ARGS.data_type, SHELL_ARGS.optimizer,
                                  '%s_%s_%d_%d' % (SHELL_ARGS.label_0, SHELL_ARGS.label_1,
                                                   SHELL_ARGS.feature_0, SHELL_ARGS.feature_1))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        self._save_result_plot(output_dir)
        self._save_train_print_data(output_dir)
        self._save_loss_plot(output_dir)

    def _save_result_plot(self, output_dir):
        plt.xlabel(self.data.feature_name_0())
        plt.ylabel(self.data.feature_name_1())

        source_data_label_0 = self.data.source_data_label_0()
        plt.scatter(source_data_label_0[:, 0],
                    source_data_label_0[:, 1],
                    label=self.data.label_0(),
                    c='blue',
                    alpha=0.5)
        source_data_label_1 = self.data.source_data_label_1()
        plt.scatter(source_data_label_1[:, 0],
                    source_data_label_1[:, 1],
                    label=self.data.label_1(),
                    c='red',
                    alpha=0.5)

        # w1 * x + w2 * y + b = 0
        # y = -w1/w2 * x - b/w2
        _draw_line(-self.model.weights[0] / self.model.weights[1],
                   -self.model.weights[2] / self.model.weights[1],
                   self.data.source_min_bound[0],
                   self.data.source_max_bound[0])

        plt.legend()

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
