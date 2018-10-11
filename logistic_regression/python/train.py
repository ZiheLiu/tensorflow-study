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


def train():
    """读取数据, 对于线性回归模型, 使用牛顿法进行训练."""
    data = Data()
    model = LogisticRegressionModel()

    is_stop = False
    gradient1_norm = 0
    loss = 0
    while not is_stop:
        is_stop, gradient1_norm = model.train(data.source_data, data.target_data, constants.STOP_VALUE)
        loss = model.loss(data.source_data, data.target_data)
        print('gradient1_norm: %.6f, loss: %.6f' % (gradient1_norm, loss))

    output_dir = os.path.join(constants.OUTPUT_DIR,
                              '%s_%s_%d_%d' % (constants.LABEL_0, constants.LABEL_1,
                                               constants.FEATURE_0, constants.FEATURE_1))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    _write_string_to_file(os.path.join(output_dir, 'result.txt'), 'gradient1_norm: %f, loss: %f' % (gradient1_norm, loss))

    colors = np.array(data.source_data.shape[0] * ['b'])
    colors[data.target_data == 1] = 'r'
    plt.scatter(data.source_data[:, 0], data.source_data[:, 1], c=colors)

    # w1 * x + w2 * y + b = 0
    # y = -w1/w2 * x - b/w2
    _draw_line(-model.weights[0] / model.weights[1],
               -model.weights[2] / model.weights[1],
               data.source_min_bound[0],
               data.source_max_bound[0])

    plt.savefig(os.path.join(output_dir, 'result.png'))


if __name__ == '__main__':
    train()
