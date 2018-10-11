import numpy as np

from utils.shell_args import SHELL_ARGS


class LogisticRegressionModel(object):
    def __init__(self):
        self.weights = np.array((0., 0., 0.))

    def _get_bias_mask_data(self, data):
        """输入二维向量，每一行加一列1.

        Parameters:
            data: array.
                shape: [n, m].
        Return:
            out: array.
                shape: [n, m+1].
        Examples:
            >>> self._get_bias_mask_data([[1, 2], [3, 4], [4, 5]])
            [[1, 2, 1], [3, 4, 1], [4, 5, 1]]
        """
        bias_mask = [1] * data.shape[0]
        return np.column_stack((data, bias_mask))

    def get_predicted_labels(self, source_data, bias_mask=True):
        """以矩阵的形式输入多条数据，输出每条数据的labels组成的矩阵.

        f(x) = 1 / (1 + e^(weights * x + bias)).

        Parameters:
            source_data: array.
                shape: [n, 2].
                每行是一条数据的两个特征值.
            bias_mask: bool.
                是否要对source_data处理，加上一列1.
        Return:
            out: array.
                shape: [n].
                每行是一条数据的label.
        """
        if bias_mask:
            source_data = self._get_bias_mask_data(source_data)

        linear = np.matmul(source_data, self.weights)
        return 1 / (1 + np.power(np.e, -linear))

    def void_zero(self, vector, min_value=1e-8):
        """把向量中小于1e-8的数字都置为1e-8"""
        vector[vector < min_value] = min_value
        return vector

    def loss(self, source_data, target_data):
        """根据源数据和目标数据, 求得损失值.

        loss = sum(yi * ln(f(xi)) + (1 - yi) * ln(1 - f(xi))) 0<= i < n.

        Parameters:
            source_data: array.
                shape: [n, 2].
                每行是一条数据的两个特征值.
            target_data: array.
                shape: [n].
                每个元素是label值, 0或1.
        Return:
            out: float.
        """
        source_data = self._get_bias_mask_data(source_data)
        predicted_data = self.get_predicted_labels(source_data, bias_mask=False)
        matrix = target_data * np.log(self.void_zero(predicted_data)) \
                 + (1 - target_data) * np.log(self.void_zero((1 - predicted_data)))
        return -np.sum(matrix) / source_data.shape[0]

    def optimize(self, source_data, target_data, stop_value):
        """根据源数据和目标数据, 利用优化算法, 对参数weights进行更新.


        使用牛顿法对参数进行优化:
            weights = weights - gradient2的逆矩阵 * gradient1.
            gradient1 = sum(xi * (yi - f(xi))) 0<= i < n.
            gradient2 = sum(xi * xi^T * f(xi) * (1 - f(xi))) 0<= i < n.
        使用梯度下降算法进行优化:
            weights = weights - learning_rate * gradient1.

        Parameters:
            source_data: array.
                shape: [n, 2].
                每行是一条数据的两个特征值.
            target_data: array.
                shape: [n].
                每个元素是label值, 0或1.
            stop_value: float.
                如果梯度的二阶范数小于stop_value, 则不更新weights, 否则更新weights.

        Return:
            is_stop: bool.
                是否应该停止训练, 即梯度的二阶范数是否小于stop_value.
            gradient1_norm: float.
                梯度的二阶范数.
        """
        source_data = self._get_bias_mask_data(source_data)
        predicted_data = self.get_predicted_labels(source_data, bias_mask=False)

        # [n, 3]
        # sum(xi * (yi - f(xi))) 0<= i < n
        gradient1_matrix = source_data * np.reshape((target_data - predicted_data), (-1, 1))
        # [3]
        gradient1 = np.sum(gradient1_matrix, axis=0)

        gradient1_norm = np.linalg.norm(gradient1)
        if gradient1_norm < stop_value:
            return True, gradient1_norm

        if SHELL_ARGS.optimizer == 'gradient_descent':
            delta = gradient1 * 0.01
        else:
            # [n, 3, 3]
            # sum(xi * xi^T * f(xi) * (1 - f(xi))) 0<= i < n
            source_data_sum = np.matmul(np.reshape(source_data, (-1, 3, 1)), np.reshape(source_data, (-1, 1, 3)))
            # [n, 3, 3]
            gradient2_matrix = source_data_sum * (np.reshape(predicted_data, (-1, 1, 1))) \
                               * np.reshape((1 - predicted_data), (-1, 1, 1))
            # [3, 3]
            gradient2 = np.sum(gradient2_matrix, axis=0)

            # [3]
            delta = np.matmul(np.linalg.inv(gradient2), gradient1)

        self.weights += delta
        return False, gradient1_norm
