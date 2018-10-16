import numpy as np

import constants
from utils.shell_args import SHELL_ARGS


class LogisticRegressionClassificationModel(object):
    def __init__(self):
        self.weights = np.array(((0.,), (0.,), (0.,)))

    def _get_bias_mask_data(self, data):
        """Input 2-D vector, output 2-D vector inserted a column of 1.

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
        """Input source_data with vector, output labels with vector.

        f(x) = 1 / (1 + e^(weights * x + bias)).

        Parameters:
            source_data: ndarray.
                shape: [n, 2].
                每行是一条数据的两个特征值.
            bias_mask: bool.
                是否要对source_data处理，加上一列1.
        Return:
            out: ndarray.
                shape: [n, 1].
                每行是一条数据的label.
        """
        if bias_mask:
            source_data = self._get_bias_mask_data(source_data)

        linear = np.matmul(source_data, self.weights)
        return 1 / (1 + np.exp(-linear))

    def void_zero(self, vector, min_value=1e-8):
        """Update numbers in vector to 1e-8 which are lower than 1e-8."""
        vector[vector < min_value] = min_value
        return vector

    def loss(self, source_data, target_data):
        """According with source data and target data, compute loss.

        loss = sum(yi * ln(f(xi)) + (1 - yi) * ln(1 - f(xi))) 0<= i < n.

        Parameters:
            source_data: array.
                shape: [n, 2].
                Each row contains 2 features.
            target_data: array.
                shape: [n, 1].
                Each row contains 1 label, 0 or 1.
        Return:
            out: float.
        """
        source_data = self._get_bias_mask_data(source_data)
        predicted_data = self.get_predicted_labels(source_data, bias_mask=False)
        matrix = target_data * np.log(self.void_zero(predicted_data)) \
                 + (1 - target_data) * np.log(self.void_zero((1 - predicted_data)))
        return -np.sum(matrix) / source_data.shape[0]

    def accuracy(self, source_data, target_data):
        predictions = self.get_predicted_labels(source_data)
        condition = predictions > 0.5
        predictions[condition] = 1.0
        predictions[~condition] = 0.0
        return (predictions == target_data).mean()

    def optimize(self, source_data, target_data, stop_value):
        """According with source data and target data, update parameters weights with optimize algorithm.


        Using Newton Method to optimize parameters:
            weights = weights - gradient2's inv * gradient1.
            gradient1 = sum(xi * (yi - f(xi))) 0<= i < n.
            gradient2 = sum(xi * xi^T * f(xi) * (1 - f(xi))) 0<= i < n.
        Using Gradient Descent to optimize parameters:
            weights = weights - learning_rate * gradient1.

        Parameters:
            source_data: ndarray.
                shape: [n, 2].
                Each row contains 2 features.
            target_data: ndarray.
                shape: [n, 1].
                Each row contains 1 label, 0 or 1.
            stop_value: float.
                If ||gradient|| < stop_value, don't update weights.

        Return:
            is_stop: bool.
                Whether stopping train, that is ||gradient|| < stop_value.
            gradient1_norm: float.
                ||gradient||.
        """
        source_data = self._get_bias_mask_data(source_data)
        predicted_data = self.get_predicted_labels(source_data, bias_mask=False)

        # [3, 1]
        # sum(xi * (yi - f(xi))) 0<= i < n
        gradient1 = np.matmul(np.transpose(source_data), target_data - predicted_data)

        gradient1_norm = np.linalg.norm(gradient1)
        if gradient1_norm < stop_value:
            return True, gradient1_norm

        print(SHELL_ARGS.optimizer)
        if SHELL_ARGS.optimizer == 'gradient_descent':
            delta = constants.LEARNING_RATE * gradient1
        else:
            # [3, 3]
            # sum(xi * xi^T * f(xi) * (1 - f(xi))) 0<= i < n
            predicted_data_diag = np.diag(np.reshape(predicted_data * (1 - predicted_data), (-1,)))
            gradient2 = np.matmul(np.matmul(np.transpose(source_data), predicted_data_diag), source_data)

            # [3, 1]
            delta = np.matmul(np.linalg.inv(gradient2), gradient1)

        self.weights += delta
        return False, gradient1_norm
