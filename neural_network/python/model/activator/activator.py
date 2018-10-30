import numpy as np

from abc import ABCMeta, abstractmethod


class Activator(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, linear_input: np.ndarray) -> np.ndarray:
        """Calculate output via linear_input.

        Args:
            linear_input: shape is (batch_size, output_size), calculated by input_data * weights.
        Return:
            shape is (batch_size, output_size).
        """
        pass

    @abstractmethod
    def backward(self, output: np.ndarray) -> np.ndarray:
        """Calculate gradient of output of activator relative to linear_input.

        Args:
            output: shape is (batch_size, output_size), the predicted value.
        Return:
            shape is (batch_size, output_size)
        """
        pass
