import numpy as np

from .activator import Activator


class FullyConnectedLayer(object):
    def __init__(self, input_size: int, output_size: int, batch_size: int, activator: Activator):
        """Init FullyConnectedLayer by size and activator.

        Args:
            input_size: the size of input vector
            output_size: the size of output vector
            batch_size: the size of data sets per batch
            activator: the activation function
        """
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.activator = activator

        # self.weights = np.random.uniform(-0.1, 0.1, (input_size, output_size))
        self.weights = np.zeros((input_size, output_size))
        self.bia = np.zeros((1, output_size))

        self.input = np.zeros((batch_size, input_size))
        self.output = np.zeros((batch_size, output_size))

        # shape: (batch_size, output_size)
        self.delta = None
        # shape: (input_size, output_size)
        self.weights_gradient = None
        # shape: (1, output_size)
        self.bia_gradient = None

    def forward(self, input_data: np.ndarray):
        """Calculate output_data via input_data.

        Store input_data and output_data in the instance.

        Args:
            input_data: shape is (batch_size, input_size).
        """
        self.input = input_data

        # shape: (batch_size, output_size)
        linear = np.matmul(self.input, self.weights) + self.bia
        self.output = self.activator.forward(linear)

    def backward(self, delta_data: np.ndarray):
        """Calculate weights_gradient and delta of this layer via delta_data of previous layer.

        Store weights_gradient and delta in the instance.

        Args:
            delta_data: shape is (batch_size, output_size).
        """
        self.delta = self.activator.backward(self.input) * np.matmul(delta_data, self.weights.T)
        self.weights_gradient = np.matmul(self.input.T, delta_data)
        self.bia_gradient = np.sum(delta_data, axis=0)

    def optimize(self, learning_rate: float):
        """Update weights according to weights_gradient.

        Args:
            learning_rate: the learning rate.
        """
        self.weights += learning_rate * self.weights_gradient
        self.bia += learning_rate * self.bia_gradient
