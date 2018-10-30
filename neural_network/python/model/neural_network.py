import numpy as np

from .fully_connected_layer import FullyConnectedLayer
from .activator import SigmoidActivator, SoftmaxActivator


class NeuralNetwork(object):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, batch_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.layers = list()
        self._build_model()

    def _build_model(self):
        self.layers.append(FullyConnectedLayer(input_size=self.input_size,
                                               output_size=self.hidden_size,
                                               batch_size=self.batch_size,
                                               activator=SigmoidActivator()))
        self.layers.append(FullyConnectedLayer(input_size=self.hidden_size,
                                               output_size=self.output_size,
                                               batch_size=self.batch_size,
                                               activator=SoftmaxActivator()))

    def _calc_cross_entropy_loss(self, target_batch: np.ndarray, output_batch: np.ndarray) -> float:
        """Calculate cross entropy loss loss according target_batch and output_batch.

        Args:
            target_batch: shape is (batch_size, output_size).
            output_batch: shape is (batch_size, output_size).
        Return:
            the average cross entropy.
        """
        return float(-np.mean(target_batch * np.log(output_batch)))

    def loss(self, source_batch: np.ndarray, target_batch: np.ndarray):
        output_batch = self.get_logistics(source_batch)
        loss_value = self._calc_cross_entropy_loss(target_batch, output_batch)
        return loss_value

    def train(self, source_batch: np.ndarray, target_batch: np.ndarray, learning_rate: float) -> float:
        output_batch = self.get_logistics(source_batch)
        loss_value = self._calc_cross_entropy_loss(target_batch, output_batch)
        self.optimize(target_batch, learning_rate)
        return loss_value

    def predict(self, source_batch: np.ndarray):
        output_batch = self.get_logistics(source_batch)
        mask_batch = np.max(output_batch, axis=1, keepdims=True) == output_batch
        return mask_batch.astype(float)

    def get_logistics(self, source_batch: np.ndarray):
        output = source_batch
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def optimize(self, target_batch: np.ndarray, learning_rate: float):
        last_layer = self.layers[-1]
        delta = last_layer.activator.backward(last_layer.output) * (target_batch - last_layer.output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            layer.optimize(learning_rate)
            delta = layer.delta
        return delta
