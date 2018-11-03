import numpy as np

from .fully_connected_layer import FullyConnectedLayer
from .activator import SigmoidActivator, SoftmaxActivator, TanhActivator


class NeuralNetwork(object):
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, batch_size: int, activator: str):
        """Use hyper parameters to build the Neural Network model.

        Args:
            input_size: size of input layer.
            hidden_sizes: sizes of each hidden layers.
            output_size: size of output layer.
            batch_size: size of batch.
            activator: name of activator.
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.batch_size = batch_size

        self.activator = self._get_activator(activator)

        self.layers = list()
        self._build_model()

    def _get_activator(self, activator: str):
        """Get activator class of hidden layer according to name of activator.

        Args:
            activator: the name of activator, including 'sigmoid' and 'tanh'.
        Return:
            Sub class of Activator.
        """
        if activator == 'sigmoid':
            return SigmoidActivator
        elif activator == 'tanh':
            return TanhActivator
        else:
            raise ValueError('Argument activator is invalid: %s' % activator)

    def _build_model(self):
        """Build the structure of model.

        There are one input layer, one output layer, and multiple hidden layers.
        """
        self.layers.append(FullyConnectedLayer(input_size=self.input_size,
                                               output_size=self.hidden_sizes[0],
                                               batch_size=self.batch_size,
                                               activator=TanhActivator()))

        for i in range(0, len(self.hidden_sizes) - 1):
            self.layers.append(FullyConnectedLayer(input_size=self.hidden_sizes[i],
                                                   output_size=self.hidden_sizes[i + 1],
                                                   batch_size=self.batch_size,
                                                   activator=TanhActivator()))

        self.layers.append(FullyConnectedLayer(input_size=self.hidden_sizes[-1],
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
        loss_value = target_batch * np.log(output_batch)
        loss_value = np.sum(loss_value) / target_batch.shape[0]
        return float(-loss_value)

    def loss(self, source_batch: np.ndarray, target_batch: np.ndarray) -> float:
        """Calculate the loss value according to source batch data and target batch data.

        Args:
            source_batch: shape is (batch_size, input_size). Each row has input_size features.
            target_batch: shape is (batch_size, output_size). Each row has output_size probabilities of each category.

        Return:
            The average cross entropy loss.
        """
        output_batch = self.forward(source_batch)
        loss_value = self._calc_cross_entropy_loss(target_batch, output_batch)
        return loss_value

    def train(self, source_batch: np.ndarray, target_batch: np.ndarray, learning_rate: float) -> float:
        """Optimize weights and bias according to source and target batch data, and calculate the loss of it.

        Args:
            source_batch: shape is (batch_size, input_size).
            target_batch: shape is (batch_size, input_size).
            learning_rate: the learning rate.

        Return:
            The The average cross entropy loss of training data set.
        """
        output_batch = self.forward(source_batch)
        loss_value = self._calc_cross_entropy_loss(target_batch, output_batch)
        self.backward_optimize(target_batch, learning_rate)
        return loss_value

    def predict(self, source_batch: np.ndarray) -> np.ndarray:
        """Get the predicted probabilities distribution by source batch data.

        Args:
            source_batch: shape is (batch_size, input_size).

        Return:
            shape is (batch_size, output_size).
            Each row is a one hot vector, And the index position of the value one is the predicted category.
        """
        output_batch = self.forward(source_batch)
        mask_batch = np.max(output_batch, axis=1, keepdims=True) == output_batch
        return mask_batch.astype(float)

    def forward(self, source_batch: np.ndarray) -> np.ndarray:
        """Calculate forward output of each layer from the input layer.

        Args:
            source_batch: shape is (batch_size, input_size)

        Return:
            shape is (batch_size, output_size).
            The output vector of output layer.
        """
        output = source_batch
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def backward_optimize(self, target_batch: np.ndarray, learning_rate: float) ->np.ndarray:
        """Calculate backward gradient delta of each layer from the output layer.

        Args:
            target_batch: shape is (batch_size, output_size).
            learning_rate: learning rate.

        Return:
            shape is (batch_size, input_size).
        """
        last_layer = self.layers[-1]
        delta = last_layer.activator.backward(last_layer.output) * (target_batch - last_layer.output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            layer.optimize(learning_rate)
            delta = layer.delta
        return delta
