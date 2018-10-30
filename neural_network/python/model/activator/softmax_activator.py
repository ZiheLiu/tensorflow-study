import numpy as np

from model.activator import Activator


class SoftmaxActivator(Activator):
    def forward(self, linear_input):
        exp_input = np.exp(linear_input)
        exp_sum = np.sum(exp_input, axis=1, keepdims=True)
        return exp_input / exp_sum

    def backward(self, output):
        """Calculate gradient of output of activator relative to linear_input.

        See also Activator.backward.
        Because gradient of output of cost function to linear_input is (output-target),
        when using cross entropy with softmax as cost function,
        return vector fill with 1 here.
        """
        return np.full(output.shape, 1)
