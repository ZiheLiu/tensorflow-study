import numpy as np

from model.activator import Activator


class SigmoidActivator(Activator):
    def forward(self, linear_input):
        return 1.0 / (1.0 + np.exp(-linear_input))

    def backward(self, output):
        return output * (1 - output)
