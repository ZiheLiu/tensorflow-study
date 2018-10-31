import numpy as np

from model.activator import Activator

DELTA = 1e-5


class ReluActivator(Activator):
    def forward(self, linear_input):
        mask = linear_input < 0
        linear_input[mask] *= 1e-5
        return linear_input

    def backward(self, output):
        output = np.ones(output.shape)
        mask = output < 0
        output[mask] = DELTA
        return output
