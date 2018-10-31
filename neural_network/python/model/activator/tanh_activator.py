import numpy as np

from model.activator import Activator


class TanhActivator(Activator):
    def forward(self, linear_input):
        sigmod = 1.0 / (1.0 + np.exp(-2 * linear_input))
        return 2 * sigmod - 1

    def backward(self, output):
        return 1 - output * output
