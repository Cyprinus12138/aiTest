import numpy as np
import math


class LR:

    def __init__(self, learning_rate, shape):
        self.x = []
        self.y = []
        w = np.zeros(shape)
        w = w.reshape([3, shape])
        self.w = w

    def train(self, x_list, y_list):
        pass

