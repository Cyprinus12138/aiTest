import numpy as np


class Weaker:
    def __init__(self, learning_rate, shape):
        w = np.random.normal(0.0, 1.0, shape)
        w = w.reshape([1, shape])
        self.w = w
        self.learning_rate = learning_rate

    def classify(self, x):
        if np.multiply(x, np.transpose(self.w)) > 0 :
            return 1
        return -1

    def train(self, x_list, y_list):
        false_dict = {}
        false_num = 0
        for i in range(len(x_list)):
            y = self.classify(x_list[i])
            if y_list[i] * y < 0:
                false_dict[x_list[i]] = y_list[i]
                delta_w = x_list[i] * self.learning_rate
                self.w = self.w + delta_w
        for i in range(len(x_list)):
            y = self.classify(x_list[i])
            if y_list[i] * y <0:
                false_num += 1
        return false_num / len(x_list)