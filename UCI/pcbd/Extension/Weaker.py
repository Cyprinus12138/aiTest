import numpy as np
import math


class Weaker:

    def __init__(self, learning_rate, shape):
        self.alpha = 1
        self.em = 1
        self.false_x = []
        self.false_y = []
        self.true_x = []
        self.true_y = []
        w_r = np.random.normal(0.0, 1.0, shape)
        self.w_r = w_r.reshape([1, shape])
        w = np.zeros(shape)
        w = w.reshape([1, shape])
        self.w = w
        self.learning_rate = learning_rate

    def classify(self, x):
        return np.matmul(x, np.transpose(self.w))

    def classify_rand(self, x):
        return np.matmul(x, np.transpose(self.w_r))

    def train(self, x_list, y_list):
        false_num = 0
        for i in range(len(x_list)):
            y = self.classify(x_list[i])
            if y * y_list[i] <= 0:
                delta_w = -x_list[i] * self.learning_rate
                self.w = self.w + delta_w
        for i in range(len(x_list)):
            y = self.classify(x_list[i])
            if y_list[i] * y <= 0:
                false_num += 1
                self.false_x.append(x_list[i])
                self.false_y.append(y_list[i])
        return false_num / len(x_list)

    def get_false(self):
        return self.false_x, self.false_y

    def rand_classify(self, x_list, y_list):
        false_num = 0
        for i in range(len(x_list)):
            y = self.classify_rand(x_list[i])
            if y_list[i] * y <= 0:
                false_num += 1
        return false_num / len(x_list)

    def train_with_batch(self, x_list, y_list, size):
        false_num = 0
        for i in range(len(x_list) // size):
            delta_w = 0
            for j in range(size):
                y = self.classify(x_list[i * size + j])
                if y_list[i] * y <= 0:
                    delta_w += -x_list[i * size + j] * self.learning_rate
            self.w = self.w + delta_w
        for i in range(len(x_list)):
            y = self.classify(x_list[i])
            if y_list[i] * y <= 0:
                false_num += 1
                self.false_x.append(x_list[i])
                self.false_y.append(y_list[i])
            else:
                self.true_x.append(x_list[i])
                self.true_y.append(y_list[i])
        self.em = false_num / len(x_list)
        return self.em

    def get_alpha(self):
        self.alpha = 0.5 * math.log((1 - self.em) / self.em)
        return self.alpha
