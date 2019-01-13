import numpy as np
import math
import random
from UCI.iris.Extension.Decoder import Decoder


PATH = r"D:\Git\aiTest\UCI\iris\DataSet\iris.data"
BATCH = 150


class LR:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.max_iteration = 2000
        self.weight_lambda = 0.01
        w = np.zeros((3, 4))
        self.w = w
        self.eval_mat = [[0] * 3] * 3

    def cal_e(self, x, l):
        theta_l = self.w[l]
        product = np.dot(theta_l, x)
        return math.exp(product)

    def cal_probability(self, x, j):
        molecule = self.cal_e(x, j)
        denominator = sum([self.cal_e(x, i) for i in range(3)])
        return molecule / denominator

    def cal_partial_derivative(self, x, y, j):
        first = int(y == j)     # 计算示性函数
        second = self.cal_probability(x, j)     # 计算概率
        return -x * (first - second) + self.weight_lambda * self.w[j]

    def predict_(self, x):
        result = np.matmul(self.w, x)
        row, column = result.shape
        _positon = np.argmax(result)
        m, n = divmod(_positon, column)
        return _positon

    def train_one_batch(self, features, labels):
        idx = 0
        while idx < self.max_iteration:
            print('loop %d' % idx)
            idx += 1
            index = random.randint(0, len(labels) - 1)

            x = features[index]
            y = labels[index]

            x = list(x)
            x = np.array(x)

            derivatives = [self.cal_partial_derivative(x, y, j) for j in range(3)]

            for j in range(len(self.w)):
                self.w[j] -= self.learning_rate * derivatives[j]

    def train(self):
        dataset = Decoder(PATH)
        x_list, y_list = dataset.get_data(BATCH)
        while len(y_list) >= BATCH:
            self.train_one_batch(x_list, y_list)
            x_list, y_list = dataset.get_data(BATCH)

    def predict(self, features):
        labels = []
        for feature in features:
            x = list(feature)
            x = np.matrix(x)
            x = np.transpose(x)
            labels.append(self.predict_(x))
        return labels

    def evaluation(self):
        data = Decoder(PATH)
        x_list, y_list = data.get_data(150)
        i = 0
        acc = 0
        for x in x_list:
            x = [[i] for i in x]
            self.eval_mat[int(y_list[i])][int(self.predict_(x))] += 1
            print(int(self.predict_(x)))
            if int(y_list[i]) == int(self.predict_(x)):
                acc += 1
            i += 1
        print(self.eval_mat)
        print(acc / 150)


