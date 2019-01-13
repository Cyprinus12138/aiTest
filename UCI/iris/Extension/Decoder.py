import numpy as np


type_map = {"Iris-setosa\n": 0, "Iris-versicolor\n": 1, "Iris-virginica\n": 2}


class Decoder:
    def __init__(self, file_path):
        self.file = open(file_path, 'r')
        self.shape = 0

    def get_data(self, size):
        x_list = []
        y_list = []
        for i in range(size):
            line = self.file.readline()
            x = line.split(",")
            if len(x) <= 2:
                return x_list, y_list
            self.shape = len(x)-1
            y = type_map[x[-1]]
            x = [float(data) for data in x[:-1]]
            # x = [x[i] / np.sum(x[:-1]) for i in range(len(x) - 1)]
            x = np.array(x)
            x = x.reshape([np.shape(x)[0]])
            x_list.append(x)
            y_list.append(y)
        return x_list, y_list
