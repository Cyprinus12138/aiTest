import numpy as np


class Decoder:
    def __init__(self, file_path):
        self.file = open(file_path, 'r')
        self.shape = 0
        line = self.file.readline()
        while not ("@data" in line):
            line = self.file.readline()

    def get_data(self, size):
        x_list = []
        y_list = []
        for i in range(size):
            line = self.file.readline()
            x = line.split(",")
            if len(x) <= 2:
                return x_list, y_list
            self.shape = len(x)-1
            for j in range(len(x)):
                if x[j] == '?':
                    x[j] = '0'
            x = [float(data) for data in x]
            x = [data / np.sum(x) for data in x]      # 归一化可提高5个百分点
            y = x[-1]
            if y == 0:
                y = -1
            x = np.array(x[:-1])
            x = x.reshape([1, np.shape(x)[0]])
            x_list.append(x)
            y_list.append(y)
        return x_list, y_list
