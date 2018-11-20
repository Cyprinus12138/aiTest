from UCI.Extension.Decoder import *

INPUT_NODES = 64
HIDDEN_1 = 128
HIDDEN_2 = 128
OUTPUT_NODES = 1


class ActivateFunc:
    @staticmethod
    def mat_max(a, b):
        return np.array([max(a, i) for i in b])

    @staticmethod
    def fp(x):
        return ActivateFunc.mat_max(0, x)

    @staticmethod
    def bp(x):
        return ActivateFunc.mat_max(0, x) * (1 / x)


class BPNN:
    def __init__(self, data_path):
        self.learning_rate = 0.001
        test_set = Decoder(data_path)
        self.x, self.y = test_set.get_data(50000)
        converter = lambda x: min(1, x + 1)
        self.y = [converter(item) for item in self.y]
        self.weight_1 = np.random.normal([INPUT_NODES, HIDDEN_1])
        self.bias_1 = np.zeros([HIDDEN_1])
        self.weight_2 = np.random.normal([HIDDEN_1, HIDDEN_2])
        self.bias_2 = np.zeros([HIDDEN_2])
        self.weight_3 = np.random.normal([HIDDEN_2, OUTPUT_NODES])
        self.bias_3 = np.zeros([OUTPUT_NODES])
        self.input = 0
        self.hidden_1 = 0
        self.hidden_1_bp = 0
        self.hidden_2 = 0
        self.hidden_2_bp = 0
        self.output = 0
        self.output_bp = 0

    @staticmethod
    def hidden_layer_fp(weight, bias, x, activate=lambda x: x):
        return activate(np.add(np.matmul(x, weight), bias))

    @staticmethod
    def loss_func(y_, y):
        return abs(y_ - y), np.sign(y_ - y)

    def feed_forward(self, x):
        self.input = x
        self.hidden_1 = self.hidden_layer_fp(self.weight_1, self.bias_1, x, activate=ActivateFunc.fp)
        self.hidden_2 = self.hidden_layer_fp(self.weight_2, self.bias_2, self.hidden_1, activate=ActivateFunc.fp)
        self.output = self.hidden_layer_fp(self.weight_3, self.bias_3, self.hidden_2)
        return self.output

    def back_propagate_update(self, y):
        self.output_bp = self.loss_func(self.output, y)
        self.hidden_2_bp = self.output_bp[0] * self.weight_3
        self.hidden_1_bp = self.hidden_2_bp * self.weight_2 * ActivateFunc.fp(self.hidden_1)
        weight_1_delta = - np.sign(self.hidden_1_bp) * self.input * self.learning_rate
        weight_2_delta = - np.sign(self.hidden_2_bp) * self.hidden_1 * self.learning_rate
        weight_3_delta = -np.sign(self.output_bp) * self.hidden_2 * self.learning_rate
        self.weight_1 += weight_1_delta
        self.weight_2 += weight_2_delta
        self.weight_3 += weight_3_delta

