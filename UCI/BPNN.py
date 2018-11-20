from UCI.Extension.Decoder import *
import time

INPUT_NODES = 64
HIDDEN_1 = 1024
HIDDEN_2 = 128
OUTPUT_NODES = 1
PATH = "./DataSet/2year.arff"


class ActivateFunc:
    @staticmethod
    def mat_max(a, b):
        return np.maximum(b, a)

    @staticmethod
    def fp(x):
        return ActivateFunc.mat_max(0, x)

    @staticmethod
    def bp(x):
        return np.maximum(np.sign(x), 0)


class BPNN:
    def __init__(self, learning_rate=0.001, data_path=PATH):
        self.learning_rate = learning_rate
        test_set = Decoder(data_path)
        self.x, self.y = test_set.get_data(50000)
        converter = lambda x: min(1, x + 1)
        self.y = [converter(item) for item in self.y]
        self.weight_1 = np.random.normal(loc=0.0, size=[INPUT_NODES, HIDDEN_1])
        self.bias_1 = np.zeros([1, HIDDEN_1])
        self.weight_2 = np.random.normal(loc=0.0, size=[HIDDEN_1, HIDDEN_2])
        self.bias_2 = np.zeros([1, HIDDEN_2])
        self.weight_3 = np.random.normal(loc=0.0, size=[HIDDEN_2, OUTPUT_NODES])
        self.bias_3 = np.zeros([1, OUTPUT_NODES])
        self.input = 0
        self.input_bp = 0
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
        return abs(y_ - y)

    def feed_forward(self, x):
        self.input = x
        self.hidden_1 = self.hidden_layer_fp(self.weight_1, self.bias_1, x, activate=ActivateFunc.fp)
        self.hidden_2 = self.hidden_layer_fp(self.weight_2, self.bias_2, self.hidden_1)
        self.output = self.hidden_layer_fp(self.weight_3, self.bias_3, self.hidden_2)
        return self.output

    def back_propagate_update(self, y):
        self.output_bp = self.loss_func(self.output, y)
        self.hidden_2_bp = np.matmul(self.output_bp, np.transpose(self.weight_3))
        self.hidden_1_bp = np.matmul(self.hidden_2_bp, np.transpose(self.weight_2))
        self.input_bp = np.matmul(self.hidden_1_bp, np.transpose(self.weight_1))
        weight_1_delta = - np.sign(self.input_bp) * self.input * ActivateFunc.bp(self.input) * self.learning_rate
        weight_2_delta = - np.sign(self.hidden_1_bp) * self.hidden_1 * self.learning_rate
        weight_3_delta = - np.sign(self.hidden_2_bp) * self.hidden_2 * self.learning_rate
        self.weight_1 += weight_1_delta.transpose() * np.ones(np.shape(self.weight_1))
        self.weight_2 += weight_2_delta.transpose() * np.ones(np.shape(self.weight_2))
        self.weight_3 += weight_3_delta.transpose() * np.ones(np.shape(self.weight_3))
        self.bias_1 += self.hidden_1_bp * self.learning_rate
        self.bias_2 += self.hidden_2_bp * self.learning_rate
        self.bias_3 += self.output_bp * self.learning_rate

    def run_training(self):
        t = time.time()
        for i in range(len(self.x)):
            self.feed_forward(self.x[i])
            self.back_propagate_update(self.y[i])
            if i % 10 == 0:
                print("Training the {} op, duration:{}s".format(i // 10, round(time.time() - t, 2)))
                print(self.output_bp[0], self.output)
                t = time.time()

    def run_evaluation_for_args(self, threshold=0.5):
        fp, tp, tn, fn = 0, 0, 0, 0
        test_set = Decoder(PATH)
        x, y = test_set.get_data(50000)
        for i in range(len(x)):
            tmp = self.feed_forward(x[i])[0][0]
            if tmp > threshold:
                tmp = 1
            else:
                tmp = 0
            if tmp == 1:
                if y[i] == 1:
                    tn += 1
                if y[i] == 0:
                    fn += 1
            elif tmp == 0:
                if y[i] == 1:
                    fp += 1
                if y[i] == 0:
                    tp += 1
        # precision = tp / (tp + fp)
        # recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print(tp,tn,fp,fn)
        return accuracy
        # return {"precision": precision, "recall": recall, "accuracy": accuracy}

    def run_evaluation_for_roc(self, threshold=0.5):
        fp, tp, tn, fn = 0, 0, 0, 0
        test_set = Decoder(PATH)
        x, y = test_set.get_data(50000)
        converter = lambda x: min(1, x + 1)
        y = [converter(item) for item in y]
        for i in range(len(x)):
            tmp = self.feed_forward(x[i])
            if tmp > threshold:
                tmp = 1
            else:
                tmp = 0
            if tmp == 1:
                if y[i] == 1:
                    tn += 1
                if y[i] == 0:
                    fn += 1
            elif tmp == 0:
                if y[i] == 1:
                    fp += 1
                if y[i] == 0:
                    tp += 1
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        return tpr, fpr
