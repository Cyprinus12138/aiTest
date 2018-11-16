from UCI.Extension.Weaker import *
from UCI.Extension.Decoder import *
import time

WEAKER_NUM = 6
PATH = "./DataSet/2year.arff"


class AdaBoost:
    weaker_set = []

    def __init__(self, weaker_num=WEAKER_NUM, learning_rate=0.0001):
        dataset = Decoder(PATH)
        x, y = dataset.get_data(50000)
        self.weaker_num = weaker_num
        for i in range(weaker_num):
            self.weaker_set.append(Weaker(learning_rate, dataset.shape))
        self.weaker_set[0].train_with_batch(x, y, 580)
        self.weaker_set[0].get_alpha()

    def inference(self, x, threshold=0.0):
        result = sum([weaker.alpha * self.inf_as_one(weaker.classify(x), 0) for weaker in self.weaker_set])
        return self.inf_as_one(result, threshold)

    @staticmethod
    def inf_as_one(y, m=0.0):
        if y > m:
            return 1
        else:
            return -1

    def run_training(self):
        wp, wn = 1, 1
        for i in range(1, self.weaker_num):
            t = time.time()
            wn = wn * math.exp(- self.weaker_set[i - 1].get_alpha())
            wp = wp * math.exp(self.weaker_set[i - 1].get_alpha())
            z = len(self.weaker_set[i - 1].false_x) * wn + len(self.weaker_set[i - 1].true_x) * wp
            wn /= z
            wp /= z
            x = [sample * wn for sample in self.weaker_set[i - 1].false_x] + [sample * wp for sample in
                                                                              self.weaker_set[i - 1].true_x]
            y = [sample for sample in self.weaker_set[i - 1].false_y] + [sample for sample in
                                                                         self.weaker_set[i - 1].true_y]
            acc = self.weaker_set[i].train_with_batch(x, y, 580)
            alpha = self.weaker_set[i].get_alpha()
            print("Weaker {} trained OK, Duration:{}s, Acc:{}%, Alpha={}".format(i + 1, round(time.time() - t, 2), round((1 - acc) * 100, 2), alpha))
        self.alpha_normalizing()

    def alpha_normalizing(self):
        temp = sum([x.alpha for x in self.weaker_set])
        for i in range(self.weaker_num):
            self.weaker_set[i].alpha = self.weaker_set[i].alpha / temp

    def run_evaluation_for_roc(self, threshold):
        test_set = Decoder(PATH)
        x, y = test_set.get_data(50000)
        fp, tp, tn, fn = 0, 0, 0, 0
        for i in range(len(x)):
            tmp = self.inference(x[i], threshold)
            if tmp == 1:
                if y[i] == 1:
                    tn += 1
                if y[i] == -1:
                    fn += 1
            elif tmp == -1:
                if y[i] == 1:
                    fp += 1
                if y[i] == -1:
                    tp += 1
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        return tpr, fpr

    def run_evaluation_for_args(self, threshold):
        test_set = Decoder(PATH)
        x, y = test_set.get_data(50000)
        fp, tp, tn, fn = 0, 0, 0, 0
        for i in range(len(x)):
            tmp = self.inference(x[i], threshold)
            if tmp == 1:
                if y[i] == 1:
                    tn += 1
                if y[i] == -1:
                    fn += 1
            elif tmp == -1:
                if y[i] == 1:
                    fp += 1
                if y[i] == -1:
                    tp += 1
        # print(tp, tn, fp, fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return {"precision": precision, "recall": recall, "accuracy": accuracy}