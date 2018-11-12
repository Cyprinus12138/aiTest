from UCI.Extension.Weaker import *
from UCI.Extension.Decoder import *

WEAKER_NUM = 6


class AdaBoost:
    weaker_set = []

    def __init__(self, weaker_num):
        dataset = Decoder("./DataSet/1year.arff")
        x, y = dataset.get_data(50000)
        self.weaker_num = weaker_num
        for i in range(weaker_num):
            self.weaker_set.append(Weaker(0.1, dataset.shape))
        self.weaker_set[0].train_with_batch(x, y, 580)
        self.weaker_set[0].get_alpha()

    def inference(self, x, threshold):
        result = sum([weaker.alpha * self.inf_as_one(weaker.classify(x), 0) for weaker in self.weaker_set])
        return self.inf_as_one(result, threshold)

    @staticmethod
    def inf_as_one(y, m):
        if y > m:
            return 1
        else:
            return -1

    def run_training(self):
        wp, wn = 1, 1
        for i in range(1, self.weaker_num):
            wn = wn * math.exp(- self.weaker_set[i - 1].get_alpha())
            wp = wp * math.exp(self.weaker_set[i - 1].get_alpha())
            z = len(self.weaker_set[i - 1].false_x) * wn + len(self.weaker_set[i - 1].true_x) * wp
            wn /= z
            wp /= z
            x = [sample * wn for sample in self.weaker_set[i - 1].false_x] + [sample * wp for sample in
                                                                              self.weaker_set[i - 1].true_x]
            y = [sample for sample in self.weaker_set[i - 1].false_y] + [sample for sample in
                                                                         self.weaker_set[i - 1].true_y]
            self.weaker_set[i].train_with_batch(x, y, 580)
            self.weaker_set[i].get_alpha()

    def run_evaluation(self):
        test_set = Decoder("./DataSet/1year.arff")
        x, y = test_set.get_data(50000)
        for threshold1 in range(1, 10):
            threshold = 2.5 / threshold1
            true_num = 0
            for i in range(len(x)):
                if self.inference(x[i], threshold) * y[i] > 0:
                    true_num += 1
            print(true_num / len(x))
