from UCI.Extension.Weaker import *
from UCI.Extension.Decoder import *


WEAKER_NUM = 10


class AdaBoost:
    weaker_set = []

    def __init__(self, weak_num):
        dataset = Decoder("./DataSet/1year.arff")
        x, y = dataset.get_data(50000)
        self.weak_num = weak_num
        for i in range(weak_num):
            self.weaker_set.append(Weaker(0.1, dataset.shape))
        self.weaker_set[0].train_with_batch(x, y, 580)

    def inference(self, x):
        result = sum([weaker.alpha * self.inf_as_one(weaker.classify(x)) for weaker in self.weaker_set])
        return self.inf_as_one(result)

    @staticmethod
    def inf_as_one(y):
        if y > 0:
            return 1
        else:
            return -1

