import tensorflow as tf
import math
from UCI.pcbd.Extension.Decoder import *
import time


BATCH_SIZE = 5
INPUT_NODES = 4
HIDDEN_1 = 128
HIDDEN_2 = 16
OUTPUT_NODES = 3
LEARNING_RATE = 0.001
MAX_STEP = 30
PATH = r"D:\Git\aiTest\UCI\iris\DataSet\iris.data"


class DataFeeder:
    def __init__(self, batch_size, path):
        self.get_next = None
        self.batch_index = 0
        self.dataset_data, self.dataset_label = Decoder(path).get_data(150)
        self.batch_size = batch_size
        self.data_place = tf.placeholder(tf.float32, shape=(batch_size, INPUT_NODES))
        self.label_place = tf.placeholder(tf.int32, shape=batch_size)

    def next_batch(self):
        data = [self.dataset_data[self.batch_index:self.batch_index + self.batch_size], self.dataset_label[self.batch_index:self.batch_index + self.batch_size]]
        self.batch_index += self.batch_size
        return data

    def feed_place_holder(self):
        data_pl = self.data_place
        label_pl = self.label_place
        self.get_next = self.next_batch()
        return {data_pl: self.get_next[0]}

    def get_label(self):
        return self.get_next[1]


with tf.Graph().as_default():
    def inference(_input):
        with tf.name_scope('hidden_1'):
            weights = tf.Variable(
                tf.truncated_normal([INPUT_NODES, HIDDEN_1], mean=0, stddev=1.0 / math.sqrt(float(HIDDEN_1))),
                name='weights')
            biases = tf.Variable(tf.zeros([HIDDEN_1]), name='biases')
            hidden_1 = tf.add(tf.matmul(_input, weights), biases)
        with tf.name_scope('hidden_2'):
            weights = tf.Variable(
                tf.truncated_normal([HIDDEN_1, HIDDEN_2], mean=0, stddev=1.0 / math.sqrt(float(HIDDEN_2))),
                name='weights')
            biases = tf.Variable(tf.zeros([HIDDEN_2]), name='biases')
            hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, weights), biases))
        with tf.name_scope('output'):
            weights = tf.Variable(
                tf.truncated_normal([HIDDEN_2, OUTPUT_NODES], mean=0, stddev=1.0 / math.sqrt(float(OUTPUT_NODES))),
                name='weights')
            biases = tf.Variable(tf.zeros([OUTPUT_NODES]), name='biases')
            output = tf.add(tf.matmul(hidden_2, weights), biases)
        return output
    e_pl = DataFeeder(batch_size=BATCH_SIZE, path=PATH)
    e_feed_dict = e_pl.feed_place_holder
    input_ = e_pl.data_place
    logits = inference(input_)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, save_path='./model/FCBP.ckpt')
    temp = 0
    eval_mat = [[0] * 3] * 3
    accuracy = 0
    for i in range(150):
        if i % 1 == 0:
            DICT = e_feed_dict()
            corre = sess.run([logits], feed_dict=DICT)
            y = e_pl.get_label()
            for j in range(BATCH_SIZE):
                tmp = np.argmax(corre[0][j])
                eval_mat[int(tmp)][y] += 1
                if int(tmp) == y:
                    accuracy += 1
    print({"accuracy": accuracy/150})
