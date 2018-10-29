import tensorflow as tf
import math
from MNIST.Extension.MNISTDecoder import *
import time


BATCH_SIZE = 1
INPUT_NODES = 28 * 28
HIDDEN_1 = 2048
HIDDEN_2 = 1024
OUTPUT_NODES = 10
TRAIN_IMG_PATH = 'DataSet/train-images.idx3-ubyte'
TRAIN_LABEL_PATH = 'DataSet/train-labels.idx1-ubyte'
EVAL_IMG_PATH = 'DataSet/t10k-images.idx3-ubyte'
EVAL_LABEL_PATH = 'DataSet/t10k-labels.idx1-ubyte'


class DataFeeder:
    def __init__(self, batch_size, img_path, label_path):
        self.batch_index = 0
        self.dataset_image = MNISTDecoder(img_path)
        self.dataset_image = self.dataset_image.decode_to_matrix()
        self.dataset_label = MNISTDecoder(label_path)
        self.dataset_label = self.dataset_label.decode_to_matrix()
        self.batch_size = batch_size
        self.image_place = tf.placeholder(tf.float32, shape=(batch_size, INPUT_NODES))
        # self.label_place = tf.placeholder(tf.int32, shape=batch_size)

    def next_batch(self):
        data = [self.dataset_image[self.batch_index:self.batch_index+self.batch_size], self.dataset_label[self.batch_index:self.batch_index+self.batch_size]]
        self.batch_index += self.batch_size
        return data

    def feed_place_holder(self):
        image_pl = self.image_place
        get_next = self.next_batch()
        return {image_pl: get_next[0]}

    def get_label(self):
        return self.dataset_label[self.batch_index - self.batch_size:self.batch_index]


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
    e_pl = DataFeeder(batch_size=1, img_path=EVAL_IMG_PATH, label_path=EVAL_LABEL_PATH)
    e_feed_dict = e_pl.feed_place_holder
    input_ = e_pl.image_place
    logits = inference(input_)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, save_path='./model/FCBP.ckpt')
    temp = 0
    t = time.time()
    for i in range(10000):
        if i % 1 == 0:
            DICT = e_feed_dict()
            lab = e_pl.get_label()
            corre = sess.run([logits], feed_dict=DICT)
            if np.argmax(corre) == lab[0]:
                temp += 1
    t1 = time.time()
    print("识别准确率为", temp / 100, "%", "计算耗时", t1 - t, "s")
