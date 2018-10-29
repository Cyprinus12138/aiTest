import tensorflow as tf
from MNIST.Extension.MNISTDecoder import *
import time


CHANNEL = 1
KERNEL_SIZE_1 = 5
KERNEL_SIZE_2 = 3
KERNEL_NUM_1 = 32
KERNEL_NUM_2 = 64
FC_1_NODES = 512
FC_2_NODES = 10
BATCH_SIZE = 1
IMAGE_SIZE = 28
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
        self.image_place = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, CHANNEL))
        self.label_place = tf.placeholder(tf.int32, shape=batch_size)

    def next_batch(self):
        data = [self.dataset_image[self.batch_index:self.batch_index+self.batch_size], self.dataset_label[self.batch_index:self.batch_index+self.batch_size]]
        self.batch_index += self.batch_size
        return data

    def feed_place_holder(self):
        image_pl = self.image_place
        label_pl = self.label_place
        get_next = self.next_batch()
        get_next[0] = [x.reshape([IMAGE_SIZE, IMAGE_SIZE, CHANNEL]) for x in get_next[0]]
        return {image_pl: get_next[0],
                label_pl: get_next[1]}

    def get_label(self):
        return self.dataset_label[self.batch_index - self.batch_size:self.batch_index]


with tf.Graph().as_default():
    def inference( _input):
        with tf.name_scope('conv_1'):
            weight = tf.Variable(tf.truncated_normal([KERNEL_SIZE_1, KERNEL_SIZE_1, CHANNEL, KERNEL_NUM_1], stddev=0.1), name="weights")
            biases = tf.Variable(tf.zeros([KERNEL_NUM_1]), name="biases")
            conv_1 = tf.nn.relu(tf.add(tf.nn.conv2d(_input, weight, strides=[1, 1, 1, 1], padding='SAME'), biases))
        with tf.name_scope("max_pool_1"):
            pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        with tf.name_scope('conv_2'):
            weight = tf.Variable(tf.truncated_normal([KERNEL_SIZE_2, KERNEL_SIZE_2, KERNEL_NUM_1, KERNEL_NUM_2], stddev=0.1), name="weights")
            biases = tf.Variable(tf.zeros([KERNEL_NUM_2]), name="biases")
            conv_2 = tf.nn.relu(tf.add(tf.nn.conv2d(pool_1, weight, strides=[1, 1, 1, 1], padding='SAME'), biases))
        with tf.name_scope("max_pool_2"):
            pool_2 = tf.nn.max_pool(conv_2, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')
        with tf.name_scope("fc_1"):
            weight = tf.Variable(tf.truncated_normal([7 * 7 * KERNEL_NUM_2, FC_1_NODES], stddev=0.1), name="weights")
            biases = tf.Variable(tf.zeros([FC_1_NODES]), name="biases")
            fc_1 = tf.nn.relu(tf.add(tf.matmul(tf.reshape(pool_2, [BATCH_SIZE, 7 * 7 * KERNEL_NUM_2]), weight), biases))
        with tf.name_scope("fc_2"):
            weight = tf.Variable(tf.truncated_normal([FC_1_NODES, FC_2_NODES], stddev=0.1), name="weights")
            biases = tf.Variable(tf.zeros([FC_2_NODES]), name="biases")
            fc_2 = tf.add(tf.matmul(fc_1, weight), biases)
        return fc_2
    e_pl = DataFeeder(batch_size=1, img_path=EVAL_IMG_PATH, label_path=EVAL_LABEL_PATH)
    e_feed_dict = e_pl.feed_place_holder
    input_ = e_pl.image_place
    logits = inference(input_)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, save_path='./model/CNN.ckpt')
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