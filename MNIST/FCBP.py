import tensorflow as tf
import math
from MNIST.Extension.MNISTDecoder import *

BATCH_SIZE = 100
INPUT_NODES = 28 * 28
HIDDEN_1 = 2048
HIDDEN_2 = 1024
OUTPUT_NODES = 10
LEARNING_RATE = 0.000077
MAX_STEP = 600
TRAIN_IMG_PATH = 'DataSet/train-images.idx3-ubyte'
TRAIN_LABEL_PATH = 'DataSet/train-labels.idx1-ubyte'
EVAL_IMG_PATH = 'DataSet/t10k-images.idx3-ubyte'
EVAL_LABEL_PATH = 'DataSet/t10k-labels.idx1-ubyte'


class FCNet:
    def __init__(self):
        self.INPUT_NODES = INPUT_NODES
        self.HIDDEN_1 = HIDDEN_1
        self.HIDDEN_2 = HIDDEN_2
        self.OUTPUT_NODES = OUTPUT_NODES
        self.LEARNING_RATE = LEARNING_RATE

    def inference(self, input_):
        with tf.name_scope('hidden_1'):
            weights = tf.Variable(tf.truncated_normal([self.INPUT_NODES, self.HIDDEN_1], stddev=1.0 / math.sqrt(float(self.HIDDEN_1))), name='weights')
            biases = tf.Variable(tf.zeros([self.HIDDEN_1]), name='biases')
            hidden_1 = tf.add(tf.matmul(input_, weights), biases)
        with tf.name_scope('hidden_2'):
            weights = tf.Variable(tf.truncated_normal([self.HIDDEN_1, self.HIDDEN_2], stddev=1.0 / math.sqrt(float(self.HIDDEN_2))), name='weights')
            biases = tf.Variable(tf.zeros([self.HIDDEN_2]), name='biases')
            hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, weights), biases))
        with tf.name_scope('output'):
            weights = tf.Variable(tf.truncated_normal([self.HIDDEN_2, self.OUTPUT_NODES], stddev=1.0 / math.sqrt(float(self.OUTPUT_NODES))), name='weights')
            biases = tf.Variable(tf.zeros([self.OUTPUT_NODES]), name='biases')
            output = tf.add(tf.matmul(hidden_2, weights), biases)
        return output

    def loss(self, outputs, labels):
        labels = tf.to_int64(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=outputs, name='xentropy')
        return tf.reduce_mean(cross_entropy, name='xentropy_mean')

    def training(self, loss):
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def evaluation(self, outputs, labels):
        correct = tf.nn.in_top_k(outputs, labels, 1)
        return tf.reduce_sum(tf.cast(correct, tf.int32))


class DataFeeder:
    def __init__(self, batch_size, img_path, label_path):
        self.batch_index = 0
        self.dataset_image = MNISTDecoder(img_path)
        self.dataset_image = self.dataset_image.decode_to_matrix()
        self.dataset_label = MNISTDecoder(label_path)
        self.dataset_label = self.dataset_label.decode_to_matrix()
        self.batch_size = batch_size
        self.image_place = tf.placeholder(tf.float32, shape=(batch_size, INPUT_NODES))
        self.label_place = tf.placeholder(tf.int32, shape=batch_size)

    def next_batch(self):
        data = [self.dataset_image[self.batch_index:self.batch_index+self.batch_size], self.dataset_label[self.batch_index:self.batch_index+self.batch_size]]
        self.batch_index += self.batch_size
        return data

    def feed_place_holder(self):
        image_pl = self.image_place
        label_pl = self.label_place
        get_next = self.next_batch()
        return {image_pl: get_next[0],
                label_pl: get_next[1]}


def run_training():
    with tf.Graph().as_default():
        model = FCNet()
        pl = DataFeeder(batch_size=BATCH_SIZE, img_path=TRAIN_IMG_PATH, label_path=TRAIN_LABEL_PATH)
        logits = model.inference(pl.image_place)
        loss = model.loss(outputs=logits, labels=pl.label_place)
        eval = model.evaluation(outputs=logits, labels=pl.label_place)
        train_op = model.training(loss)
        sess = tf.Session()
        # summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        builder = tf.saved_model.builder.SavedModelBuilder("./model0/")
        summary_writer = tf.summary.FileWriter('./', sess.graph)
        for step in range(MAX_STEP):
            feed_dict = pl.feed_place_holder
            _, loss_value, corr, t = sess.run([train_op, loss, eval, logits], feed_dict=feed_dict())
            if step % 10 == 0:
                print(loss_value, corr)
                # summary_str = sess.run(summary, feed_dict=feed_dict)
                # summary_writer.add_summary(summary_str, step)
                # summary_writer.flush()
                # checkpoint_file = os.path.join('./model/', 'model.ckpt')
                # saver.save(sess, checkpoint_file, global_step=step)
        save_path = saver.save(sess, "./model/FCBP.ckpt")
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map=None, assets_collection=None)
        builder.save()
        print(sess.run(["hidden_1/weights"]))
        return save_path


run_training()