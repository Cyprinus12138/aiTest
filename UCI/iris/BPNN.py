import tensorflow as tf
import math
from UCI.iris.Extension.Decoder import *
import time

BATCH_SIZE = 5
INPUT_NODES = 4
HIDDEN_1 = 128
HIDDEN_2 = 16
OUTPUT_NODES = 3
LEARNING_RATE = 0.001
MAX_STEP = 30
PATH = r"D:\Git\aiTest\UCI\iris\DataSet\iris.data"


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
    def __init__(self, batch_size, path):
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
        get_next = self.next_batch()
        return {data_pl: get_next[0],
                label_pl: get_next[1]}


def run_training():
    with tf.Graph().as_default():
        ti = time.time()
        model = FCNet()
        pl = DataFeeder(batch_size=BATCH_SIZE, path=PATH)
        logits = model.inference(pl.data_place)
        loss = model.loss(outputs=logits, labels=pl.label_place)
        eval = model.evaluation(outputs=logits, labels=pl.label_place)
        train_op = model.training(loss)
        sess = tf.Session()
        # summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        # builder = tf.saved_model.builder.SavedModelBuilder("./model0/")
        summary_writer = tf.summary.FileWriter('./', sess.graph)
        for step in range(MAX_STEP):
            feed_dict = pl.feed_place_holder
            _, loss_value, corr, t = sess.run([train_op, loss, eval, logits], feed_dict=feed_dict())
            if step % 10 == 0:
                ti = time.time() - ti
                print("Loss Function:", loss_value, "Duration:", round(ti, 2), "s")
                ti = time.time()
                # summary_str = sess.run(summary, feed_dict=feed_dict)
                # summary_writer.add_summary(summary_str, step)
                # summary_writer.flush()
                # checkpoint_file = os.path.join('./model/', 'model.ckpt')
                # saver.save(sess, checkpoint_file, global_step=step)
        save_path = saver.save(sess, "./model/FCBP.ckpt")
        # builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map=None, assets_collection=None)
        # builder.save()
        return save_path


run_training()