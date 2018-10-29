import tensorflow as tf
from MNIST.Extension.MNISTDecoder import *


CHANNEL = 1
KERNEL_SIZE_1 = 5
KERNEL_SIZE_2 = 3
KERNEL_NUM_1 = 32
KERNEL_NUM_2 = 32
FC_1_NODES = 128
FC_2_NODES = 10
LEARNING_RATE = 0.000077
BATCH_SIZE = 100
MAX_STEP = 600
IMAGE_SIZE = 28
TRAIN_IMG_PATH = 'DataSet/train-images.idx3-ubyte'
TRAIN_LABEL_PATH = 'DataSet/train-labels.idx1-ubyte'
EVAL_IMG_PATH = 'DataSet/t10k-images.idx3-ubyte'
EVAL_LABEL_PATH = 'DataSet/t10k-labels.idx1-ubyte'


class CNN:
    def __init__(self):
        self.LEARNING_RATE = LEARNING_RATE

    def inference(self, _input):
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


def run_training():
    with tf.Graph().as_default():
        model = CNN()
        pl = DataFeeder(batch_size=BATCH_SIZE, img_path=TRAIN_IMG_PATH, label_path=TRAIN_LABEL_PATH)
        logits = model.inference(pl.image_place)
        loss = model.loss(outputs=logits, labels=pl.label_place)
        eval = model.evaluation(outputs=logits, labels=pl.label_place)
        train_op = model.training(loss)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        # builder = tf.saved_model.builder.SavedModelBuilder("./model0/")
        # summary_writer = tf.summary.FileWriter('./', sess.graph)
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
        save_path = saver.save(sess, "./model/CNN.ckpt")
        # builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map=None, assets_collection=None)
        # builder.save()


run_training()
