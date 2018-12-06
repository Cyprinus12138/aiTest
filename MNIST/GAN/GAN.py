from MNIST.Extension.MNISTDecoder import *
import tensorflow as tf
from PIL import Image
import os


G_LEARNING_RATE = 0.00001
D_LEARNING_RATE = 0.00001
BATCH_SIZE = 1
G_INPUT_NODES = 100
G_LAYER_1 = 128
G_LAYER_2 = 28 * 28
D_INPUT_NODES = 28 * 28
D_LAYER_1 = 128
D_LAYER_2 = 1
MAX_STEP = 30000
TRAIN_IMG_PATH = r'D:\Git\aiTest\MNIST\DataSet\train-images.idx3-ubyte'


class GAN:
    @staticmethod
    def generator(z):
        with tf.name_scope("Generator"):
            with tf.name_scope("layer_1"):
                weight = tf.Variable(tf.random_normal([G_INPUT_NODES, G_LAYER_1], stddev=0.1), name='weight')
                biases = tf.Variable(tf.zeros([G_LAYER_1]), name='biases')
                layer_1 = tf.nn.relu(tf.add(tf.matmul(z, weight), biases))
            with tf.name_scope("layer_2"):
                weight = tf.Variable(tf.random_normal([G_LAYER_1, G_LAYER_2], stddev=0.1), name='weight')
                biases = tf.Variable(tf.zeros([G_LAYER_2]), name='biases')
                layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weight), biases))
        return layer_2

    @staticmethod
    def discriminator(x):
        with tf.name_scope("Discriminator"):
            with tf.name_scope("layer_1"):
                weight = tf.Variable(tf.random_normal([D_INPUT_NODES, D_LAYER_1], stddev=0.1), name='weight')
                biases = tf.Variable(tf.zeros([D_LAYER_1]), name='biases')
                layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weight), biases))
            with tf.name_scope("layer_2"):
                weight = tf.Variable(tf.random_normal([D_LAYER_1, D_LAYER_2], stddev=0.1), name='weight')
                biases = tf.Variable(tf.zeros([D_LAYER_2]), name='biases')
                layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weight), biases))
        return layer_2

    @staticmethod
    def d_loss(d_fake, d_real):
        return - tf.reduce_mean(tf.log(d_real) + tf.log(1 - d_fake))

    @staticmethod
    def g_loss(d_fake):
        return - tf.reduce_mean(tf.log(d_fake))

    @staticmethod
    def training(g_loss, d_loss):
        tf.summary.scalar("G_loss", g_loss)
        tf.summary.scalar("D_loss", d_loss)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=G_LEARNING_RATE)
        d_optimizer = tf.train.AdamOptimizer(learning_rate=D_LEARNING_RATE)
        train_op = [g_optimizer.minimize(g_loss, global_step=global_step), d_optimizer.minimize(d_loss)]
        return train_op


class DataFeeder:
    def __init__(self, batch_size, img_path):
        self.batch_index = 0
        self.dataset_image = MNISTDecoder(img_path)
        self.dataset_image = self.dataset_image.decode_to_matrix()
        self.batch_size = batch_size
        self.x_pl = tf.placeholder(tf.float32, shape=(batch_size, D_INPUT_NODES), name='X')
        self.z_pl = tf.placeholder(tf.float32, shape=(batch_size, G_INPUT_NODES), name='Z')

    def next_batch(self):
        data = self.dataset_image[self.batch_index:self.batch_index+self.batch_size]
        self.batch_index += self.batch_size
        return data

    def feed_place_holder(self):
        get_next = self.next_batch()
        z = np.random.uniform(-1., 1., size=[self.batch_size, G_INPUT_NODES])
        return {self.x_pl: get_next,
                self.z_pl: z}


def run_training():
    with tf.Graph().as_default():
        model = GAN()
        pl = DataFeeder(batch_size=BATCH_SIZE, img_path=TRAIN_IMG_PATH)
        generate = model.generator(pl.z_pl)
        d_real = model.discriminator(pl.x_pl)
        d_fake = model.discriminator(generate)
        g_loss = model.g_loss(d_fake=d_fake)
        d_loss = model.d_loss(d_fake=d_fake, d_real=d_real)
        train_op = model.training(g_loss=g_loss, d_loss=d_loss)
        sess = tf.Session()
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter('./', sess.graph)
        for step in range(MAX_STEP):
            feed_dict = pl.feed_place_holder()
            _, _, g_loss_value, d_loss_value = sess.run(train_op + [d_loss, g_loss], feed_dict=feed_dict)
            if step % 100 == 0:
                print("g", g_loss_value, "d", d_loss_value)
                gen = sess.run(generate, feed_dict={pl.z_pl: np.random.uniform(-1., 1., size=[BATCH_SIZE, G_INPUT_NODES])})
                gen = np.reshape(gen, [28, 28])
                im = Image.fromarray(gen)
                im.save(str(step) + ".GIF")
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
                checkpoint_file = os.path.join('./model/', 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
        save_path = saver.save(sess, "./model/GAN.ckpt")
        return save_path


run_training()