from MNIST.Extension.MNISTDecoder import *
import tensorflow as tf
from PIL import Image
import os


G_LEARNING_RATE = 0.02
D_LEARNING_RATE = 0.0002
BATCH_SIZE = 100
G_INPUT_NODES = 100
G_LAYER_1 = 1024
G_LAYER_2 = 28 * 28
D_INPUT_NODES = 28 * 28
D_LAYER_1 = 128
D_LAYER_2 = 1
MAX_STEP = 60000
TRAIN_IMG_PATH = r'D:\Git\aiTest\MNIST\DataSet\train-images.idx3-ubyte'
G_THETA = []
D_THETA = []


class GAN:
    def __init__(self):
        with tf.variable_scope("Generator"):
            with tf.name_scope("layer_1"):
                weight = tf.Variable(tf.random_normal([G_INPUT_NODES, G_LAYER_1], stddev=0.1), name='weight')
                biases = tf.Variable(tf.zeros([G_LAYER_1]), name='biases')
                G_THETA.append(weight)
                G_THETA.append(biases)
            with tf.name_scope("layer_2"):
                weight = tf.Variable(tf.random_normal([G_LAYER_1, G_LAYER_2], stddev=0.1), name='weight')
                biases = tf.Variable(tf.zeros([G_LAYER_2]), name='biases')
                G_THETA.append(weight)
                G_THETA.append(biases)
        with tf.variable_scope("Discriminator"):
            with tf.name_scope("layer_1"):
                weight = tf.Variable(tf.random_normal([D_INPUT_NODES, D_LAYER_1], stddev=0.1), name='weight')
                biases = tf.Variable(tf.zeros([D_LAYER_1]), name='biases')
                D_THETA.append(weight)
                D_THETA.append(biases)
            with tf.name_scope("layer_2"):
                weight = tf.Variable(tf.random_normal([D_LAYER_1, D_LAYER_2], stddev=0.1), name='weight')
                biases = tf.Variable(tf.zeros([D_LAYER_2]), name='biases')
                D_THETA.append(weight)
                D_THETA.append(biases)

    @staticmethod
    def generator(z):
        with tf.variable_scope("Generator", reuse=True):
            with tf.name_scope("layer_1"):
                weight = tf.Variable(tf.random_normal([G_INPUT_NODES, G_LAYER_1], stddev=0.1), name='weight')
                biases = tf.Variable(tf.zeros([G_LAYER_1]), name='biases')
                layer_1 = tf.nn.relu(tf.add(tf.matmul(z, weight), biases))
                G_THETA.append(weight)
                G_THETA.append(biases)
            with tf.name_scope("layer_2"):
                weight = tf.Variable(tf.random_normal([G_LAYER_1, G_LAYER_2], stddev=0.1), name='weight')
                biases = tf.Variable(tf.zeros([G_LAYER_2]), name='biases')
                layer_2 = tf.add(tf.matmul(layer_1, weight), biases)
                G_THETA.append(weight)
                G_THETA.append(biases)
        return layer_2

    @staticmethod
    def discriminator(x):
        with tf.variable_scope("Discriminator", reuse=True):
            with tf.name_scope("layer_1"):
                weight = tf.Variable(tf.random_normal([D_INPUT_NODES, D_LAYER_1], stddev=0.1), name='weight')
                biases = tf.Variable(tf.zeros([D_LAYER_1]), name='biases')
                layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weight), biases))
                D_THETA.append(weight)
                D_THETA.append(biases)
            with tf.name_scope("layer_2"):
                weight = tf.Variable(tf.random_normal([D_LAYER_1, D_LAYER_2], stddev=0.1), name='weight')
                biases = tf.Variable(tf.zeros([D_LAYER_2]), name='biases')
                layer_2 = tf.add(tf.matmul(layer_1, weight), biases)
                D_THETA.append(weight)
                D_THETA.append(biases)
        return layer_2

    @staticmethod
    def d_loss(d_fake, d_real):
        return tf.reduce_mean((tf.abs(tf.nn.sigmoid_cross_entropy_with_logits(labels=[[1.0] for i in range(BATCH_SIZE)], logits=d_real)) + tf.abs(tf.nn.sigmoid_cross_entropy_with_logits(labels=[[0.0] for i in range(BATCH_SIZE)], logits=d_fake))) / 2)

    @staticmethod
    def g_loss(d_fake):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=[[1.0]for i in range(BATCH_SIZE)], logits=d_fake))

    @staticmethod
    def training(g_loss, d_loss):
        tf.summary.scalar("G_loss", g_loss)
        tf.summary.scalar("D_loss", d_loss)
        tf.summary.scalar("D/G", d_loss / g_loss)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        g_optimizer = tf.train.GradientDescentOptimizer(learning_rate=G_LEARNING_RATE, name="G_Optimizer")
        d_optimizer = tf.train.GradientDescentOptimizer(learning_rate=D_LEARNING_RATE, name="D_Optimizer")
        train_op = [g_optimizer.minimize(g_loss, global_step=global_step, var_list=G_THETA), d_optimizer.minimize(d_loss, var_list=D_THETA)]
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
        d_loss = model.d_loss(d_fake=d_fake, d_real=d_real)
        g_loss = model.g_loss(d_fake=d_fake)
        train_op = model.training(g_loss=g_loss, d_loss=d_loss)
        sess = tf.Session()
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter('./', sess.graph)
        feed_dict = pl.feed_place_holder()
        for step in range(MAX_STEP):
            _, _, d_loss_value, g_loss_value, gen = sess.run(train_op + [d_loss, g_loss, generate], feed_dict=feed_dict)
            if step % 100 == 0:
                feed_dict = pl.feed_place_holder()
                print("g", g_loss_value, "d", d_loss_value, "g/d", g_loss_value / d_loss_value)
                gen = np.reshape(gen, [100, 28, 28])
                gen = gen[0] * 225
                im = Image.fromarray(gen)
                im.save(str(step) + ".gif")
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
        save_path = saver.save(sess, "./model/GAN.ckpt")
        return save_path


if __name__ == "__main__":
    run_training()
