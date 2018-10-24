import tensorflow as tf
import math


INPUT_NODES = 28 * 28
HIDDEN_1 = 2048
HIDDEN_2 = 1024
OUTPUT_NODES = 10


with tf.Graph().as_default():
    def inference(input_):
        with tf.name_scope('hidden_1'):
            weights = tf.Variable(tf.truncated_normal([INPUT_NODES, HIDDEN_1], stddev=1.0 / math.sqrt(float(HIDDEN_1))), name='weights')
            biases = tf.Variable(tf.zeros([HIDDEN_1]), name='biases')
            hidden_1 = tf.add(tf.matmul(input_, weights), biases)
        with tf.name_scope('hidden_2'):
            weights = tf.Variable(tf.truncated_normal([HIDDEN_1, HIDDEN_2], stddev=1.0 / math.sqrt(float(HIDDEN_2))), name='weights')
            biases = tf.Variable(tf.zeros([HIDDEN_2]), name='biases')
            hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, weights), biases))
        with tf.name_scope('output'):
            weights = tf.Variable(tf.truncated_normal([HIDDEN_2, OUTPUT_NODES], stddev=1.0 / math.sqrt(float(OUTPUT_NODES))), name='weights')
            biases = tf.Variable(tf.zeros([OUTPUT_NODES]), name='biases')
            output = tf.add(tf.matmul(hidden_2, weights), biases)
        return output


    x = tf.placeholder(tf.float32, shape=(1, INPUT_NODES))
    y = inference(x)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, save_path='./model/FCBP.ckpt')
    builder = tf.saved_model.builder.SavedModelBuilder("./inference/")
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
    builder.save()