import tensorflow as tf
import os

pb_file_path = os.getcwd()

with tf.compat.v1.Session(graph=tf.Graph()) as sess:

    input_1 = tf.compat.v1.placeholder(dtype="float16", shape=(1, 1, 2, 2, 2))
    input_2 = tf.compat.v1.placeholder(dtype="float16", shape=(1, 1, 2, 2, 2))
    filter_1 = tf.compat.v1.placeholder(dtype="float16", shape=(1, 1, 1, 1, 1))
    filter_2 = tf.compat.v1.placeholder(dtype="float16", shape=(1, 1, 1, 1, 1))
    input = tf.add(input_1, input_2)
    filter = tf.add(filter_1, filter_2)
    op = tf.nn.conv3d(input, filter,
                      strides=[1, 1, 1, 1, 1],
                      padding="VALID",
                      data_format='NCDHW',
                      name='Conv3d')

    tf.io.write_graph(sess.graph, logdir="./", name="conv3d_plugin_case_1.pb", as_text=False)
