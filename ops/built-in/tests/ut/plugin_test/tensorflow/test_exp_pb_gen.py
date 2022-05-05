import os
import tensorflow as tf
from tensorflow.python.framework import graph_util

pb_file_path = os.getcwd()

with tf.Session(graph=tf.Graph()) as sess:
    x = tf.placeholder(dtype="float16", shape=(16, 16))
    op = tf.exp(x, name='exp_test_1')
    tf.io.write_graph(sess.graph, logdir="./", name="exp_case_1.pb", as_text=False)
