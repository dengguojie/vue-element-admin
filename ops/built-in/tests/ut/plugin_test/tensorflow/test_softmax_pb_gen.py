import tensorflow as tf
import os
from tensorflow.python.framework import graph_util

pb_file_path = os.getcwd()

with tf.Session(graph=tf.Graph()) as sess:
    x = tf.placeholder(dtype="float16", shape=(1,))
    y = tf.placeholder(dtype="float16", shape=(1,))
    add_op = tf.add(x, y)
    op = tf.nn.softmax(add_op, name='softmax_test_1')
    tf.io.write_graph(sess.graph, logdir="./", name="softmax_case_1.pb", as_text=False)
