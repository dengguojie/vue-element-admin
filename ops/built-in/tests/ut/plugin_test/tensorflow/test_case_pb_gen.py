import tensorflow as tf
import os
from tensorflow.python.framework import graph_util

pb_file_path = os.getcwd()

with tf.Session(graph=tf.Graph()) as sess:
    x = tf.placeholder(dtype = "float16", shape=(1,))
    y = tf.placeholder(dtype = "float16", shape=(1,))
    op = tf.add(x, y, name='add_test_1')
    tf.io.write_graph(sess.graph,logdir="./", name="add_case_1.pb", as_text=False)

