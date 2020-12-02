import tensorflow as tf
import os
from tensorflow.python.framework import graph_util

pb_file_path = os.getcwd()

with tf.Session(graph=tf.Graph()) as sess:
    # x = tf.constant((1,), "float16")
    # y = tf.constant((1,), "float16")
    x = tf.placeholder(dtype = "float16", shape=(1,))
    y = tf.placeholder(dtype = "float16", shape=(1,))
    op = tf.add(x, y, name='add_test_1')
    # sess.run(op)
    tf.train.write_graph(sess.graph_def, "./", "add_case.txt")
    tf.io.write_graph(sess.graph,logdir="./", name="add_case_1.txt", as_text=False)

