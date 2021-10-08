import tensorflow as tf
import os
from tensorflow.python.framework import graph_util

pb_file_path = os.getcwd()
nbins=5
with tf.Session(graph=tf.Graph()) as sess:
    x = tf.placeholder(dtype="int32", shape=(2,))
    y = tf.placeholder(dtype="int32", shape=(2,))
    op = tf.histogram_fixed_width(x, y, nbins,dtype="int32",name='histogram_fixed_width')
    tf.io.write_graph(sess.graph, logdir="./", name="histogram_fixed_width.pb", as_text=False)
