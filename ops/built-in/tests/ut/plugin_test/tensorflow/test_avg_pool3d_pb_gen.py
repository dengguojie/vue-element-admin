import tensorflow as tf
import os
import numpy as np
from tensorflow.python.framework import graph_util

pb_file_path = os.getcwd()

def generate_case_0():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        x_1 = tf.compat.v1.placeholder(dtype="float16", shape=(1,3,3,3,1))
        x_2 = tf.compat.v1.placeholder(dtype="float16", shape=(1,3,3,3,1))
        x = tf.add(x_1, x_2)
        op = tf.nn.avg_pool3d(input=x, ksize=[1,2,2,2,1], strides=[1,1,1,1,1],
                              padding="SAME",data_format="NDHWC")
        tf.io.write_graph(sess.graph, logdir="./", name="avgpool3d_case_1.pb", as_text=False)

if __name__=='__main__':
    generate_case_0()
