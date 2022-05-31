import tensorflow as tf
import os
import numpy as np
from tensorflow.python.framework import graph_util

pb_file_path = os.getcwd()

def generate_case_0():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        x0_holder = tf.compat.v1.placeholder(dtype="float16", shape=(16,16,16,64))
        x1_holder = tf.compat.v1.placeholder(dtype="float16", shape=(16,16,16,64))
        x2_holder = tf.compat.v1.placeholder(dtype="float16", shape=(16,16,16,64))
        x3_holder = tf.compat.v1.placeholder(dtype="float16", shape=(16,16,16,64))
        add0_op = tf.add(x0_holder, x1_holder)
        add1_op = tf.add(x2_holder, x3_holder)
        add_op = tf.add(add0_op, add1_op)
        op = tf.split(add_op, 4, 3, 4, name='split_1')
        tf.io.write_graph(sess.graph, logdir="./", name="split_case_1.pb", as_text=False)
        # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['split_1'])
        # with tf.gfile.FastGFile('./split_case_1.pb', mode='wb') as f:
        #     f.write(constant_graph.SerializeToString())

if __name__=='__main__':
    generate_case_0()
