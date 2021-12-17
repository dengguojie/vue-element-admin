import numpy as np
import tensorflow as tf
from npu_bridge.npu_cpu import npu_cpu_ops
from tensorflow.python.framework import graph_util
import os

pb_file_path = os.getcwd()

def generate_case_0():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        values1 = tf.compat.v1.placeholder(dtype="float32", shape=(100,))
        values2 = tf.compat.v1.placeholder(dtype="float32", shape=(100,))
        indices1 = tf.compat.v1.placeholder(dtype="int32", shape=(200,))
        indices2 = tf.compat.v1.placeholder(dtype="int32", shape=(200,))
        count_list = tf.compat.v1.placeholder(dtype="int32", shape=(1,))
        values = tf.add(values1, values2)
        indices = tf.add(indices1, indices2)
        result = npu_cpu_ops.non_zero_with_value_shape(value=values, index=indices, count=count_list)

        tf.io.write_graph(sess.graph, logdir="./", name="non_zero_with_value_shape_plugin_1.pb", as_text=False)

if __name__=='__main__':
    generate_case_0()

