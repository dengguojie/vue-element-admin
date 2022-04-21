import numpy as np
import tensorflow as tf
from npu_bridge.estimator import npu_aicore_ops
from tensorflow.python.framework import graph_util
import os

pb_file_path = os.getcwd()

def generate_case_0():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        values1 = tf.compat.v1.placeholder(dtype="float32", shape=(128,))
        values2 = tf.compat.v1.placeholder(dtype="float32", shape=(128,))
        values3 = tf.compat.v1.placeholder(dtype="float32", shape=(128,))
        values4 = tf.compat.v1.placeholder(dtype="float32", shape=(128,))
        values_pre1 = tf.add(values1, values2)
        values_pre2 = tf.add(values3, values4)
        values = tf.add(values_pre1, values_pre2)
        result = npu_aicore_ops.nonzerowithvalue(x=values)

        tf.io.write_graph(sess.graph, logdir="./", name="non_zero_with_value_plugin_1.pb", as_text=False)

if __name__=='__main__':
    generate_case_0()