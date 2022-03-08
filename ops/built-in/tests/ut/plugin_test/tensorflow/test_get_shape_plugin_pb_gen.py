import numpy as np
import tensorflow as tf
from npu_bridge.experimental import get_shape
from tensorflow.python.framework import graph_util
import os

config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["dynamic_input"].b = True
custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("dynamic_execute")
custom_op.parameter_map["dynamic_inputs_shape_range"].s = tf.compat.as_bytes("getnext:[128]")


def generate_case_0():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        values1 = tf.compat.v1.placeholder(dtype="float32", shape=(None,))
        values2 = tf.compat.v1.placeholder(dtype="float32", shape=(128,))
        values3 = tf.compat.v1.placeholder(dtype="float32", shape=(128,))
        values4 = tf.compat.v1.placeholder(dtype="float32", shape=(128,))
        values_pre1 = tf.add(values1, values2)
        values_pre2 = tf.add(values3, values4)
        values = tf.add(values_pre1, values_pre2)
        result = get_shape.getshape(x=values)

        tf.io.write_graph(sess.graph, logdir="./", name="get_shape_plugin_1.pb", as_text=False)

if __name__=='__main__':
    generate_case_0()