import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
from tensorflow.python.ops import gen_data_flow_ops
import numpy as np

with tf.Session(graph=tf.Graph()) as sess:
    size = tf.placeholder(dtype="int32", shape=())
    value = tf.placeholder(dtype="float32", shape=(2,2))
    index = tf.placeholder(dtype="int32", shape=())
    flow = tf.placeholder(dtype="float32", shape=())
    handleTensor = gen_data_flow_ops.tensor_array_v3(size= size, dtype = np.float32)
    output = gen_data_flow_ops.tensor_array_write_v3(handle = handleTensor[0], index=index, value=value, flow_in=flow)
    tf.io.write_graph(sess.graph, logdir="./", name="tensor_array_case_1.pb", as_text=False)
