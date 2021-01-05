import tensorflow as tf
import os
import numpy as np
from tensorflow.python.framework import graph_util

pb_file_path = os.getcwd()

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    
    grads_1 = tf.compat.v1.placeholder(dtype="float16", shape=(1,2,2,2,1))
    grads_2 = tf.compat.v1.placeholder(dtype="float16", shape=(1,2,2,2,1))
    grads = tf.add(grads_1, grads_2)
    orig_input_shape = tf.constant(np.array([1,3,3,3,1]).astype("int32"))
    op = tf.raw_ops.AvgPool3DGrad(orig_input_shape=orig_input_shape,
                                  grad=grads,
                                  ksize=[1,2,2,2,1],
                                  strides=[1,1,1,1,1],
                                  padding="VALID",
                                  data_format='NDHWC',
                                  name='AvgPool3DGrad')

    tf.io.write_graph(sess.graph, logdir="./", name="avgpool3dgrad_case_1.pb", as_text=False)
