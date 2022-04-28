import tensorflow as tf
import os
from tensorflow.python.ops import gen_array_ops

pb_file_path = os.getcwd()
with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    x = tf.compat.v1.placeholder(dtype="int32", shape=(2,2))
    y = tf.compat.v1.placeholder(dtype="int32", shape=(2,2))
    concat_dim = 0
    op = gen_array_ops.concat_offset(concat_dim, [x, y], name='concat_offset')
    tf.io.write_graph(sess.graph, logdir="./", name="concat_offset.pb", as_text=False)
