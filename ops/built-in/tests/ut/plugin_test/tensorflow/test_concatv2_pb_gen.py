import tensorflow as tf
import os
from tensorflow.python.ops import gen_array_ops

pb_file_path = os.getcwd()
with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    x = tf.compat.v1.placeholder(dtype="int32", shape=(2,2))
    y = tf.compat.v1.placeholder(dtype="int32", shape=(2,2))
    concatv2_dim = 0
    op = gen_array_ops.concat_v2([x, y], concatv2_dim, name='concatv2')
    tf.io.write_graph(sess.graph, logdir="./", name="concatv2.pb", as_text=False)
