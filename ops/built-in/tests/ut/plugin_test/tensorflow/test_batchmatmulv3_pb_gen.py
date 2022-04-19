import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import batch_mat_mul_v3
import os
tf.compat.v1.disable_eager_execution()

pb_file_path = os.getcwd()

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    # NHWC
    x1_shape = [75, 152, 345]
    x2_shape = [1113, 152]
    y_shape = [75, 345, 1113]
    adj_x1, adj_x2 = True, True

    x1_1 = tf.compat.v1.placeholder(dtype="float16", shape=x1_shape)
    x1_2 = tf.compat.v1.placeholder(dtype="float16", shape=x1_shape)
    x1 = tf.add(x1_1, x1_2)

    x2_1 = tf.compat.v1.placeholder(dtype="float16", shape=x2_shape)
    x2_2 = tf.compat.v1.placeholder(dtype="float16", shape=x2_shape)
    x2 = tf.add(x2_1, x2_2)

    op = batch_mat_mul_v3(x1, x2, Tout='float32', adj_x=adj_x1, adj_y=adj_x2)

    tf.io.write_graph(sess.graph, logdir="./", name="batchmatmulv3_case_1.pb", as_text=False)
