import tensorflow as tf
import os
tf.compat.v1.disable_eager_execution()

pb_file_path = os.getcwd()

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    # NHWC
    x1_shape = [75, 152, 345]
    x2_shape = [1113, 152]
    y_shape = [75, 345, 1113]
    transpose_a, transpose_b = True, True

    x1_1 = tf.compat.v1.placeholder(dtype="float16", shape=x1_shape)
    x1_2 = tf.compat.v1.placeholder(dtype="float16", shape=x1_shape)
    x1 = tf.add(x1_1, x1_2)

    x2_1 = tf.compat.v1.placeholder(dtype="float16", shape=x2_shape)
    x2_2 = tf.compat.v1.placeholder(dtype="float16", shape=x2_shape)
    x2 = tf.add(x2_1, x2_2)

    op = tf.matmul(x1, x2, transpose_a=transpose_a, transpose_b=transpose_b)

    tf.io.write_graph(sess.graph, logdir="./", name="batchmatmulv2_case_1.pb", as_text=False)
