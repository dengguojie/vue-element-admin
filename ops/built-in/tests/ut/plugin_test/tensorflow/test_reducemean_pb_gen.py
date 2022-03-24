import tensorflow as tf
from tensorflow.python.framework import graph_util
#import os
# from tensorflow.python.framework import graph_util

# pb_file_path = os.getcwd()

with tf.Session(graph=tf.Graph()) as sess:
    x = tf.placeholder(dtype="float16", shape=(5, 3, 2))
    op = tf.math.reduce_mean(x, axis=[1], keepdims=True, name='reduce_mean_1')
    #tf.io.write_graph(sess.graph, logdir="./", name="reducemean_case_1.pb", as_text=False)
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['reduce_mean_1'])
    with tf.gfile.FastGFile('./reducemean_case_1.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())
