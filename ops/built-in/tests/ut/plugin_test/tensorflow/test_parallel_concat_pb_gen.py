import tensorflow as tf

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    input0 = tf.compat.v1.placeholder(dtype="float32", shape=(1,4))
    tf.raw_ops.ParallelConcat(
        values=[input0], shape=[1,4], name="ParallelConcat"
    )
    tf.io.write_graph(sess.graph, logdir="./", name="parallelConcat_case_1.pb", as_text=False)