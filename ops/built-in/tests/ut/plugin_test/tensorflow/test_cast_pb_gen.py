import tensorflow as tf

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    x = tf.compat.v1.placeholder(dtype="float32", shape=(4,2,2,4))
    tf.cast(
        x=x, dtype=tf.int32, name='cast'
    )
    tf.io.write_graph(sess.graph, logdir="./", name="cast_case_1.pb", as_text=False)
