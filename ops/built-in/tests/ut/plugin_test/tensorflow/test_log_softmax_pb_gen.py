import tensorflow as tf

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    x = tf.compat.v1.placeholder(dtype="float32", shape=(4,2,2,4))
    op = tf.nn.log_softmax(
        x, name='log_softmax'
    )
    tf.io.write_graph(sess.graph, logdir="./", name="log_softmax.pb", as_text=False)
