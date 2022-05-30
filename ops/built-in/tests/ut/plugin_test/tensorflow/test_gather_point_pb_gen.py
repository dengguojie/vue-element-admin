import tensorflow as tf

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    data1 = tf.compat.v1.placeholder(shape=(2,2,3), dtype=tf.float32)
    data2 = tf.compat.v1.placeholder(shape=(2,3), dtype=tf.int32)
    op = tf.gather(params=data1, indices=data2, axis=1, batch_dims=1, name="GatherPoint")

    tf.io.write_graph(sess.graph, logdir="./", name="gatherpoint_case_1.pb", as_text=False)
