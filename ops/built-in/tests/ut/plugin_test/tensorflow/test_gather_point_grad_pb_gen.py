import tensorflow as tf

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    a = [-2.25, 3.25]
    data1 = tf.constant(a, shape=[2], dtype=tf.float32)
    input_data1 = tf.Variable(data1)
    data2 = tf.compat.v1.placeholder(shape=(2), dtype=tf.int32)
    data3 = tf.compat.v1.placeholder(shape=(2), dtype=tf.float32)
    op = tf.compat.v1.scatter_update(input_data1, data2, data3, use_locking=True, name="gather_point_grad_1")

    tf.io.write_graph(sess.graph, logdir="./", name="gatherpointgrad_case_1.pb", as_text=False)