import tensorflow as tf

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    input = tf.compat.v1.placeholder(dtype="float32", shape=(10,20,10,20,10))
    tf.raw_ops.MaxPool3D(
        input=input, ksize=[1,20,10,20,1], strides=[1,20,10,20,1], padding="SAME", 
        data_format='NDHWC', name="MaxPool3D"
    )
    tf.io.write_graph(sess.graph, logdir="./", name="maxPool3D_case_1.pb", as_text=False)