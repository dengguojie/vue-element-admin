import tensorflow as tf

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    input = tf.compat.v1.placeholder(dtype="float32", shape=(10,20,20,10))
    filter = tf.compat.v1.placeholder(dtype="float32", shape=(10,10,10))
    out_backprop = tf.compat.v1.placeholder(dtype="float32", shape=(10,10,10,10))
    tf.raw_ops.Dilation2DBackpropFilter(
        input=input, filter=filter, out_backprop=out_backprop, strides=[1,10,10,1], 
        rates=[1,10,10,1], padding="SAME", name="Dilation2DBackpropFilter"
    )
    tf.io.write_graph(sess.graph, logdir="./", name="dilation2DBackpropFilter_case_1.pb", as_text=False)