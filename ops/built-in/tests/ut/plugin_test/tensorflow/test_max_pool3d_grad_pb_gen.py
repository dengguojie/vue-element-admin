import tensorflow as tf

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    orig_input = tf.compat.v1.placeholder(dtype="float32", shape=(10,20,10,20,10))
    orig_output = tf.compat.v1.placeholder(dtype="float32", shape=(10,20,10,20,10))
    grad = tf.compat.v1.placeholder(dtype="float32", shape=(10,20,10,20,10))
    tf.raw_ops.MaxPool3DGrad(
        orig_input=orig_input, orig_output=orig_output, grad=grad,
        ksize=[1,20,10,20,1], strides=[1,20,10,20,1], padding="SAME",
        data_format='NDHWC', name='MaxPool3DGrad'
    )
    tf.io.write_graph(sess.graph, logdir="./", name="maxPool3DGrad_case_1.pb", as_text=False)