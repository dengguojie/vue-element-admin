import tensorflow as tf
import os

pb_file_path = os.getcwd()

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    # NHWC 
    input_size = [17, 101, 101, 17]
    filter_shape = [5, 5, 17, 1]
    dy_shape = [17, 49, 49, 17]
    strideh, stridew = [2, 2]
    padding = 'VALID'
    tensor_filter1 = tf.compat.v1.placeholder(dtype="float16", shape=filter_shape)
    tensor_filter2 = tf.compat.v1.placeholder(dtype="float16", shape=filter_shape)
    tensor_filter = tf.add(tensor_filter1, tensor_filter2)
    tensor_dy1 = tf.compat.v1.placeholder(dtype="float16", shape=dy_shape)
    tensor_dy2 = tf.compat.v1.placeholder(dtype="float16", shape=dy_shape)
    tensor_dy = tf.add(tensor_dy1, tensor_dy2)
    
    op = tf.nn.depthwise_conv2d_backprop_input(input_size, tensor_filter, tensor_dy,
                                  strides=[1, strideh, stridew, 1],
                                  padding=padding,
                                  data_format='NHWC',
                                  dilations=[1,1,1,1])

    tf.io.write_graph(sess.graph, logdir="./", name="depthwiseconv2dbackpropinput_case_1.pb", as_text=False)
