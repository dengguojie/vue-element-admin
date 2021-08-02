import tensorflow as tf
import os

pb_file_path = os.getcwd()

def generate_case_1():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        input_x = tf.compat.v1.placeholder(dtype="float32", shape=(1,56,56,64))
        input_filter = tf.compat.v1.placeholder(dtype="float32", shape=(3,3,64,64))
        op = tf.nn.conv2d(input_x, input_filter, strides=[1,1,1,1], padding=[[0,0],[1,1],[1,1],[0,0]],
                          data_format="NHWC", dilations=[1,1,1,1], name='conv2d_res')
        tf.io.write_graph(sess.graph, logdir="./", name="conv2d_explicit_pad.pb", as_text=False)

if __name__=='__main__':
    generate_case_1()