import os
import tensorflow as tf
from npu_bridge.npu_cpu.npu_cpu_ops import dense_image_warp


pb_file_path = os.getcwd()

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    image1 = tf.compat.v1.placeholder(dtype="float32", shape=(1, 10, 10, 3))
    image2 = tf.compat.v1.placeholder(dtype="float32", shape=(1, 10, 10, 3))
    flow1 = tf.compat.v1.placeholder(dtype="float32", shape=(1, 10, 10, 2))
    flow2 = tf.compat.v1.placeholder(dtype="float32", shape=(1, 10, 10, 2))
    image = tf.add(image1, image2)
    flow = tf.add(flow1, flow2)
    op = dense_image_warp(image=image, flow=flow)

    tf.io.write_graph(sess.graph, logdir="./", name="denseimagewarp_case_1.pb", as_text=False)
