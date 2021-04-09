import os
import tensorflow as tf


pb_file_path = os.getcwd()

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    grads1 = tf.compat.v1.placeholder(dtype="float32", shape=(2, 2, 2, 2))
    grads2 = tf.compat.v1.placeholder(dtype="float32", shape=(2, 2, 2, 2))
    grads = tf.add(grads1, grads2)
    original_image1 = tf.compat.v1.placeholder(dtype="float32", shape=(2, 2, 2, 2))
    original_image2 = tf.compat.v1.placeholder(dtype="float32", shape=(2, 2, 2, 2))
    original_image = tf.add(original_image1, original_image2)
    op = tf.raw_ops.ResizeBilinearGrad(grads=grads,
                                       original_image=original_image,
                                       align_corners=False,
                                       half_pixel_centers=False,
                                       name=None)

    tf.io.write_graph(sess.graph, logdir="./", name="resizebilineargrad_case_1.pb", as_text=False)
