import tensorflow as tf
def fill_tf(dims, value):
    dims_holder = tf.placeholder(dims.dtype, shape=dims.shape)
    re = tf.fill(dims_holder, value)
    with tf.Session() as sess:
        result = sess.run(re, feed_dict={dims_holder: dims})
    return result
def fill(dims, value, y, kernel_name="fill"):
    dims_value = dims['value']
    value_value = value['value']
    res = fill_tf(dims_value, value_value[0])
    return res