import tensorflow as tf

def pad_v2_tf(x, paddings, constant_values):
    x_holder = tf.placeholder(x.dtype, shape=x.shape)
    re = tf.pad(x_holder, paddings, mode='constant', constant_values=constant_values)
    with tf.Session() as sess:
        result = sess.run(re, feed_dict={x_holder: x})
    return result

def pad_v2(x, paddings, constant_values, y, kernel_name="pad"):
    res = pad_v2_tf(x['value'], paddings['value'], constant_values['value'][0])
    return res