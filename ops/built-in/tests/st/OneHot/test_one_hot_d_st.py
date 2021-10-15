import tensorflow as tf

def calc_expect_func(input_x, input_on_val, input_off_val, output_y, depth, axis):
    res = tf.one_hot(input_x[value], depth, input_on_val["value"][0], input_off_val["value"][0], axis)
    return res.eval(session=tf.compat.v1.session())