import tensorflow as tf
import numpy as np
import os

def calc_expect_func(input, k,padding_value,diagonal, align):
    # res = tf.less(x1['value'], x2['value'])
    tf.compat.v1.disable_eager_execution()
    data_input= tf.convert_to_tensor(input['value'])
    data_k= tf.convert_to_tensor(k['value'])
    data_padding_value= tf.convert_to_tensor(padding_value['value'])
    res = tf.raw_ops.MatrixDiagPartV3(input=data_input, k=data_k,padding_value=data_padding_value[0], align=align)
    with tf.compat.v1.Session() as sess:
        res = sess.run(res)
    return [res,]
