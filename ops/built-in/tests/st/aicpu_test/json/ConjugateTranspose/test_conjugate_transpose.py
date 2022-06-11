import tensorflow as tf
import numpy as np
def test_conjugate_transpose(x1,perm1):
    x2=tf.transpose(x1['value'])
    perm2=tf.constant(perm1['value'])
    res=tf.raw_ops.ConjugateTranspose(x=x2,perm=perm2)
    with tf.Session() as sess:
        result=sess.run(res)
    return result
