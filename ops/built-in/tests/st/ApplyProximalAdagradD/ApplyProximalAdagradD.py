import tensorflow as tf
from tensorflow.python.training import gen_training_ops


def apply_proximal_adagrad_d_compute(var, accum, lr, l1, l2, grad, var_out,
                                     accum_out, use_locking=False,
                                     kernel_name="apply_proximal_adagrad_d"):
    Var = tf.Variable(var.get("value"))
    Accum = tf.Variable(accum.get("value"))
    output = gen_training_ops.apply_proximal_adagrad(Var, Accum, lr.get("value")[0], l1.get("value")[0], l2.get("value")[0], grad.get("value"))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(output)
    return [res]