import tensorflow as tf
from tensorflow.python.training import gen_training_ops


def calc_expect_func(var, accum, accum_update, lr, rho, epsilon, grad, var_out,
                     accum_out, accum_update_out):
    var = var.get("value")
    accum = accum.get("value")
    accum_update = accum_update.get("value")
    lr = lr.get("value")[0]
    rho = rho.get("value")[0]
    epsilon = epsilon.get("value")[0]
    grad = grad.get("value")

    var_holder = tf.Variable(var)
    accum_holder = tf.Variable(accum)
    accum_update_holder = tf.Variable(accum_update)
    lr_holder = tf.constant(lr)
    rho_holder = tf.constant(rho)
    epsilon_holder = tf.constant(epsilon)
    grad_holder = tf.placeholder(grad.dtype, shape=grad.shape)

    out = gen_training_ops.apply_adadelta(var_holder, accum_holder, accum_update_holder,
                                          lr_holder, rho_holder, epsilon_holder, grad_holder,
                                          use_locking=False)
                                                  
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(out, feed_dict={grad_holder: grad})
    return [res]