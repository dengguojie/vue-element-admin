import tensorflow as tf
from tensorflow.python.training import gen_training_ops


def calc_expect_func(var, mg, ms, mom, lr, rho, momentum, epsilon,
                     grad, var_out, mg_out, ms_out, mom_out):
    var = var.get("value")
    mg = mg.get("value")
    ms = ms.get("value")
    mom = mom.get("value")
    lr = lr.get("value")[0]
    rho = rho.get("value")[0]
    momentum = momentum.get("value")[0]
    epsilon = epsilon.get("value")[0]
    grad = grad.get("value")

    var_holder = tf.Variable(var)
    mg_holder = tf.Variable(mg)
    ms_holder = tf.Variable(ms)
    mom_holder = tf.Variable(mom)
    lr_holder = tf.constant(lr)
    rho_holder = tf.constant(rho)
    momentum_holder = tf.constant(momentum)
    epsilon_holder = tf.constant(epsilon)
    grad_holder = tf.placeholder(grad.dtype, shape=grad.shape)

    out = gen_training_ops.apply_centered_rms_prop(var_holder, mg_holder, ms_holder, mom_holder,
                                                   lr_holder, rho_holder, momentum_holder,
                                                   epsilon_holder, grad_holder, use_locking=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(out, feed_dict={grad_holder: grad})
    return [res]