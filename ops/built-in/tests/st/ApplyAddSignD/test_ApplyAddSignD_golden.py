import tensorflow as tf
from tensorflow.python.training import gen_training_ops


def calc_expect_func(var, m, lr, alpha, sign_decay, beta, grad,
                     var_out, m_out):
    var = var.get("value")
    m = m.get("value")
    lr = lr.get("value")[0]
    alpha = alpha.get("value")[0]
    sign_decay = sign_decay.get("value")[0]
    beta = beta.get("value")[0]
    grad = grad.get("value")

    var_holder = tf.Variable(var)
    m_holder = tf.Variable(m)
    lr_holder = tf.constant(lr)
    alpha_holder = tf.constant(alpha)
    sign_decay_holder = tf.constant(sign_decay)
    beta_holder = tf.constant(beta)
    grad_holder = tf.placeholder(grad.dtype, shape=grad.shape)

    out = gen_training_ops.apply_add_sign(var_holder, m_holder, lr_holder,
                                          alpha_holder, sign_decay_holder,
                                          beta_holder, grad_holder, use_locking=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(out, feed_dict={grad_holder: grad})
    return [res]