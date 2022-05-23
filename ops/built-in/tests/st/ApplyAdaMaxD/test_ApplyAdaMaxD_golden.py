import tensorflow as tf
from tensorflow.python.training import gen_training_ops


def calc_expect_func(var, m, v, beta1_power, lr, beta1, beta2, epsilon,
                     grad, var_out, m_out, v_out):
    var = var.get("value")
    m = m.get("value")
    v = v.get("value")
    beta1_power = beta1_power.get("value")[0]
    lr = lr.get("value")[0]
    beta1 = beta1.get("value")[0]
    beta2 = beta2.get("value")[0]
    epsilon = epsilon.get("value")[0]
    grad = grad.get("value")

    var_holder = tf.Variable(var)
    m_holder = tf.Variable(m)
    v_holder = tf.Variable(v)
    beta1_power_holder = tf.constant(beta1_power)
    lr_holder = tf.constant(lr)
    beta1_holder = tf.constant(beta1)
    beta2_holder = tf.constant(beta2)
    epsilon_holder = tf.constant(epsilon)
    grad_holder = tf.placeholder(grad.dtype, shape=grad.shape)

    out = gen_training_ops.apply_ada_max(var_holder, m_holder, v_holder,
                                         beta1_power_holder,
                                         lr_holder, beta1_holder,
                                         beta2_holder, epsilon_holder,
                                         grad_holder, use_locking=False)
    with tf.Session() as sess:
        res = sess.run(out, feed_dict={grad_holder: grad})
    return [res]