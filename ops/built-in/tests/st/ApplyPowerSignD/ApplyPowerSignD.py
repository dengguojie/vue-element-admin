import tensorflow as tf
from tensorflow.python.training import gen_training_ops


def apply_power_sign_d_compute(var,
                               m,
                               lr,
                               logbase,
                               sign_decay,
                               beta,
                               grad,
                               var_out,
                               m_out,
                               kernel_name="apply_power_sign_d"):
    Var = tf.Variable(var.get("value"))
    M = tf.Variable(m.get("value"))
    output = gen_training_ops.apply_power_sign(Var, M, lr.get("value")[0], logbase.get("value")[0], sign_decay.get("value")[0], beta.get("value")[0], grad.get("value"))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(output)
    return [res]