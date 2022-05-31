import tensorflow as tf
from tensorflow.python.training import gen_training_ops

def apply_ftrl_d_compute(var,
                         accum,
                         linear,
                         grad,
                         lr,
                         l1,
                         l2,
                         lr_power,
                         var_out,
                         accum_out,
                         linear_out,
                         kernel_name='apply_ftrl_d'):
    Var = tf.Variable(var.get("value"))
    Accum = tf.Variable(accum.get("value"))
    Linear = tf.Variable(linear.get("value"))
    output = gen_training_ops.apply_ftrl(Var, Accum, Linear, grad.get("value"), lr.get("value")[0], l1.get("value")[0], l2.get("value")[0], lr_power.get("value")[0])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(output)
    return [res]
