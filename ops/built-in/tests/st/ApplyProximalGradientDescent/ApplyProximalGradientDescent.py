import tensorflow as tf
from tensorflow.python.training import gen_training_ops


def apply_proximal_gradient_descent_compute(
        var,
        alpha,
        l1,
        l2,
        delta,
        out,
        kernel_name="apply_proximal_gradient_descent"):
    Var = tf.Variable(var.get("value"))
    output = gen_training_ops.apply_proximal_gradient_descent(Var, alpha.get("value")[0], l1.get("value")[0], l2.get("value")[0], delta.get("value"))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(output)
    return [res]