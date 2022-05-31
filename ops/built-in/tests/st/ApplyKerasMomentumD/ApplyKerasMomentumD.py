import tensorflow as tf


def apply_keras_momentum_d_compute(var,
                           accum,
                           lr,
                           grad,
                           momentum,
                           out_var,
                           out_accum,
                           use_locking=False,
                           use_nesterov=False,
                           kernel_name="apply_keras_momentum_d"):
    var = var.get("value")
    accum = accum.get("value")
    grad = grad.get("value")
    momentum = momentum.get("value")
    lr = lr.get("value")
    # update var and accum according to the momentum scheme
    # `accum = accum * momentum - grad * lr`
    accum_momen = tf.multiply(accum, momentum)
    grad_lr = tf.multiply(grad, lr)
    out_accum = tf.subtract(accum_momen, grad_lr)

    # `var = var + accum * momentum - grad * lr`
    if use_nesterov:
        accum_momen2 = tf.multiply(out_accum, momentum)
        add_var_am = tf.add(var, accum_momen2)
        out_var = tf.subtract(add_var_am, grad_lr)
    # `var = var + accum`
    else:
        out_var = tf.add(var, out_accum)

    with tf.Session() as sess:
        out_var = sess.run(out_var)
        out_accum = sess.run(out_accum)

    res = [out_var, out_accum]

    return res