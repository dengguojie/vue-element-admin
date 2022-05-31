import tensorflow as tf
class Constant:
    NUM_TWO = 2
    NUM_ONE = 1


def _cosh_taylor_compute(data):
    taylor_second = 0.5
    taylor_fourth = 1 / 24.0
    taylor_sixth = 1 / 720.0
    # x^2 / 6!
    pow_2 = tf.multiply(data, data)
    pow_2_div = pow_2 * taylor_sixth

    # 1/4! + x^2 / 6!
    pow_2_plus = pow_2_div + taylor_fourth

    # 1/2! + x^2( 1/4! + x^2/6!)
    pow_4 = tf.multiply(pow_2_plus, pow_2)
    pow_4_plus = pow_4 + taylor_second

    # 1 + x^2( 1/2! + x^2( 1/4! + x^2/6!))
    pow_6 = tf.multiply(pow_4_plus, pow_2)
    res = pow_6 + Constant.NUM_ONE
    return res


def _cosh_repeat(data):
    num_minus_one = -1
    data_square = tf.multiply(data, data)
    data_mul = data_square * Constant.NUM_TWO
    res = data_mul + num_minus_one
    return res


def asinh_grad_compute(y, dy, z, kernel_name="cce_asinh_grad"):
    y = y.get("value")
    dy = dy.get("value")
    num_repeat = 0.125
    dtype = y.dtype
    # if tbe_platform.api_check_support('tbe.dsl.vexp', 'float32'):
    #     # use vexp,vdiv api for high efficiency computation
    #     # `cosh(y) = (e^y + e^-y) / 2`
    #     #           (e^2y + 1) / 2e^y
    #     exp_pos = tbe.vexp(y)
    #     res = tbe.vmul(exp_pos, exp_pos)
    #     res = tbe.vadds(res, tvm.const(Constant.NUM_ONE, y.dtype))
    #     data_dy1 = tbe.vmuls(dy, tvm.const(Constant.NUM_TWO, y.dtype))
    #     data_dy1 = tbe.vmul(data_dy1, exp_pos)
    #     res = tbe.vdiv(data_dy1, res)
    # else:
        # use taylor's method for high accuracy result
    y = y * num_repeat
    cosh_value_0 = _cosh_taylor_compute(y)
    # repeat 3 times
    cosh_value_1 = _cosh_repeat(cosh_value_0)
    cosh_value_2 = _cosh_repeat(cosh_value_1)
    cosh_value = _cosh_repeat(cosh_value_2)
    res = tf.reciprocal(cosh_value)
    res = tf.multiply(res, dy)
    with tf.Session() as sess:
        res = sess.run(res)
    return res