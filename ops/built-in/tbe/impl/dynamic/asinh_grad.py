# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
asinh_grad

  Op_description :
    Computes gradients for Asinh operation

    # asinh_grad(
    #   y,
    #   dy,
    #   z,
    #   kernel_name="cce_asinh_grad")

  Supportive_dtype_format :
    ['float16', 'float32']
    ['ALL']

  Constraint :
    [1] All : 'y' and 'dy' must have the same type and shape.
    [2] All : shape size limit is 2147483648.

"""
from impl.util.platform_adapter import tbe
import te.platform as tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator

# scalar in asinh_grad
NUM_MINUS_ONE = -1
NUM_TWO = 2
NUM_ONE = 1
NUM_REPEAT = 0.125

# scalar 1/2! , 1/4! and 1/6! used in taylor
TAYLOR_SECOND = 0.5
TAYLOR_FOURTH = 1 / 24.0
TAYLOR_SIXTH = 1 / 720.0


def _cosh_taylor_compute(data):
    """
    Calculate cosh  = 1 + x^2( 1/2! + x^2( 1/4! + x^2/6!))

    Parameters:
    ----------
    data : the placeholder of data input

    Returns
    -------
    A Tensor represents cosh(data). Has the same type as data.
    """

    # x^2 / 6!
    pow_2 = tbe.vmul(data, data)
    pow_2_div = tbe.vmuls(pow_2, tvm.const(TAYLOR_SIXTH, data.dtype))

    # 1/4! + x^2 / 6!
    pow_2_plus = tbe.vadds(pow_2_div, tvm.const(TAYLOR_FOURTH, data.dtype))

    # 1/2! + x^2( 1/4! + x^2/6!)
    pow_4 = tbe.vmul(pow_2_plus, pow_2)
    pow_4_plus = tbe.vadds(pow_4, tvm.const(TAYLOR_SECOND, data.dtype))

    # 1 + x^2( 1/2! + x^2( 1/4! + x^2/6!))
    pow_6 = tbe.vmul(pow_4_plus, pow_2)
    res = tbe.vadds(pow_6, tvm.const(NUM_ONE, data.dtype))

    return res


def _cosh_repeat(data):
    """
    Calculate f(2x) = 2f(x)^2 -1

    Parameters:
    ----------
    data : the placeholder of data input

    Returns
    -------
    A Tensor represents f(2x). Has the same type as data.
    """

    data_square = tbe.vmul(data, data)
    data_mul = tbe.vmuls(data_square, tvm.const(NUM_TWO, data.dtype))
    res = tbe.vadds(data_mul, tvm.const(NUM_MINUS_ONE, data.dtype))

    return res


# pylint: disable=unused-argument,invalid-name,too-many-locals
@tbe_platform.fusion_manager.fusion_manager.register("asinh_grad")
def asinh_grad_compute(y, dy, output_res, kernel_name="cce_asinh_grad"):
    """
    do element-wise asinh_grad compute

    Parameters:
    ----------
    y : the placeholders of input y

    dy : the placeholders of input dy

    output_res : output dict

    kernel_name : cce kernel name, default value is "cce_asinh_grad"

    Return :
    -------
    dy * (1/cosh(y))
    """

    dtype = y.dtype
    if dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd", "float32"):
        y = tbe.cast_to(y, "float32")
        dy = tbe.cast_to(dy, "float32")

    if tbe_platform.cce_conf.api_check_support('te.lang.cce.vexp', 'float32'):
        # use vexp,vdiv api for high efficiency computation
        # cosh(y) = (e^y + e^-y) / 2
        #           (e^2y + 1) / 2e^y
        exp_pos = tbe.vexp(y)
        res = tbe.vmul(exp_pos, exp_pos)
        res = tbe.vadds(res, tvm.const(NUM_ONE, y.dtype))
        data_dy1 = tbe.vmuls(dy, tvm.const(NUM_TWO, y.dtype))
        data_dy1 = tbe.vmul(data_dy1, exp_pos)
        res = tbe.vdiv(data_dy1, res)
    else:
        # use taylor's method for high accuracy result
        y = tbe.vmuls(y, tvm.const(NUM_REPEAT, y.dtype))
        cosh_value_0 = _cosh_taylor_compute(y)
        # repeat 3 times
        cosh_value_1 = _cosh_repeat(cosh_value_0)
        cosh_value_2 = _cosh_repeat(cosh_value_1)
        cosh_value = _cosh_repeat(cosh_value_2)
        res = tbe.vrec(cosh_value)
        res = tbe.vmul(res, dy)

    if dtype == "float16":
        res = tbe.cast_to(res, "float16")

    return res


@register_operator("AsinhGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def asinh_grad(y, dy, z, kernel_name="cce_asinh_grad"):
    """
    do element-wise asinh_grad operation between two input tensors

    Parameters:
    ----------
    y : dict of y, include shape and dtype, dtype support float16, float32

    dy : dict of dy, include shape and dtype, dtype support float16, float32

    z : dict of output

    kernel_name : cce kernel name, default value is "cce_asinh_grad"

    Returns
    -------
    None
    """

    # get the shape and dtype
    shape_y = y.get("shape")
    shape_dy = dy.get("shape")
    dtype_y = y.get("dtype").lower()
    dtype_dy = dy.get("dtype").lower()

    # kernel name check: should be unique

    # check whether the shape is right
    para_check.check_shape(shape_y, param_name="y")
    para_check.check_shape(shape_dy, param_name="dy")

    # check whether dtypes are fp16,fp32 and whether they are the same
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_y, check_list, param_name="y")
    para_check.check_dtype(dtype_dy, check_list, param_name="dy")
    if dtype_y != dtype_dy:
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_018
        error_info['op_name'] = 'asinh_grad'
        error_info['param_name1'] = 'dtype_y'
        error_info['param_name2'] = 'dtype_dy'
        error_info['param1_dtype'] = str(dtype_y)
        error_info['param2_dtype'] = str(dtype_dy)
        raise RuntimeError(error_info,
                           "In op[%s], the parameter[%s][%s] are not equal in "
                           "dtype with dtype[%s][%s]." % (
                               error_info['op_name'],
                               error_info['param_name1'],
                               error_info['param_name2'],
                               error_info['param1_dtype'],
                               error_info['param2_dtype']))

    ins = classify([y, dy], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (input_y, input_dy) in ins:
        with tbe.compute():
            shape_y, shape_dy = shape_util.variable_shape([input_y, input_dy])
            data_y = tvm.placeholder(shape_y, name="data_y", dtype=dtype_y)
            data_dy = tvm.placeholder(shape_dy, name="data_dy", dtype=dtype_dy)
            res = asinh_grad_compute(data_y, data_dy, z, kernel_name)

            tensors.append([data_y, data_dy, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
