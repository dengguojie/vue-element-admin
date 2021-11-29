# Copyright 2019 Huawei Technologies Co., Ltd
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
"""
import operator

import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector


# 'pylint: disable=too=few-public-methods
class Constant:
    """
    the class for constant.
    """
    NUM_MINUS_ONE = -1
    NUM_TWO = 2
    NUM_ONE = 1
    NUM_REPEAT = 0.125
    TAYLOR_SECOND = 0.5
    TAYLOR_FOURTH = 1 / 24.0
    TAYLOR_SIXTH = 1 / 720.0


# 'pylint: disable=too-many-locals
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
    pow_2_div = tbe.vmuls(pow_2, tvm.const(Constant.TAYLOR_SIXTH, data.dtype))

    # 1/4! + x^2 / 6!
    pow_2_plus = tbe.vadds(pow_2_div, tvm.const(Constant.TAYLOR_FOURTH, data.dtype))

    # 1/2! + x^2( 1/4! + x^2/6!)
    pow_4 = tbe.vmul(pow_2_plus, pow_2)
    pow_4_plus = tbe.vadds(pow_4, tvm.const(Constant.TAYLOR_SECOND, data.dtype))

    # 1 + x^2( 1/2! + x^2( 1/4! + x^2/6!))
    pow_6 = tbe.vmul(pow_4_plus, pow_2)
    res = tbe.vadds(pow_6, tvm.const(Constant.NUM_ONE, data.dtype))

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
    data_mul = tbe.vmuls(data_square, tvm.const(Constant.NUM_TWO, data.dtype))
    res = tbe.vadds(data_mul, tvm.const(Constant.NUM_MINUS_ONE, data.dtype))

    return res


# 'pylint: disable=unused-argument,invalid-name
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
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd", "float32"):
        y = tbe.cast_to(y, "float32")
        dy = tbe.cast_to(dy, "float32")

    if tbe_platform.cce_conf.api_check_support('te.lang.cce.vexp', 'float32'):
        exp_pos = tbe.vexp(y)
        res = tbe.vmul(exp_pos, exp_pos)
        res = tbe.vadds(res, tvm.const(Constant.NUM_ONE, y.dtype))
        data_dy1 = tbe.vmuls(dy, tvm.const(Constant.NUM_TWO, y.dtype))
        data_dy1 = tbe.vmul(data_dy1, exp_pos)
        res = tbe.vdiv(data_dy1, res)
    else:
        # use taylor's method for high accuracy result
        y = tbe.vmuls(y, tvm.const(Constant.NUM_REPEAT, y.dtype))
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


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
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
    dtype_y = y.get("dtype")
    dtype_dy = dy.get("dtype")

    # kernel name check: should be unique

    # check whether the shape is right
    para_check.check_shape(shape_y, param_name="y")
    para_check.check_shape(shape_dy, param_name="dy")
    if not operator.eq(shape_y, shape_dy):
        error_detail = "shape of y and dy should be same"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "y", "dy", error_detail)
    shape_y, _ = shape_util.refine_shape_axes(shape_y, [])
    shape_dy, _ = shape_util.refine_shape_axes(shape_dy, [])

    # check whether dtypes are fp16,fp32 and whether they are the same
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_y, check_list, param_name="y")
    para_check.check_dtype(dtype_dy, check_list, param_name="dy")
    dtype_y = dtype_y.lower()
    if dtype_y != dtype_dy.lower():
        error_detail = "dtype of y and dy should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "y", "dy", error_detail)

    # get 2 input tensors: data_y, data_dy
    data_y = tvm.placeholder(shape_y, name="data_y", dtype=dtype_y)
    data_dy = tvm.placeholder(shape_y, name="data_dy", dtype=dtype_y)
    res = asinh_grad_compute(data_y, data_dy, z, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_y, data_dy, res]}
    tbe.cce_build_code(sch, config)
