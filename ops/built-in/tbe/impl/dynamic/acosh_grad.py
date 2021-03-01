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
dynamic acosh_grad

  Op_description :
    Computes gradients for Acosh operation

    # acosh_grad(
    #   y,
    #   dy,
    #   z,
    #   kernel_name="cce_acosh_grad")

  Supportive_dtype_format :
    ['float16', 'float32']
    ['ALL']

  Constraint :
    [1] All : 'y' and 'dy' must have the same type and shape.
    [2] All : shape size limit is 2147483648.
"""
# pylint: disable=invalid-name,too-many-locals
import operator

from impl.util.platform_adapter import tbe
import te.platform as tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
NUM_ONE = 1
NUM_TWO = 2
NUM_REPEAT = 0.125

TAYLOR_SECOND_ORDER_PARAM = 0.1666666666666666666666666666666666
TAYLOR_THIRD_ORDER_PARAM = 0.0083333333333333333333333333333333
TAYLOR_FOURTH_ORDER_PARAM = 0.0001984126984126984126984126984126


def _taylor_sinh_compute(input_data):
    """
    do taylor sinh compute
    Parameters:
    ----------------
    input_data: input tensor
    return: sinh result
    ----------------
    """

    # x^2 / 7!
    data_power_2 = tbe.vmul(input_data, input_data)
    data_power_res = tbe.vmuls(
        data_power_2,
        tvm.const(TAYLOR_FOURTH_ORDER_PARAM, input_data.dtype))

    # 1/5! + x^2 / 7!
    data_power_res = tbe.vadds(
        data_power_res,
        tvm.const(TAYLOR_THIRD_ORDER_PARAM, input_data.dtype))

    # 1/3! + x^2( 1/5! + x^2/7!)
    data_power_res = tbe.vmul(data_power_res, data_power_2)
    data_power_res = tbe.vadds(
        data_power_res,
        tvm.const(TAYLOR_SECOND_ORDER_PARAM, input_data.dtype))

    # 1 + x^2( 1/3! + x^2(1/5! + x^2/7!))
    data_power_res = tbe.vmul(data_power_res, data_power_2)

    data_power_res = tbe.vadds(data_power_res, \
                               tvm.const(NUM_ONE, input_data.dtype))

    # x * (1 + x^2( 1/3! + x^2(1/5! + x^2/7!)))
    data_power_res = tbe.vmul(data_power_res, input_data)
    return data_power_res


def _sinh_repeat_with_sqrt(data):
    """
    do sinh convert compute with sqrt
    Calculate f(2x) = 2f(x) * (f(x)^2 + 1)^0.5
    Parameters:
    ----------------
    data: input tensor
    return: sinh repeat result
    ----------------
    """

    data_square = tbe.vmul(data, data)
    data_square = tbe.vadds(data_square, tvm.const(NUM_ONE,
                                                   data.dtype))

    data_square = tbe.vsqrt(data_square, 1)

    data_square = tbe.vmul(data_square, data)
    data_square = tbe.vmuls(data_square, tvm.const(NUM_TWO,
                                                   data.dtype))

    return data_square


# pylint: disable=unused-argument
@register_operator_compute("AcoshGrad", op_mode="dynamic", support_fusion=False)
def acosh_grad_compute(y, dy, z, kernel_name="acos_grad"):
    """
    do acosh_grad compute
    Parameters:
    ----------------
    y: input tensor y
    dy: input tensor dy
    z: output dict
    kernel_name: cce kernel name, default value is "acosh_grad"
    return: dy * (1 / sinh(y))
    ----------------
    """

    dtype = y.dtype
    dtype_1 = dtype
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd", "float32"):
        y = tbe.cast_to(y, "float32")
        dy = tbe.cast_to(dy, "float32")
        dtype = "float32"

    data_y = tbe.vmuls(y, tvm.const(NUM_REPEAT, dtype))
    sinh_value_0 = _taylor_sinh_compute(data_y)
    sinh_value_1 = _sinh_repeat_with_sqrt(sinh_value_0)
    sinh_value_2 = _sinh_repeat_with_sqrt(sinh_value_1)
    data_sinh = _sinh_repeat_with_sqrt(sinh_value_2)

    res = tbe.vdiv(dy, data_sinh)

    if dtype_1 == "float16":
        res = tbe.cast_to(res, "float16")
    return res


@register_operator("AcoshGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def acosh_grad(y, dy, z, kernel_name="acosh_grad"):
    """
    do element-wise acosh_grad operation between two input tensors
    Parameters:
    ----------
    y : dict of y, include shape and dtype, dtype support float16, float32
    dy : dict of dy, include shape and dtype, dtype support float16, float32
    z : dict of z
    kernel_name : cce kernel name, default value is "acosh_grad"
    -------
    """

    # get the shape and dtype for input_1,input_2
    shape_y = y.get("shape")
    shape_dy = dy.get("shape")
    dtype = y.get("dtype")
    dtype_dy = dy.get("dtype")

    # raise runtimeerror if the input paras are invalid
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype, check_list, param_name="y")
    para_check.check_dtype(dtype_dy, check_list, param_name="dy")
    para_check.check_elewise_shape_range([y, dy],
                                         support_broadcast=True)
    dtype = dtype.lower()
    dtype_dy = dtype_dy.lower()

    if dtype != dtype_dy:
        error_detail = "dtype of y and dy should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "y", "dy", error_detail)

    ins = classify([y, dy], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (y, dy) in ins:
        with tbe.compute():
            # shape
            shape_x1, shape_x2 = shape_util.variable_shape([y, dy])
            # mul_compute
            data_x1 = tvm.placeholder(shape_x1, dtype=dtype, name="data_x1")
            data_x2 = tvm.placeholder(shape_x2, dtype=dtype_dy, name="data_x2")
            res = acosh_grad_compute(data_x1, data_x2, z, kernel_name)
            tensors.append((data_x1, data_x2, res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
