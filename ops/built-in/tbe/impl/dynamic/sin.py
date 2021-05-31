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
sin
"""

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector

# define a string name of "float16"
FLOAT_16 = "float16"
# define a string name of "float32"
FLOAT_32 = "float32"

PI = 3.14159265358979

# the first factor to use Taylor series in circle
FIRST_ORDER = 5
# the last factor to use Taylor series in circle
LAST_ORDER = 13
# the first factor of Taylor series
FIRST_FACTOR = -1.0 / 6.0


# pylint: disable=invalid-name
def _sin(x):
    """
    algorithm: sin
    calculating data's sin x = x-x^3/3!+x ^5/5!-x^7/7!+x^9/9!-x^11/11! (-pai/2 < x < pai/2)

    Parameters
    ----------
    x : TVM tensor
        the placeholders of input data

    Returns
    -------
    res : the res of sin
    """
    input_x_power = tbe.vmul(x, x)
    iter_value = tbe.vmul(tbe.vmuls(input_x_power, FIRST_FACTOR), x)
    res = tbe.vadd(x, iter_value)

    i = FIRST_ORDER
    while i < LAST_ORDER:
        iter_value = tbe.vmuls(tbe.vmul(input_x_power, iter_value), -1.0 / (i * (i - 1)))
        res = tbe.vadd(res, iter_value)

        # add 2 to get the next order
        i = i + 2

    return res


# pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-locals
def sin_compute(x, y, kernel_name="sin"):
    """
    algorithm: sin
    calculating data's sin x = x - x^3/3! + x ^5/5! + ... + (-1)^k*x^2(k+1)/(2(k+1))!

    Parameters
    ----------
    x : TVM tensor
        the placeholders of input data
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is "sin"

    Returns
    -------
    res : the res of sin
    """
    dtype = x.dtype
    shape = shape_util.shape_to_list(x.shape)

    has_improve_precision = False
    cast_dtype = dtype
    if tbe_platform.api_check_support("tbe.dsl.vmul", "float32"):
        has_improve_precision = True
        cast_dtype = FLOAT_32

    # cast to type float32 when type is float16
    if dtype == FLOAT_16 and has_improve_precision:
        x = tbe.cast_to(x, FLOAT_32)

    pai_multiple = tbe.vmuls(x, 1 / PI)
    # pai_round = tbe.round(pai_multiple)
    if not tbe_platform.api_check_support("tbe.dsl.round", "float32") and cast_dtype == FLOAT_32:
        pai_16 = tbe.cast_to(pai_multiple, FLOAT_16)
        round_float = tbe.cast_to(tbe.round(pai_16), cast_dtype)
    else:
        round_float = tbe.cast_to(tbe.round(pai_multiple), cast_dtype)
    # to adjust x to [-pai/2,pai/2]
    x = tbe.vsub(x, tbe.vmuls(round_float, PI))

    res = _sin(x)

    # if round is odd, the final result need to mutiply -1.Need to multipy 1/2 to get the ceil value
    ran_ = tbe.vmuls(round_float, 1 / 2)
    if not tbe_platform.api_check_support("tbe.dsl.ceil", "float32") and cast_dtype == FLOAT_32:
        ran_16 = tbe.cast_to(ran_, FLOAT_16)
        ceil_value = tbe.ceil(ran_16)
        ceil_value = tbe.cast_to(ceil_value, cast_dtype)
    else:
        ceil_value = tbe.ceil(ran_)
    # if odd, ceil*2-round is 1,if even, the value is 0
    tmp = tbe.cast_to(tbe.vmuls(ceil_value, tvm.const(2, dtype)), cast_dtype)
    sub_value = tbe.vsub(tmp, round_float)
    # sub_value = tbe.vsub(tbe.vmuls(ceil_value, tvm.const(2, dtype)), round_float)

    tensor_one = tbe.broadcast(tvm.const(1, cast_dtype), shape)
    odd_tensor = tbe.vsub(tensor_one, sub_value)
    even_tensor = tbe.vsub(odd_tensor, tensor_one)
    odd_even_tensor = tbe.vadd(odd_tensor, even_tensor)
    res = tbe.vmul(res, odd_even_tensor)

    # cast the dtype to float16
    if dtype == FLOAT_16 and has_improve_precision:
        res = tbe.cast_to(res, FLOAT_16)

    return res


@register_operator("Sin")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sin(x, y, kernel_name="sin"):
    """
    algorithm: sin
    calculating data's sin x = x - x^3/3! + x^5/5! + ... + (-1)^k*x^2(k+1)/(2(k+1))!

    Parameters
    ----------
    x: dict
        shape and dtype of input, only support float16, float32
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is "sin"

    Returns
    -------
    None
    """
    # check input x dtypey
    x_dtype = x.get("dtype").lower()
    y_dtype = y.get("dtype").lower()
    check_list = ("float16", "float32", "int32", "int64")
    para_check.check_dtype(x_dtype, check_list, param_name="x")

    # check input x and output y dtype
    if x_dtype != y_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("sin", "x", "y",
                                                              str(x_dtype), str(y_dtype))

    # op compute and schedule
    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_x,) in ins:
        # op compute
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            input_data = tvm.placeholder(x_shape[0], name="input_data", dtype=x_dtype)

            res = sin_compute(input_data, y, kernel_name)
            tensors.append([input_data, res])
        # target auto schedule
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        # append schedule 2 schedules
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
