# Copyright 2021 hHuawei Technologies Co., Ltd
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
atanh

  Op_description :
    Computes inverse hyperbolic tangent of x element-wise

    # atanh(
    #   x,
    #   y,
    #   kernel_name="atanh_cce")

  Supportive_dtype_format :
    ['float16', 'float32']
    ['ALL']

  Constraint :
    [1] All : shape size limit is 2147483648.
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
import te.lang.base as tbe_base
from te import tvm
from te.utils import para_check
from te.utils import shape_util
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
import functools

# const value
CONST_HALF = 0.5
CONST_ONE = 1
CONST_NEG_ONE = -1


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
def atanh_compute(x, y, kernel_name="atanh"):
    """
    Algrithm : atanh(x) = 0.5 * log((1 + x) / (1 - x)) if abs(x) < 1

    Parameters
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name

    Returns
    -------
    res : result of atanh
    """

    inp_dtype = x.dtype
    shape = x.shape

    if inp_dtype == "float16" and tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd", "float32") \
        and tbe_platform.cce_conf.api_check_support("te.lang.cce.vlog", "float32"):
        x = tbe.cast_to(x, "float32")

    data_res = _compute(x, shape)

    if inp_dtype == "float16":
        data_res = tbe.cast_to(data_res, "float16")
    else:
        data_res = tbe.cast_to(data_res, "float32")

    return data_res


def _compute(data_input, shape):
    """
    Algrithm: atanh(x) = 0.5*log((1+x)/(1-x))

    Parameters
    ----------
    data_input: the placeholder of data input

    shape: the shape of data_input

    Returns
    -------
    data_res :  return of atanh
    """

    data_1_sum_x = tbe.vadds(data_input, tvm.const(CONST_ONE, data_input.dtype))
    data_sub_x = tbe.vmuls(data_input, tvm.const(CONST_NEG_ONE, data_input.dtype))
    data_1_sub_x = tbe.vadds(data_sub_x, tvm.const(CONST_ONE, data_input.dtype))
    data_x_mul = tbe.vdiv(data_1_sum_x, data_1_sub_x)
    data_x_log = tbe.vlog(data_x_mul, 1)
    data_res = tbe.vmuls(data_x_log, tvm.const(CONST_HALF, data_input.dtype))

    return data_res


@tbe_base.register_operator("Atanh")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def atanh(x, y, kernel_name="atanh"):
    """
    Algrithm: atanh(x) = atanh

    Parameters
    ----------
    Algorithm: atanh

    Parameters:

    x: the dict of input data, only support float16, float32.

    y: the dict of output

    kernel_name: cce kernel name, default value is "atanh".

    Returns
    -------
    None
    """
    dtype = x.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype, check_list, param_name="x")
    ins = classify([x], Mode.ELEWISE)
    schedules, tensors = [], []
    for (x,) in ins:
        with tbe_base.compute():
            shape_x = shape_util.variable_shape([x])
            fuseshape = [1]
            fuseshape[0] = functools.reduce(lambda x, y: x * y, shape_x[0])
            input_data = tvm.placeholder(fuseshape, dtype, "input_data")
            res = atanh_compute(input_data, y, kernel_name)
            tensors.append([input_data, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors, "bool-stoage_as_1bit": False}
    tbe.build(schedules, config)
