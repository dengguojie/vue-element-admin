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
acosh

Op_description :
Computes inverse hyperbolic cosine of x element-wise

# acosh(
#   input_data,
#   output_res,
#   kernel_name="cce_acosh")

Supportive_dtype_format :
['float16', 'float32']
['ALL']

Constraint :
[1] All : shape size limit is 2147483648.
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
@register_operator_compute("Acosh", op_mode="dynamic", support_fusion=True)
def acosh_compute(input_data, output_res, kernel_name="acosh"):
    """
    do element-wise acosh compute
    f(x) = log(x+sqrt(x^2-1)),  for all inputs

    Parameters:
    ----------
    input_data: the placeholder of data input

    output_res : the dict of output

    kernel_name : cce kernel name, default value is "acosh"

    Returns : A Tensor. Has the same type as input_data.
    -------
    """
    data = input_data
    const_neg_one = -1.0

    input_dtype = data.dtype.lower()
    if input_dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        data = tbe.cast_to(data, "float32")

    res = tbe.vmul(data, data)
    res = tbe.vadds(res, tvm.const(const_neg_one, data.dtype))
    res = tbe.vsqrt(res, 1)
    res = tbe.vadd(res, data)
    if res.dtype == "float32" and not tbe_platform.api_check_support("tbe.dsl.vlog", "float32"):
        res = tbe.cast_to(res, "float16")
    res = tbe.vlog(res, 1)

    res = tbe.cast_to(res, input_dtype)

    return res


@register_operator("Acosh")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def acosh(input_data, output_res, kernel_name="acosh"):
    """
    calculating data's acosh,y= log(x+sqrt(x^(2)-1))

    Parameters
    ----------
    input_data: the dict of input, only support float16, float32

    output_res : the dict of output

    kernel_name : cce kernel name, default value is "cce_acosh"

    Returns
    -------
    None

    """
    input_dtype = input_data.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="input_data")
    schedules, tensors = [], []
    ins = classify([input_data], OpPatternMode.ELEWISE)
    for (_input_data,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_input_data])[0]
            data_input = tvm.placeholder(x_shape, dtype=input_dtype, name="data_input")
            res = acosh_compute(data_input, output_res, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "print_ir": False, "tensor_list": tensors, "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
