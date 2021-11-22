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
dynamic elu
Op_description :
do element-wise elu operation.

# elu(
#   x,
#   y,
#   kernel_name='cce_elu')

Supportive_dtype_format :
["float16", "float32"]
['ND', 'NCHW', 'NHWC', 'NC1HWC0']

Constraint :
[1] All : shape size limit is 2147483648

"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute





def _elu_computer_performance(data, scalar_one_neg):
    negative_data = tbe.vmuls(data, scalar_one_neg)
    negative_data = tbe.vrelu(negative_data)
    negative_data = tbe.vmuls(negative_data, scalar_one_neg)
    positive_data = tbe.vrelu(data)
    return negative_data, positive_data


# 'pylint: disable=invalid-name
def _elu_computer_precision(data, dtype):
    NUM_ZERO = 0.0
    scalar_zero = tvm.const(NUM_ZERO, dtype)
    negative_data = tbe.vmins(data, scalar_zero)
    positive_data = tbe.vmaxs(data, scalar_zero)
    return negative_data, positive_data


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@register_operator_compute("Elu", op_mode="dynamic", support_fusion=True)
def elu_compute(x, y, alpha, kernel_name="elu"):
    """
    do element-wise elu compute
    f(x) = max(min(alpha(e^x - 1), 0), x),  in cloud scene, for all inputs
    f(x) = max(min(alpha(e^x - 1), 0), x),  in mini scene, for x <= TAYLOR_THRESHOLD or x >= 0
    f(x) = fifth taylor computer,    in mini scene, for TAYLOR_THRESHOLD < x < 0

    Parameters:
    ----------
    x: the placeholder of data input

    alpha: float, coefficient when input tensor is less than zero

    y: the dict of output

    kernel_name : cce kernel name, default value is "elu"

    Returns : A Tensor. Has the same type as data_input.
    -------
    """
    NUM_ONE_NEG = -1.0
    data = x
    dtype = data.dtype
    has_improve_precision = False
    if tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        has_improve_precision = True
    if dtype.lower() == "float16" and has_improve_precision:
        data = tbe.cast_to(data, "float32")
        cvt_dtype = "float32"
    else:
        cvt_dtype = dtype

    scalar_one_neg = tvm.const(NUM_ONE_NEG, cvt_dtype)

    if not has_improve_precision and cvt_dtype == "float16":
        _negative_data, _positive_data = _elu_computer_performance(data, scalar_one_neg)
        exp_res = tbe.vexp(_negative_data)
    else:
        _negative_data, _positive_data = _elu_computer_precision(data, cvt_dtype)
        if not tbe_platform.api_check_support("tbe.dsl.vexp", "float32") and cvt_dtype == "float32":
            _negative_data = tbe.cast_to(_negative_data, "float16")
            exp_res = tbe.vexp(_negative_data)
            exp_res = tbe.cast_to(exp_res, "float32")
        else:
            exp_res = tbe.vexp(_negative_data)

    exp_res = tbe.vadds(exp_res, scalar_one_neg)
    res = tbe.vaxpy(exp_res, _positive_data, tvm.const(alpha, cvt_dtype))

    if dtype != cvt_dtype:
        res = tbe.cast_to(res, "float16")

    return res


# 'pylint: disable=invalid-name
@register_operator("Elu")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def elu(x, y, alpha=1.0, kernel_name="elu"):
    """
    do element-wise elu operation

    Parameters:
    ----------
    x: the dict of input, only support float16, float32

    alpha: float, coefficient when input tensor is less than zero.

    output_res : the dict of output

    kernel_name : cce kernel name, default value is "elu"

    Returns
    -------
    None
    """
    input_dtype = x.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="x")

    schedules, tensors = [], []
    ins = classify([x], OpPatternMode.ELEWISE)
    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(x_shape[0], dtype=input_dtype,
                                         name="data_input")
            res = elu_compute(data_input, y, alpha, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
