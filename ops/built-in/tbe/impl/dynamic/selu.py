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
selu
`if < 0:scale * alpha * (exp(features) - 1)`
`otherwise:`scale * features`
"""
import functools

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# define selu oprator's required constants
ALPHA = 1.67326324235
SCALE = 1.05070098736
# define product of scale and alpha
SCALE_ALPHA_PRODUCT = 1.75809934085
# define a scalar, value = -1, the calculation of exp need minus one
SCALAR_NEGATIVE_ONE = -1


# pylint: disable=locally-disabled,unused-argument,too-many-locals
# pylint: disable=invalid-name
@register_operator_compute("Selu", op_mode="dynamic", support_fusion=True)
def selu_compute(input_x, y, kernel_name="selu"):
    """
    Computes scaled exponential linear: `scale * alpha * (exp(features) - 1)`
    if < 0, `scale * features` otherwise.
    alpha =  1.6732632423543772848170429916717
    scale =  1.0507009873554804934193349852946

    Parameters
    ----------
    input_x: TVM tensor
        input tensor has shape and dtype attributes
    y: TVM tensor
        outputtensor has shape and dtype attributes
    kernel_name : str
        cce kernel name, default value is "selu"

    Returns
    ------
    res: TVM tensor
        the calculation results
    """
    # if input_dtype is float16,convert it to float32
    input_data = input_x
    dtype = input_data.dtype
    if dtype in ("float16", "float32"):
        input_data = tbe.cast_to(input_data, "float32")
        type_tmp = "float32"
    else:
        input_data = tbe.cast_to(input_data, "float16")
        type_tmp = "float16"

    # generate tensor_zero to be compared
    tensor_zero = tbe.vmuls(input_data, tvm.const(0, dtype=type_tmp))
    # generate negative_res and positive_res to compute
    # When the element value is greater than 0 and less than 0
    negative_res = tbe.vmin(input_data, tensor_zero)
    positive_res = tbe.vmax(input_data, tensor_zero)
    exp_res = tbe.vexp(negative_res)
    sub_res = tbe.vadds(exp_res, tvm.const(SCALAR_NEGATIVE_ONE, dtype=type_tmp))
    negative_muls_res = tbe.vmuls(sub_res, tvm.const(SCALE_ALPHA_PRODUCT, dtype=type_tmp))
    if dtype == "int8":
        negative_muls_res = tbe.ceil(negative_muls_res)

    positive_muls_res = tbe.vmuls(positive_res, tvm.const(SCALE, dtype=type_tmp))
    negative_muls_res = tbe.cast_to(negative_muls_res, positive_muls_res.dtype.lower())
    res = tbe.vadd(negative_muls_res, positive_muls_res)
    # if input_dtype is float16, has converted to float32,
    # output should convert back
    if dtype in ("float16", "int8", "int32"):
        res = tbe.cast_to(res, dtype)

    return res


@register_operator("Selu")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def selu(x, y, kernel_name="selu"):
    """
    Generate selu_cce operator use selu_compute

    Parameters
    ----------
    x: dict
        dict{"shape":tuple or list,"dtype":str}
        shape of data, assume src_shape equals dst_shape,
        the data type, src_dtype equals dst_dtype,
         support fp16, fp32, int8, int32
    y: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        cce kernel name, default value is "selu"

    Returns
    ------
    None
    """
    # get dtype and shape attributes
    dtype = x.get("dtype")
    # check_kernel_name & shape
    input_dtype = dtype.lower()
    # check input tensor data_type
    check_list = ("float16", "float32", "int8", "int32")
    para_check.check_dtype(input_dtype, check_list, param_name="x")
    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            fuse_shape = [1]
            fuse_shape[0] = functools.reduce(lambda x, y: x * y, x_shape[0])
            data_input = tvm.placeholder(fuse_shape, name="data_input", dtype=dtype)
            res = selu_compute(data_input, y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
