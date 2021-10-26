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
hard_sigmoid
"""
import te.lang.cce as tbe
from te import tvm
from te.utils import para_check
from te.platform.fusion_manager import fusion_manager
import te.platform as tbe_platform


# 'pylint: disable=unused-argument,too-many-locals
@fusion_manager.register("hard_sigmoid")
def hard_sigmoid_compute(input_x, output_y, alpha, beta, kernel_name="hard_sigmoid"):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "hard_sigmoid"

    Returnst
    -------
    output tensor
    """
    dtype = input_x.dtype
    shape = input_x.shape
    mul_support_fp32 = tbe_platform.api_check_support("te.lang.cce.vmuls", "float32")
    if dtype != "float32" and mul_support_fp32:
        input_x = tbe.cast_to(input_x, "float32")
    elif dtype != "float16" and not mul_support_fp32:
        cast_support_f322f16 = tbe_platform.api_check_support("te.lang.cce.cast_to", "f322f16")
        cast_support_s322f16 = tbe_platform.api_check_support("te.lang.cce.cast_to", "s322f16")
        if cast_support_f322f16 and dtype == "float32" or  cast_support_s322f16 and dtype == "int32":
            input_x = tbe.cast_to(input_x, "float16")
        else:
            raise RuntimeError("Type of input x must be float16")


    alpha_x = tbe.vmuls(input_x, tvm.const(alpha, input_x.dtype))
    alpha_x_beta = tbe.vadds(alpha_x, tvm.const(beta, input_x.dtype))

    vcmpsel_support_fp32 = tbe_platform.api_check_support("te.lang.cce.vcmpsel", "float32")
    if alpha_x_beta.dtype != "float32" and vcmpsel_support_fp32:
        alpha_x_beta = tbe.cast_to(alpha_x_beta, "float32")
    elif alpha_x_beta.dtype != "float16" and not vcmpsel_support_fp32:
        cast_support_f322f16 = tbe_platform.api_check_support("te.lang.cce.cast_to", "f322f16")
        if cast_support_f322f16:
            alpha_x_beta = tbe.cast_to(alpha_x_beta, "float16")
        else:
            raise RuntimeError("Type of input x must be float16")

    one_tensor = tbe.broadcast(tvm.const(1, alpha_x_beta.dtype), shape)
    zero_tensor = tbe.broadcast(tvm.const(0, alpha_x_beta.dtype), shape)
    result1 = tbe.vcmpsel(alpha_x_beta, one_tensor, 'ge', one_tensor, alpha_x_beta)
    result = tbe.vcmpsel(result1, zero_tensor, 'ge', result1, zero_tensor)
    if dtype != result.dtype:
        result = tbe.cast_to(result, dtype)
    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def hard_sigmoid(input_x, output_y, alpha=0.16666666, beta=0.5, kernel_name="hard_sigmoid"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "hard_sigmoid"

    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()
    para_check.check_shape(input_shape, param_name="input_x")
    check_tuple = ("int32", "float16", "float32")
    para_check.check_dtype(input_dtype, check_tuple, param_name="input_x")

    input_data = tvm.placeholder(input_shape, name="input_x", dtype=input_dtype)
    res = hard_sigmoid_compute(input_data, output_y, alpha, beta, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_data, res]}

    tbe.cce_build_code(schedule, config)
