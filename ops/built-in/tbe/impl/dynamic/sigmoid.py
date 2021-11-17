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
dynamic sigmoid
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpImplMode


# 'pylint: disable=unused-argument,too-many-locals,invalid-name
@register_operator_compute("Sigmoid", op_mode="dynamic", support_fusion=True)
def sigmoid_compute(x, y, kernel_name="sigmoid"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    y : dict
        dict of y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "sigmoid"

    Returns
    -------
    output tensor
    """
    data_input = x
    dtype = x.dtype
    exp_support = tbe_platform.api_check_support("tbe.dsl.vexp", "float32")
    mul_support = tbe_platform.api_check_support("tbe.dsl.vmuls", "float32")
    if dtype == "float32" and not mul_support:
        error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, 'x', ("float16",), dtype)

    const_num_neg_one = tvm.const(-1, dtype=dtype)
    const_num_one = tvm.const(1, dtype=dtype)
    tmp_negative = tbe.vmuls(data_input, const_num_neg_one)
    if dtype == "float32" and not exp_support:
        tmp_negative_16 = tbe.cast_to(tmp_negative, "float16")
        tmp_exp_16 = tbe.vexp(tmp_negative_16)
        tmp_exp = tbe.cast_to(tmp_exp_16, dtype)
    else:
        tmp_exp = tbe.vexp(tmp_negative)
    tmp_sum = tbe.vadds(tmp_exp, const_num_one)
    if dtype == "float32":
        inp_shape = tmp_sum.shape
        tensor_one = tbe.broadcast(tvm.const(1, dtype), inp_shape)
        tmp_rec = tbe.vdiv(tensor_one, tmp_sum)
    else:
        tmp_rec = tbe.vrec(tmp_sum, OpImplMode.HIGH_PRECISION)

    return tmp_rec


@register_operator("Sigmoid")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sigmoid(x, y, kernel_name="sigmoid"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        dict of x, include keys(shape and dtype)
    y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "sigmoid"

    Returns
    -------
    None
    """
    dtype = x.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype, check_list, param_name="x")

    schedules, tensors = [], []
    ins = classify([x], OpPatternMode.ELEWISE)
    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(x_shape[0], dtype=dtype,
                                         name="dtype")
            res = sigmoid_compute(data_input, y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
