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
round
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
@register_operator_compute("Round", op_mode="dynamic", support_fusion=True)
def round_compute(x, y, kernel_name="round"):
    """
    calculating data round, round to the nearst,tie to the even

    Parameters
    ----------
    x: TVM tensor
        the placeholder of input data
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is round

    Returns
    -------
    result: TVM tensor
        the result of round
    """
    dtype = x.dtype
    if dtype == "int32":
        input_data_one = tbe.broadcast(tvm.const(0, dtype), x.shape, dtype)
        result = tbe.vadd(x, input_data_one)
        return result

    if not tbe_platform.api_check_support("tbe.dsl.round", dtype):
        x = tbe.cast_to(x, "float16")
    result = tbe.round(x)
    result = tbe.cast_to(result, dtype)

    return result


# 'pylint: disable=locally-disabled,redefined-builtin
@register_operator("Round")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def round(x, y, kernel_name="round"):
    """
    algorithm: round
    calculating data round, round to the nearst,tie to the even

    Parameters
    ----------
    x : dict
        shape and dtype of input, only support float16,float32,int32
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is round

    Returns
    -------
    None
    """
    input_dtype = x.get("dtype").lower()
    check_list = ("float16", "float32", "int32")
    para_check.check_dtype(input_dtype, check_list, param_name="x")

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            data_x = tvm.placeholder(x_shape[0], dtype=input_dtype, name="data_x")
            res = round_compute(data_x, y, kernel_name)

            tensors.append([data_x, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
