# Copyright 2020 Huawei Technologies Co., Ltd
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
addcdiv
"""

import te.lang.cce as tbe
from te import tvm
import te.platform as tbe_platform
from te.utils import shape_util
from te.utils import para_check
from te.utils.shape_util import broadcast_shapes
from te.utils.para_check import REQUIRED_INPUT
from te.utils.para_check import REQUIRED_OUTPUT
from te.utils.para_check import KERNEL_NAME
from te.utils.para_check import check_op_params
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.constant_util import SHAPE_SIZE_LIMIT


# 'pylint: disable=invalid-name,too-many-arguments,too-many-locals,unused-argument
def op_select_format(input_data, x1, x2, value, y, kernel_name="addcdiv"):
    """
    op_select_format
    """
    dtype_list = ["float16", "float32"]
    dtype_len = len(dtype_list)

    support_format = ["ND"]
    support_dtype = []

    input_data_shape = input_data.get("ori_shape")
    x1_shape = x1.get("ori_shape")
    x2_shape = x2.get("ori_shape")
    input_data_shape = list(shape_util.scalar2tensor_one(input_data_shape))
    x1_shape = list(shape_util.scalar2tensor_one(x1_shape))
    x2_shape = list(shape_util.scalar2tensor_one(x2_shape))

    if input_data_shape == x1_shape and x1_shape == x2_shape:
        support_format.append("FRACTAL_Z")
        support_format.append("FRACTAL_NZ")
        support_format.append("NC1HWC0")

    for dtype in dtype_list:
        support_dtype.extend([dtype] * len(support_format))

    support_format = support_format * dtype_len
    last_format = ["ND"] * len(support_format)

    input0 = gen_param(classify="input0", name="input_data",
                       datatype=",".join(support_dtype),
                       format=",".join(support_format))
    input1 = gen_param(classify="input1", name="x1",
                       datatype=",".join(support_dtype),
                       format=",".join(support_format))
    input2 = gen_param(classify="input2", name="x2",
                       datatype=",".join(support_dtype),
                       format=",".join(support_format))
    input3 = gen_param(classify="input3", name="value",
                       datatype=",".join(support_dtype),
                       format=",".join(last_format))
    output0 = gen_param(classify="output0", name="y",
                        datatype=",".join(support_dtype),
                        format=",".join(support_format))

    param_list = [input0, input1, input2, input3, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def check_op_dtype(dtype_input, dtype_x1, dtype_x2, dtype_value):
    """
    check_op_dtype
    """
    check_list = ["float16", "float32"]
    value_list = ["float16", "float32", "int32"]
    para_check.check_dtype(dtype_input, check_list)
    para_check.check_dtype(dtype_x1, check_list)
    para_check.check_dtype(dtype_x2, check_list)
    para_check.check_dtype(dtype_value, value_list)

    if dtype_input != dtype_x1 or dtype_input != dtype_x2 \
            or dtype_input != dtype_value:
        raise RuntimeError("the dtype of input_data, x1, x2"
                           " and value must be same")


@tbe_platform.fusion_manager.fusion_manager.register("addcdiv")
def addcdiv_compute(data_input, data_x1, data_x2, value, shape_max, kernel_name="addcdiv"):
    """
    calculating data's addcdiv, y = data_input + value * (data_x1 / data_x2)
    :param data_input: TVM tensor
    :param data_x1: TVM tensor
    :param data_x2: TVM tensor
    :param value: TVM tensor
    :param shape_max: list
    :param y: dict
    :param kernel_name: str
    :return: TVM tensor
    """
    input_dtype = data_input.dtype.lower()
    value_dtype = value.dtype.lower()
    if value_dtype != input_dtype:
        value = tbe.cast_to(value, input_dtype)
    data_input = tbe.broadcast(data_input, shape_max)
    data_x1 = tbe.broadcast(data_x1, shape_max)
    data_x2 = tbe.broadcast(data_x2, shape_max)
    value_val = tbe.broadcast(value, shape_max)
    div_val = tbe.vdiv(data_x1, data_x2)
    mul_val = tbe.vmul(div_val, value_val)
    res = tbe.vadd(data_input, mul_val)
    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def addcdiv(input_data, x1, x2, value, y=None, kernel_name="addcdiv"):
    """
    calculating data's addcdiv, y = data_input + value * (data_x1 / data_x2)
    :param input_data: dict
        shape and dtype of first input, only support float16, float32
    :param x1: dict
        shape and dtype of second input, only support float16, float32
    :param x2: dict
        shape and dtype of third input, only support float16, float32
    :param value: dict
        shape and dtype of value, only support float16, float32, int32
    :param y: dict
    :param kernel_name: str
    """
    shape_input = input_data.get("shape")
    dtype_input = input_data.get("dtype").lower()

    shape_x1 = x1.get("shape")
    dtype_x1 = x1.get("dtype").lower()

    shape_x2 = x2.get("shape")
    dtype_x2 = x2.get("dtype").lower()

    shape_value = value.get("shape")
    dtype_value = value.get("dtype").lower()

    para_check.check_shape_rule(shape_input)
    para_check.check_shape_size(shape_input, SHAPE_SIZE_LIMIT)

    para_check.check_shape_rule(shape_x1)
    para_check.check_shape_size(shape_x1, SHAPE_SIZE_LIMIT)

    para_check.check_shape_rule(shape_x2)
    para_check.check_shape_size(shape_x2, SHAPE_SIZE_LIMIT)

    para_check.check_shape_rule(shape_value)
    para_check.check_shape_size(shape_value, SHAPE_SIZE_LIMIT)

    check_op_dtype(dtype_input, dtype_x1, dtype_x2, dtype_value)

    para_check.check_kernel_name(kernel_name)

    shape_x1, shape_x2, shape_max1 = broadcast_shapes(shape_x1, shape_x2)
    shape_input, _, shape_max = broadcast_shapes(shape_input, shape_max1)
    shape_x1, _, _ = broadcast_shapes(shape_x1, shape_max)
    shape_x2, _, _ = broadcast_shapes(shape_x2, shape_max)
    shape_value, _, _ = broadcast_shapes(shape_value, shape_max)
    para_check.check_shape_size(shape_max)

    data_input = tvm.placeholder(shape_input, name="data_input", dtype=dtype_input)
    data_x1 = tvm.placeholder(shape_x1, name="data_x1", dtype=dtype_x1)
    data_x2 = tvm.placeholder(shape_x2, name="data_x2", dtype=dtype_x2)
    data_value = tvm.placeholder(shape_value, name="data_value", dtype=dtype_value)
    res = addcdiv_compute(data_input, data_x1, data_x2, data_value, shape_max, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, data_x1, data_x2, data_value, res]}

    tbe.cce_build_code(schedule, config)
