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
addcmul
"""

import te.lang.cce as tbe
from te import tvm
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from te.utils.shape_util import broadcast_shapes
from te.utils.para_check import REQUIRED_INPUT
from te.utils.para_check import REQUIRED_OUTPUT
from te.utils.para_check import KERNEL_NAME
from te.utils.para_check import check_op_params
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# 'pylint: disable=invalid-name,too-many-arguments,too-many-locals,unused-argument
def op_select_format(input_data, x1, x2, value, y, kernel_name="addcmul"):
    """
    op_select_format
    """
    dtype_list = ["float16", "float32", "int32"]
    dtype_len = len(dtype_list)

    support_format = ["ND"]
    support_dtype = []

    input_data_shape = input_data.get("ori_shape")
    x1_shape = x1.get("ori_shape")
    x2_shape = x2.get("ori_shape")
    input_data_shape = list(shape_util.scalar2tensor_one(input_data_shape))
    x1_shape = list(shape_util.scalar2tensor_one(x1_shape))
    x2_shape = list(shape_util.scalar2tensor_one(x2_shape))

    if input_data_shape == x1_shape == x2_shape:
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
    :param dtype_input: str
    :param dtype_x1: str
    :param dtype_x2: str
    :param dtype_value: str
    :return: none
    """
    check_list = ["float16", "float32", "int32"]

    para_check.check_dtype(dtype_input, check_list)
    para_check.check_dtype(dtype_x1, check_list)
    para_check.check_dtype(dtype_x2, check_list)
    para_check.check_dtype(dtype_value, check_list)

    if dtype_input != dtype_x1 or dtype_input != dtype_x2:
        raise RuntimeError("the dtype of input_data, x1, x2 must be same")

    if dtype_input != dtype_value:
        raise RuntimeError("the dtype of input_data, value must be same")


@tbe_platform.fusion_manager.fusion_manager.register("addcmul")
def addcmul_compute(input_data, x1, x2, value, shape_max, y, kernel_name="addcmul"):
    """
    calculating data's addcmul, y = input_data + value * (x1 * x2)
    :param input_data: TVM tensor
    :param x1: TVM tensor
    :param x2: TVM tensor
    :param value: TVM tensor
    :param shape_max: list
    :param y: dict
    :param kernel_name: str
    :return: TVM tensor
    """
    input_dtype = input_data.dtype.lower()

    input_data = tbe.broadcast(input_data, shape_max)
    x1 = tbe.broadcast(x1, shape_max)
    x2 = tbe.broadcast(x2, shape_max)
    value = tbe.broadcast(value, shape_max)

    vmul_val = tbe.vmul(x1, x2)
    vmul_val2 = tbe.vmul(vmul_val, value)
    res = tbe.vadd(input_data, vmul_val2)

    if res.dtype.lower() != input_dtype:
        res = tbe.cast_to(res, input_dtype)
    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def addcmul(input_data, x1, x2, value, y, kernel_name="addcmul"):
    """
    algorithm: addcmul
    calculating data's addcmul, y = input_data + value * (x1 * x2)

    Parameters
    ----------
    input_data : dict
        shape and dtype of first input, only support float16, float32, int32, int8, uint8
    x1 : dict
        shape and dtype of second input, only support float16, float32, int32, int8, uint8
    x2 : dict
        shape and dtype of third input, only support float16, float32, int32, int8, uint8
    value: dict
        shape and dtype of value, only support float16, float32, int32, int8, uint8
    y: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name : str
        cce kernel name, default value is addcmul

    Returns
    -------
    None
    """
    shape_input = input_data.get("shape")
    shape_x1 = x1.get("shape")
    shape_x2 = x2.get("shape")
    shape_value = value.get("shape")
    dtype_input = input_data.get("dtype").lower()
    dtype_x1 = x1.get("dtype").lower()
    dtype_x2 = x2.get("dtype").lower()
    dtype_value = value.get("dtype").lower()

    shape_input = shape_util.scalar2tensor_one(shape_input)
    shape_x1 = shape_util.scalar2tensor_one(shape_x1)
    shape_x2 = shape_util.scalar2tensor_one(shape_x2)
    shape_value = shape_util.scalar2tensor_one(shape_value)
    para_check.check_shape(shape_input)
    para_check.check_shape(shape_x1)
    para_check.check_shape(shape_x2)

    para_check.check_kernel_name(kernel_name)

    check_op_dtype(dtype_input, dtype_x1, dtype_x2, dtype_value)

    if not para_check.is_scalar(shape_value):
        raise RuntimeError("value should be 0D or 1D tensor")

    shape_x1, shape_x2, shape_max1 = broadcast_shapes(shape_x1, shape_x2)
    shape_input, _, shape_max = broadcast_shapes(shape_input, shape_max1)
    shape_x1, _, _ = broadcast_shapes(shape_x1, shape_max)
    shape_x2, _, _ = broadcast_shapes(shape_x2, shape_max)
    shape_value, _, _ = broadcast_shapes(shape_value, shape_max)
    para_check.check_shape_size(shape_max)

    data_input = tvm.placeholder(shape_input, dtype=dtype_input, name="data_input")
    data_x1 = tvm.placeholder(shape_x1, dtype=dtype_x1, name="data_x1")
    data_x2 = tvm.placeholder(shape_x2, dtype=dtype_x2, name="data_x2")
    data_value = tvm.placeholder(shape_value, dtype=dtype_value, name="data_value")
    res = addcmul_compute(data_input, data_x1, data_x2, data_value, shape_max, y, kernel_name="addcmul")

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    tensor_list = [data_input, data_x1, data_x2, data_value, res]

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    tbe.cce_build_code(schedule, config)
