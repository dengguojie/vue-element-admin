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
bitwise_xor
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=unused-argument,invalid-name
@register_operator_compute("BitwiseXor", op_mode="dynamic", support_fusion=False)
def bitwise_xor_compute(x1, x2, y, kernel_name="bitwise_xor"):
    """
    calculating data's bitwise xor
    (x&y)|!(x|y)

    Parameters
    ----------
    x1 : tvm tensor
              input data x
    x2 : tvm tensor
              input data y
    y : dict
               the shape and dtype of the tensor
    kernel_name : string
                  cce kernel name, default value is "bitwise_and"

    Returns
    -------
    result : y of the data's bitwise xor
    """
    shape_x = shape_util.shape_to_list(x1.shape)
    shape_y = shape_util.shape_to_list(x2.shape)
    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x,
                                                              shape_y,
                                                              param_name_input1="x1",
                                                              param_name_input2="x2")

    data_x = tbe.broadcast(x1, shape_max)
    data_y = tbe.broadcast(x2, shape_max)

    data_and = tbe.vand(data_x, data_y)
    data_not = tbe.vnot(data_and)
    data_or = tbe.vor(data_x, data_y)
    res = tbe.vand(data_or, data_not)

    return res


def _pre_broadcast(x1, x2, y, kernel_name):
    """
    check the input parameters
    make len of shape and range same
    add dims in shape and range
    return x1, x2 after pre broadcast and dtype
    """
    shape_x = list(x1.get("shape"))
    shape_y = list(x2.get("shape"))
    range_x = list(x1.get("range"))
    range_y = list(x2.get("range"))
    dtype_x = x1.get("dtype").lower()
    dtype_y = x2.get("dtype").lower()
    dtype_z = y.get("dtype").lower()

    x1["shape"] = shape_x
    x2["shape"] = shape_y
    x1["range"] = range_x
    x2["range"] = range_y

    check_tuple = ("int16", "uint16", "int32")
    para_check.check_dtype(dtype_x, check_tuple, param_name="x1")
    para_check.check_dtype(dtype_y, check_tuple, param_name="x2")
    para_check.check_dtype(dtype_z, check_tuple, param_name="y")
    if dtype_x != dtype_y:
        error_detail = "dtype of x1 and x2 should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x1", "x2", error_detail)

    if len(shape_y) > len(shape_x):
        x1["range"] = [(1, 1)] * (len(shape_y) - len(shape_x)) + range_x
        x1["shape"] = [1] * (len(shape_y) - len(shape_x)) + shape_x
    elif len(shape_x) > len(shape_y):
        x2["range"] = [(1, 1)] * (len(shape_x) - len(shape_y)) + range_y
        x2["shape"] = [1] * (len(shape_x) - len(shape_y)) + shape_y
    else:
        pass

    if dtype_x == "int32":
        dtype_x = "int16"
        x1["dtype"] = "int16"
        x2["dtype"] = "int16"
        x1["range"].append((2, 2))
        x2["range"].append((2, 2))
        x1["shape"] = x1["shape"] + [2]
        x2["shape"] = x2["shape"] + [2]


    return x1, x2, dtype_x


# 'pylint: disable=too-many-locals
@register_operator("BitwiseXor")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def bitwise_xor(x1, x2, y, kernel_name="bitwise_xor"):
    """
    algorithm: bitwise_xor
    computes the bitwise xor of `x1` and `x2`

    Parameters
    ----------
    x1 : dict
              the shape and dtype of the tensor x1, only support int16,uint16
    x2 : dict
              the shape and dtype of the tensor x2, only support int16,uint16
    y : dict
              the shape and dtype of the tensor y, only support int16,uint16
    kernel_name : string
                  cce kernel name, default value is "bitwise_xor"

    Returns
    -------
    None
    """
    input_x, input_y, dtype = _pre_broadcast(x1, x2, y, kernel_name)
    schedules, tensors = [], []
    extra_params = {"disable_optimization": True}
    ins = classify([input_x, input_y], OpPatternMode.ELEWISE_WITH_BROADCAST, extra_params)
    for (_x, _y) in ins:
        with tbe.compute():
            x_shape, y_shape = shape_util.variable_shape([_x, _y])
            data_x = tvm.placeholder(x_shape, name="data_x", dtype=dtype)
            data_y = tvm.placeholder(y_shape, name="data_y", dtype=dtype)
            res = bitwise_xor_compute(data_x, data_y, y, kernel_name)
            tensors.append([data_x, data_y, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name": kernel_name,
        "tensor_list": tensors
    }
    tbe.build(schedules, config)
