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
dynamic minimum
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import error_manager_vector

SHAPE_SIZE_LIMIT = 2147483648  # shape limit


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=unused-variable,invalid-name,too-many-locals
def minimum_compute(x1, x2, y, kernel_name="minimum"):
    """dynamic minimum compute

    Parameters:
    ----------
    x1: TVM tensor
        input_x tensor.
    x2: TVM tensor
        input_y tensor.
    y: dict
        shape and dtype of output.
    kernel_name: str
        cce kernel name, default value is "minimum".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """

    shape_x = shape_util.shape_to_list(x1.shape)
    shape_y = shape_util.shape_to_list(x2.shape)
    shape1, shape2, shape_max = shape_util.broadcast_shapes(shape_x, shape_y,
                                                            param_name_input1="x1",
                                                            param_name_input2="x2")

    data1 = tbe.broadcast(x1, shape_max)
    data2 = tbe.broadcast(x2, shape_max)

    res = tbe.vmin(data1, data2)

    return res


# pylint: disable=redefined-argument-from-local
@register_operator("Minimum")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def minimum(x1, x2, y, kernel_name="minimum"):
    """
    do element-wise minimum operation between two input tensors

    Parameters:
    ----------
    x1 : dict
        first input dict, only support float16, float32, int32
    x2 : dict
        second input dict, only support float16, float32, int32
    y: dict
        output dict, should be the broadcast shape and type as input
    kernel_name : str
        cce kernel name, default value is minimum

    Returns
    -------
    None
    """

    # check input tensor data dtype
    check_list = ["float16", "float32", "int32"]
    dtype_x1 = x1.get("dtype").lower()
    dtype_x2 = x2.get("dtype").lower()
    para_check.check_dtype(dtype_x1, check_list, param_name="x1")
    para_check.check_dtype(dtype_x2, check_list, param_name="x2")
    para_check.check_elewise_shape_range([x1, x2], support_broadcast=True)
    if dtype_x1 != dtype_x2:
        error_manager_vector.raise_err_inputs_dtype_not_equal("minimum", "x1", "x2",
                                                              str(dtype_x1), str(dtype_x2))

    ins = classify([x1, x2], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (x1, x2) in ins:
        with tbe.compute():
            shape_x1, shape_x2 = shape_util.variable_shape([x1, x2])
            data1 = tvm.placeholder(shape_x1, dtype=dtype_x1, name="data1")
            data2 = tvm.placeholder(shape_x2, dtype=dtype_x2, name="data2")
            res = minimum_compute(data1, data2, y, kernel_name)

            tensors.append([data1, data2, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
