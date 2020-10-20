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
maximum
"""

import te.lang.cce as tbe
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util

SHAPE_SIZE_LIMIT = 2147483648  # shape limit

# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=invalid-name,unused-variable,too-many-locals
@tbe_platform.fusion_manager.fusion_manager.register("maximum")
def maximum_compute(x1, x2, y, kernel_name="maximum"):
    """maximum compute

    Parameters:
    ----------
    x1: TVM tensor
        input_x tensor.
    x2: TVM tensor
        input_y tensor.
    y: dict
        shape and dtype of output.
    kernel_name: str
        cce kernel name, default value is "maximum".

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
    res = tbe.vmax(data1, data2)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def maximum(x1, x2, y, kernel_name="maximum"):
    """
    do element-wise maximum operation between two input tensors

    Parameters:
    ----------
    x1 : dict
        shape and dtype of first input, only support float16, float32, int32
    x2 : dict
        shape and dtype of second input, only support float16, float32, int32
    y: dict
        shape and dtype of output, should be the broadcast shape and
        type as input
    kernel_name : str
        cce kernel name, default value is maximum

    Returns
    -------
    None
    """
    shape1 = shape_util.scalar2tensor_one(x1.get("shape"))
    shape2 = shape_util.scalar2tensor_one(x2.get("shape"))

    if shape1 == shape2:
        shape1, _ = shape_util.refine_shape_axes(shape1, [])
        shape2, _ = shape_util.refine_shape_axes(shape2, [])

    para_check.check_shape(shape1, param_name="x1")
    para_check.check_shape(shape2, param_name="x2")

    check_list = ["float16", "float32", "int32"]
    dtype = x1.get("dtype").lower()
    dtype_x2 = x2.get("dtype").lower()
    para_check.check_dtype(dtype, check_list, param_name="x1")
    para_check.check_dtype(dtype_x2, check_list, param_name="x2")

    shape_x, shape_y, _ = shape_util.broadcast_shapes(shape1, shape2,
                                                      param_name_input1="x1",
                                                      param_name_input2="x2")

    data1 = tvm.placeholder(shape_x, dtype=dtype, name="data1")
    data2 = tvm.placeholder(shape_y, dtype=dtype, name="data2")

    res = maximum_compute(data1, data2, y, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data1, data2, res]}
    tbe.cce_build_code(sch, config)
