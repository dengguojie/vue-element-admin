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
real_div
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from te import tvm


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=unused-argument,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("real_div")
def real_div_compute(x1, x2, y, kernel_name="real_div"):
    """
    calculating data's realdiv, c = a / b

    Parameters
    ----------
    x1: TVM tensor
        the placeholder of first input data
    x2: TVM tensor
        the placeholder of second input data
    y: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is real_div

    Returns
    -------
    res : output of the data's divide
    """
    shape_x = shape_util.shape_to_list(x1.shape)
    shape_y = shape_util.shape_to_list(x2.shape)
    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y, param_name_input1="x1",
                                                              param_name_input2="x2")
    data_x = tbe.broadcast(x1, shape_max)
    data_y = tbe.broadcast(x2, shape_max)
    res = tbe.vdiv(data_x, data_y)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def real_div(x1, x2, y, kernel_name="real_div"):
    """
    algorithm: real_div
    calculating data's real_div, c = a / b

    Parameters
    ----------
    x1 : dict
        shape and dtype of first input, only support float16, float32, int32
    x2 : dict
        shape and dtype of second input, only support float16, float32, int32
    y: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name : str
        cce kernel name, default value is real_div

    Returns
    -------
    None
    """
    shape_x = shape_util.scalar2tensor_one(x1.get("shape"))
    shape_y = shape_util.scalar2tensor_one(x2.get("shape"))
    para_check.check_shape(shape_x, param_name="x1")
    para_check.check_shape(shape_y, param_name="x2")

    check_tuple = ("float16", "float32")
    input_data_type = x1.get("dtype").lower()
    para_check.check_dtype(input_data_type, check_tuple, param_name="x1")
    input_data_type_x2 = x2.get("dtype").lower()
    para_check.check_dtype(input_data_type_x2, check_tuple, param_name="x2")

    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y, param_name_input1="x1",
                                                              param_name_input2="x2")
    if shape_x[-1] == 1 and shape_y[-1] == 1 and shape_max[-1] == 1:
        shape_x = shape_x if len(shape_x) == 1 else shape_x[:-1]
        shape_y = shape_y if len(shape_y) == 1 else shape_y[:-1]
        shape_max = shape_max if len(shape_max) == 1 else shape_max[:-1]

    shape_x, shape_y = shape_util.refine_shapes_for_broadcast(shape_x, shape_y)
    data_x = tvm.placeholder(shape_x, name="data_x", dtype=input_data_type)
    data_y = tvm.placeholder(shape_y, name="data_y", dtype=input_data_type)

    res = real_div_compute(data_x, data_y, y, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": (data_x, data_y, res)}

    tbe.cce_build_code(schedule, config)
