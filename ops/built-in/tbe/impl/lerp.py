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
lerp
"""
import te.lang.cce as tbe
from te import tvm
from te.utils import para_check
from te.platform.fusion_manager import fusion_manager
from te.utils.shape_util import broadcast_shapes, shape_to_list


@fusion_manager.register("lerp")
def lerp_compute(start, end, weight, y, kernel_name="lerp"):
    """
    Compute

    Parameters
    ----------
    start: dict
        data of input
        datatype suports float32,float16
    end: dict
        data of input
        datatype suports float32,float16
    weight: dict
        data of input
        datatype suports float32,float16    
    y: dict
        data of output
    kernel_name: str
        the name of the operator
    Returns
    -------
    None
    """
    shape_x = shape_to_list(start.shape)
    shape_y = shape_to_list(end.shape)
    shape_z = shape_to_list(weight.shape)

    _, _, shape_tmp = broadcast_shapes(shape_x, shape_y, param_name_input1="x", param_name_input2="y")
    _, _, shape_max = broadcast_shapes(shape_tmp, shape_z, param_name_input1="xy", param_name_input2="z")
    para_check.check_shape(shape_max, param_name="shape_max")

    start = tbe.broadcast(start, shape_max)
    end = tbe.broadcast(end, shape_max)
    weight = tbe.broadcast(weight, shape_max)
    tmp = tbe.vsub(end, start)
    res_tmp = tbe.vmul(weight, tmp)
    res = tbe.vadd(start, res_tmp)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def lerp(start, end, weight, y, kernel_name="lerp"):
    """
    Lerp

    Parameters
    ----------
    start: dict
        data of input
        datatype suports float32,float16
    end: dict
        data of input
        datatype suports float32,float16
    weight: dict
        data of input
        datatype suports float32,float16    
    y: dict
        data of output
    kernel_name: str
        the name of the operator
    Returns
    -------
    None
    """
    shape_x = start.get("shape")
    shape_y = end.get("shape")
    shape_z = weight.get("shape")

    input_dtype = start.get("dtype").lower()

    para_check.check_shape_rule(shape_x)
    para_check.check_shape_rule(shape_y)
    para_check.check_shape_rule(shape_z)

    para_check.check_kernel_name(kernel_name)

    check_tuple = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_tuple, param_name="x")

    data_x = tvm.placeholder(shape_x, name="data_1", dtype=input_dtype)
    data_y = tvm.placeholder(shape_y, name="data_2", dtype=input_dtype)
    data_z = tvm.placeholder(shape_z, name="data_3", dtype=input_dtype)

    res = lerp_compute(data_x, data_y, data_z, y, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, data_z, res]}

    tbe.cce_build_code(schedule, config)