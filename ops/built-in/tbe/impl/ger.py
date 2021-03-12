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
ger
"""
import te.lang.cce as tbe

from te import tvm
from te import platform as tbe_platform
from te.utils import para_check
from te.platform.fusion_manager import fusion_manager


# General limitation of the reduce size for input shape: 2***31
SHAPE_SIZE_LIMIT = 2147483648


# pylint: disable=invalid-name, unused-argument
@fusion_manager.register("ger")
def ger_compute(data_x, data_vec2, y, kernel_name="ger"):
    """
    algorithm: ger

    Parameters
    ----------
    data_x: TVM tensor
        the placeholder of input x
    data_vec2: TVM tensor
        the placeholder of input vec2
    y: dict
        shape and dtype of output
    kernel_name: str
        cce kernel name, default value is "ger"

    Returns
    -------
    res : output of the datas' ger
    """
    shape_x = tbe.util.shape_to_list(data_x.shape)
    shape_vec2 = tbe.util.shape_to_list(data_vec2.shape)
    shape_common = []
    shape_common.append(shape_x[0])
    shape_common.append(shape_vec2[0])
    broa_x = tbe.broadcast(data_x, shape_common)
    broa_vec2 = tbe.broadcast(data_vec2, shape_common)

    res = tbe.vmul(broa_x, broa_vec2)
    return res


# pylint: disable=invalid-name, unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, 
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def ger(x, vec2, y, kernel_name="ger"):
    """
    calculate the outer product of x and vec2. If x is a vector of size n and 
    vec2 is a vector of size m, then y must be a matrix of size (n*m)

    Parameters
    ----------
    x : dict
        shape and dtype of first input, only support float16, float32
    vec2 : dict
        shape and dtype of second input, only support float16, float32
    y: dict
        shape and dtype of output
    kernel_name : str
        cce kernel name, default value is "ger"

    Returns
    -------
    None
    """
    # obtain operator information
    shape_x = x.get("shape")
    shape_vec2 = vec2.get("shape")
    data_type_x = x.get("dtype").lower()
    data_type_vec2 = vec2.get("dtype").lower()
    check_tuple = ("float16", "float32")

    # operator check
    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(shape_x)
    para_check.check_shape_rule(shape_vec2)
    para_check.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)
    para_check.check_shape_size(shape_vec2, SHAPE_SIZE_LIMIT)
    para_check.check_dtype_rule(data_type_x, check_tuple)
    para_check.check_dtype_rule(data_type_vec2, check_tuple)

    # tensor placeholder
    shape_broa_x = []
    shape_broa_x.append(shape_x[0])
    shape_broa_x.append(1)
    data_x = tvm.placeholder(shape_broa_x, name="data_x", dtype=data_type_x)
    data_vec2 = tvm.placeholder(shape_vec2, name="data_vec2", dtype=data_type_vec2)

    # ger compute function
    res = ger_compute(data_x, data_vec2, y, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    # compile configuration
    config = {"print_ir": False, 
              "name": kernel_name,
              "tensor_list": [data_x, data_vec2, res]}
    tbe.cce_build_code(schedule, config)
