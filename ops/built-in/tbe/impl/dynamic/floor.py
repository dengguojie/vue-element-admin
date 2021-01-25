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
floor
"""
from functools import reduce as reduceIns
import te.lang.cce as tbe
from te import tvm
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from te.utils import shape_util
from te.utils import para_check
from te import platform as tbe_platform
from te.platform.fusion_manager import fusion_manager
import te.lang.base as tbe_base
from te.utils.error_manager import error_manager_vector
from impl.util.platform_adapter import register_operator


# pylint: disable=locally-disabled,unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("floor")
def floor_compute(input_x, output_y, kernel_name="floor"):
    """
    floor compute
    calculating element-wise largest integer not greater than input_x

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict with keys(shape and dtype) of output
    kernel_name : str
        kernel name, default value is "floor"

    Returns
    -------
    res : TVM tensor
        the result of floor(input_x)
    """
    res_int32 = tbe.floor(input_x)
    res = tbe.cast_to(res_int32, input_x.dtype)

    return res


@register_operator("Floor")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, 
                            para_check.KERNEL_NAME)
def floor(input_x, output_y, kernel_name="floor"):
    """
    algorithm: floor
    calculation element-wise largest integer not greater than input_x,
    the type of input_x is float16 or float32

    Parameters
    ----------
    input_x : dict
        dict with keys(shape and dtype) of input
    output_y : dict
        dict with keys(shape and dtype) of output
    kernel_name : str
        kernel_name, default value is "floor"

    Returns
    -------
    None
    """
    input_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    schedules, tensors = [], []
    ins = classify([input_x], Mode.ELEWISE)
    for (input_x,) in ins:
        with tbe_base.compute():
            x_shape = shape_util.variable_shape([input_x])
            fuseshape = [1]
            fuseshape[0] = reduceIns(lambda x, y: x * y, x_shape[0])
            data_input = tvm.placeholder(fuseshape, dtype=input_dtype, name="data_input")
            res = floor_compute(data_input, output_y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name": kernel_name,
        "tensor_list": tensors,
        "bool_storage_as_1bit": False
    }

    tbe.build(schedules, config) 