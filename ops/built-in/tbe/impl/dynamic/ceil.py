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
dynamic ceil
"""
import functools
import te.lang.cce as tbe
from te import tvm
from te.utils import para_check
from te.utils import shape_util
import te.lang.base as tbe_base
from te.lang.base.shape_classifier import Mode
from te.lang.base.shape_classifier import classify


# pylint: disable=unused-argument,too-many-locals,invalid-name
def ceil_compute(input_x, output_x, kernel_name="ceil"):
    """
    ceil compute
    calculating element-wise smallest integer not less than input_x

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    output_y: dict
        dict with keys(shape and dtype) of output_y
    kernel_name: str
        kernel name, default value is "ceil"

    Returns
    -------
    res: TVM tensor
        the result of ceil(input_x)
    """
    res_int32 = tbe.ceil(input_x)
    res = tbe.cast_to(res_int32, input_x.dtype)

    return res


@tbe_base.register_operator("Ceil")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def ceil(input_x, output_x, kernel_name="ceil"):
    """
     algorithm: ceil
    calculating element-wise smallest integer not less than input_x,
    the type of input_x is float16 or float32

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input
    output_y: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "ceil"

    Returns
    -------
    None
    """
    x_dtype = input_x.get("dtype").lower()
    check_list = {"float16", "float32"}
    para_check.check_dtype(x_dtype, check_list, param_name="input_x")

    ins = classify([input_x], Mode.ELEWISE)
    schedules, tensors = [], []
    for (_input_x,) in ins:
        with tbe_base.compute():
            x_shape = shape_util.variable_shape([_input_x])
            fuseshape = [1]
            fuseshape[0] = functools.reduce(lambda x, y: x * y, x_shape[0])
            input_data = tvm.placeholder(fuseshape, name="input_data", dtype=x_dtype)
            res = ceil_compute(input_data, output_x, kernel_name)
            tensors.append([input_data, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
