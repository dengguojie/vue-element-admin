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
sqrt
"""
import functools

import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from te import tvm

# shape limit for aicore equals 2**31
SHAPE_SIZE_LIMIT = 2147483648


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("sqrt")
def sqrt_compute(input_data, output_data, kernel_name="sqrt"):
    """
    calculating data sqrt,y= x**0.5,mini not support vsqrt, use exp(0.5*log(x))

    Parameters
    ----------
    input_data: TVM tensor
        the placeholder of input data
    output_data: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is sqrt

    Returns
    -------
    result: TVM tensor
        the result of sqrt
    """
    dtype = input_data.dtype
    has_improve_precision = False
    if dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vsqrt", "float32"):
        input_data = tbe.cast_to(input_data, "float32")
        has_improve_precision = True
    result = tbe.vsqrt(input_data)

    if has_improve_precision:
        result = tbe.cast_to(result, "float16")

    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sqrt(input_x, output_y, kernel_name="sqrt"):
    """
    algorithm: sqrt
    calculating data sqrt,y= x**0.5, mini not support vsqrt, use exp(0.5*log(x))

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is sqrt

    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()

    para_check.check_shape(input_shape, param_name="input_x")
    para_check.check_dtype(input_dtype, ("float16", "float32"), param_name="input_x")

    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x*y, input_shape)
    input_data = tvm.placeholder(fuseshape, name="input_data",
                                 dtype=input_dtype)
    result = sqrt_compute(input_data, output_y, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(result)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [input_data, result]}

    tbe.cce_build_code(sch, config)
