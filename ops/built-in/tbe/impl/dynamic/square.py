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
dynamic square
"""
import functools

import te.lang.cce as tbe
import te.lang.base as tbe_base
from te.utils import para_check
from te.utils import shape_util
from te import tvm


# pylint: disable=unused-argument,redefined-argument-from-local
def square_compute(input_x, output_y, kernel_name="square"):
    """
    algorithm: square
    calculating data's square,y= x*x

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is square

    Returns
    -------
    res : tvm.tensor
        the result of square
    """
    res = tbe.vmul(input_x, input_x)
    return res


@tbe_base.register_operator("Square")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def square(input_x, output, kernel_name="square"):
    """
    algorithm: square
    calculating data's square,y= x*x

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32, int32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "square"

    Returns
    -------
    None
    """

    # check dtype
    x_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32", "int32")
    para_check.check_dtype(x_dtype, check_list, param_name="input_x")

    ins = tbe_base.classify([input_x], tbe_base.Mode.ELEWISE)
    schedules, tensors = [], []
    for (input_x,) in ins:
        with tbe_base.compute():
            # shape
            x_shape = shape_util.variable_shape([input_x])
            fuseshape = [1]
            fuseshape[0] = functools.reduce(lambda x, y: x * y, x_shape[0])
            # square_compute
            data_x = tvm.placeholder(fuseshape, x_dtype, name="data_x")
            res = square_compute(data_x, output, kernel_name)

            tensors.append((data_x, res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
