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
logical_not
"""
import functools

import te.lang.cce as tbe
from te import tvm
from te import platform as tbe_platform
from te.utils import para_check


# 'pylint: disable=locally-disabled,invalid-name,unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("logical_not")
def logical_not_compute(x, y, kernel_name="logical_not"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    y : dict
        dict of y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "logical_not"

    Returns
    -------
    output tensor
    """
    const_one = tvm.const(1.0, "float16")
    shape_x = tbe.util.shape_to_list(x.shape)

    const_broad = tbe.broadcast(const_one, shape_x)
    x_cast = tbe.cast_to(x, "float16", True)
    x_abs = tbe.vabs(x_cast)
    x_min = tbe.vmin(x_abs, const_broad)
    y_sub = tbe.vsub(x_min, const_broad)
    y_abs = tbe.vabs(y_sub)
    res_y = tbe.cast_to(y_abs, x.dtype, True)

    return res_y



@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def logical_not(x, y, kernel_name="logical_not"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input, only support int8, int32
    y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "logical_not"

    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    dtype_x = x.get("dtype").lower()

    para_check.check_shape(shape_x, param_name="x")
    para_check.check_dtype(dtype_x.lower(), ("int8",), param_name="x")

    reshape_x = (functools.reduce(lambda x, y: x*y, shape_x[:]),)
    data = tvm.placeholder(reshape_x, name="data", dtype=dtype_x)
    res = logical_not_compute(data, y, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data, res]}

    tbe.cce_build_code(sch, config)
