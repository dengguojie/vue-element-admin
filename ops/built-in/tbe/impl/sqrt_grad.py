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
sqrt_grad
"""
import functools
import operator

import te.platform as tbe_platform
from te import tvm
from te.lang import cce as tbe
from te.utils import para_check


# pylint: disable=too-many-arguments,unused-argument,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("sqrt_grad")
def sqrt_grad_compute(x, dx, out, kernel_name="sqrt_grad"):
    """
    algorithm: sqrt_grad_compute
    output_grad = input_grad/(2*input)

    Parameters
    ----------
    x: a tensor of input data

    dx : a tensor of grad

    Returns
    -------
    output data

    """

    dtype = x.dtype.lower()
    mul_support = tbe_platform.api_check_support("te.lang.cce.vmuls", "float32")
    if dtype == "float32" and not mul_support:
        raise RuntimeError("Input dtype only support float16 while input dtype is float32")
    const_val_2 = tvm.const(2.0, dtype)
    mul_val = tbe.vmuls(x, const_val_2)
    res = tbe.vdiv(dx, mul_val)
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def sqrt_grad(x, dx, out, kernel_name="sqrt_grad"):
    """
    algorithm: sqrt_grad_cce

    Parameters
    ----------
    x : dict of data: dict

    dx : dict of data_grad: dict

    out : dict of output: dict

    kernel_name : cce kernel name, default value is "sqrt_grad": str

    Returns
    -------
    None

    """

    shape_x = x.get("shape")
    shape_dx = dx.get("shape")
    dtype_x = x.get("dtype").lower()
    dtype_dx = dx.get("dtype").lower()
    if not operator.eq(list(shape_x), list(shape_dx)):
        raise RuntimeError("Input shapes must be equal")
    if not dtype_x == dtype_dx:
        raise RuntimeError("Input dtype must be same")

    para_check.check_shape(shape_x, param_name="x")
    para_check.check_dtype(dtype_x, ("float16", "float32"), param_name="x")

    shape_x = [functools.reduce(lambda x, y: x * y, shape_x[:])]
    data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_x)
    data_dx = tvm.placeholder(shape_x, name="data_dx", dtype=dtype_x)
    with tvm.target.cce():
        res = sqrt_grad_compute(data_x, data_dx, kernel_name)
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": (data_x, data_dx, res)}

    tbe.cce_build_code(sch, config)
