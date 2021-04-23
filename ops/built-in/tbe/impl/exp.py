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
exp
"""
import math
import functools

import te.lang.cce as tbe
from te import tvm
import te.platform as tbe_platform
from te.utils import para_check
from impl.util.platform_adapter import error_manager_vector


# pylint: disable=too-many-locals
def isclose(valuex, valuey, rel_tol=1e-08, abs_tol=0.0):
    """
    determines whether the values of two floating-point numbers are close or equal
    """
    return math.isclose(valuex, valuey, rel_tol=rel_tol, abs_tol=abs_tol)

# pylint: disable=locally-disabled,unused-argument,too-many-arguments
@tbe_platform.fusion_manager.fusion_manager.register("exp")
def exp_compute(input_x, output_y, base=-1.0, scale=1.0, shift=0.0, kernel_name="exp"):
    """
    algorithm: exp
    calculating data's exp
    if base == -1:
       y = exp(shift + scale * x)
    if base > 0:
       y = exp((shift+scale*x)*ln(base))

    Parameters
    ----------
    input_x : TVM tensor, the placeholders of input data
    output_y : dict, shape and dtype of output, should be same shape and type as input
    base: (optional, default -1 for a value of e the base gamma
    scale: (optional, default 1) the scale alpha
    shift: (optional, default 0) the shift beta
    kernel_name : str, kernel name, default value is "exp"

    Returns
    -------
    res : the result of compute
    """
    input_x_dtype = input_x.dtype
    if isclose(scale, 1.0) and isclose(shift, 0.0):
        input_x_vadds = input_x
    else:
        scale_const = tvm.const(scale, dtype=input_x_dtype)
        shift_const = tvm.const(shift, dtype=input_x_dtype)
        input_x_vmuls = tbe.vmuls(input_x, scale_const)
        input_x_vadds = tbe.vadds(input_x_vmuls, shift_const)
    if base > 0:
        base_const = tvm.const(math.log(base), dtype=input_x_dtype)
        input_x_bases = tbe.vmuls(input_x_vadds, base_const)
        res = tbe.vexp(input_x_bases)

    # base is -1 value
    else:
        res = tbe.vexp(input_x_vadds)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def exp(input_x, output_y, base=-1.0, scale=1.0, shift=0.0, kernel_name="exp"):
    """
    algorithm: exp
        calculating data's exp
    if base == -1:
       y = exp(shift + scale * x)
    if base > 0:
       y = exp((shift+scale*x)*ln(base))

    Parameters
    ----------
    input_x : dict,shape and dtype of input, only support float16,float32
    output_y: dict,shape and dtype of output, should be same shape and type as input
    base: (optional, default -1 for a value of e the base gamma
    scale: (optional, default 1) the scale alpha
    shift: (optional, default 0) the shift beta
    kernel_name : str, kernel name, default value is "exp"

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")

    para_check.check_shape(shape, param_name="input_x")

    # input_x' dtype check, only supports fp16 and fp32
    check_list = ("float16", "float32")
    input_dtype = dtype.lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    if base <= 0 and (not isclose(base, -1.0)):
        error_manager_vector.raise_err_input_value_invalid("exp", "base", "strictly positive or -1", base)
    
    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x*y, shape)
    data_input = tvm.placeholder(fuseshape, name="data_input", dtype=input_dtype)

    res = exp_compute(data_input, output_y, base, scale, shift, kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    tbe.cce_build_code(sch, config)
