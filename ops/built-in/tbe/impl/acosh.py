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
acosh
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.utils import shape_util


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    Constant in this class
    """
    CONST_NEG_ONE = -1.0


# 'pylint: disable=unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("acosh")
def acosh_compute(input_data, output_res, kernel_name="acosh"):
    """
    do element-wise acosh compute
    f(x) = log(x+sqrt(x^2-1)),  for all inputs

    Parameters:
    ----------
    input_data: the placeholder of data input

    output_res : the dict of output

    kernel_name : cce kernel name, default value is "acosh"

    Returns : A Tensor. Has the same type as input_data.
    -------
    """
    data = input_data

    input_dtype = data.dtype.lower()
    if input_dtype == "float16" and tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd", "float32"):
        data = tbe.cast_to(data, "float32")

    res = tbe.vmul(data, data)
    res = tbe.vadds(res, tvm.const(Constant.CONST_NEG_ONE, data.dtype))
    res = tbe.vsqrt(res, 1)
    res = tbe.vadd(res, data)
    res = tbe.vlog(res, 1)

    if input_dtype == "float16":
        res = tbe.cast_to(res, "float16")

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def acosh(input_data, output_res, kernel_name="acosh"):
    """
    calculating data's acosh,y= log(x+sqrt(x^(2)-1))

    Parameters
    ----------
    input_data: the dict of input, only support float16, float32

    output_res : the dict of output

    kernel_name : cce kernel name, default value is "cce_acosh"

    Returns
    -------
    None

    """

    shape_input = input_data.get("shape")
    dtype_input = input_data.get("dtype")
    para_check.check_shape(shape_input, param_name="input_data")

    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_input, check_list, param_name="input_data")
    shape_input, _ = shape_util.refine_shape_axes(shape_input, [])

    input_dtype = dtype_input.lower()
    data = tvm.placeholder(shape_input, dtype=input_dtype, name="data_input")

    res = acosh_compute(data, output_res, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": (data, res),
              "bool_storage_as_1bit": False}

    tbe.cce_build_code(sch, config)
