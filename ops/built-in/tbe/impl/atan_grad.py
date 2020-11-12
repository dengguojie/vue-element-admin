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
atan_grad

  Op_description :
    Computes gradients for Atan operation

    # atan_grad(
    #   y,
    #   dy,
    #   z,
    #   kernel_name="cce_atan_grad")

  Supportive_dtype_format :
    ['float16', 'float32']
    ['ALL']

  Constraint :
    [1] All : 'y' and 'dy' must have the same type and shape.
    [2] All : shape size limit is 2147483648.
"""
import operator

import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector

CONST_ONE = 1


# pylint: disable=unused-argument,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("atan_grad")
def atan_grad_compute(y, dy, z, kernel_name="atan_grad"):
    """
    Calculation for backward gradient

    Parameters:
    ----------
    y: the placeholder of input data
    dy: the placeholder of input dy
    output_z : dict of output
    kernel_name : cce kernel name, default value is atan_grad

    Algorithm :
    ----------
        res = 1/(1+y^2)*dy

    Returns
    ----------
    result res
    """

    scalar_one = tvm.const(CONST_ONE, "float32")
    dtype = y.dtype

    if dtype == "float16" and tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd", "float32"):
        y = tbe.cast_to(y, "float32")
        dy = tbe.cast_to(dy, "float32")

    data_square = tbe.vmul(y, y)
    sum_tmp = tbe.vadds(data_square, scalar_one)
    res = tbe.vdiv(dy, sum_tmp)

    if dtype == "float16":
        res = tbe.cast_to(res, "float16")

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def atan_grad(y, dy, z, kernel_name="atan_grad"):
    """
    Gradient calculation for atan(x)

    Parameters:
    ----------
    y : dict of y, include shape and dtype, dtype support float16, float32
    dy : dict of dy, include shape and dtype, dtype support float16, float32
    z : dict of output, include shape and dtype
    kernel_name : cce kernel name, default value is atan_grad

    Algorithm :
    ----------
    forward :
        y = atan(x)
    backward gradient :
        de/dx = dy/dx*de/dy = 1/(1+x^2)*grad

    Returns
    ----------
    None
    """

    # get the shape and dtype
    shape = y.get("shape")
    shape_grad = dy.get("shape")
    dtype = y.get("dtype")
    dtype_grad = dy.get("dtype")

    # check whether kernel name is unique

    # check whether the shape is right
    para_check.check_shape(shape, param_name="y")
    para_check.check_shape(shape_grad, param_name="dy")
    if not operator.eq(shape, shape_grad):
        error_detail = "shape of y and dy should be same"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "y", "dy", error_detail)
    shape, _ = shape_util.refine_shape_axes(shape, [])

    # check whether dtypes are fp16,fp32 and whether they are the same
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype, check_list, param_name="y")
    para_check.check_dtype(dtype_grad, check_list, param_name="dy")
    dtype = dtype.lower()
    if dtype != dtype_grad.lower():
        error_detail = "dtype of y and dy should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "y", "dy", error_detail)

    # get 2 input placeholders: data_input, grad
    data_input = tvm.placeholder(shape, name="input_data", dtype=dtype)
    grad = tvm.placeholder(shape, name="input_grad", dtype=dtype)

    # compute the backward gradient
    res = atan_grad_compute(data_input, grad, z, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, grad, res]}
    tbe.cce_build_code(sch, config)
