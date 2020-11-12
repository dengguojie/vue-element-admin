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
asin_grad

  Op_description :
    Computes gradients for Asin operation

    # asin_grad(
    #   y,
    #   dy,
    #   z,
    #   kernel_name="cce_asin_grad")

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

# scalar in asin_grad and Newton's equation
NUM_MINUS_ONE = -1
NUM_ONE = 1


# pylint: disable=unused-argument,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("asin_grad")
def asin_grad_compute(y, dy, z, kernel_name="asin_grad"):
    """
    do element-wise asin_grad compute

    Parameters:
    ----------
    y : the placeholders of input y

    dy : the placeholders of input dy

    z : output dict

    kernel_name : cce kernel name, default value is "cce_asin_grad"

    return : dy * (1 / sqrt(1 - y^2))
    -------
    """

    dtype = y.dtype
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd", "float32"):
        y = tbe.cast_to(y, "float32")
        dy = tbe.cast_to(dy, "float32")

    # step 1: calculate num_to_vrsqrt = 1 - y^2
    data = tbe.vmul(y, y)
    data = tbe.vmuls(data, tvm.const(NUM_MINUS_ONE, y.dtype))
    num_to_vrsqrt = tbe.vadds(data, tvm.const(NUM_ONE, y.dtype))

    # step 2: calculate dy * (1 / sqrt(1 - y^2))
    vsqrt_res = tbe.vsqrt(num_to_vrsqrt, 1)
    res = tbe.vdiv(dy, vsqrt_res)

    if dtype == "float16":
        res = tbe.cast_to(res, "float16")

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def asin_grad(y, dy, z, kernel_name="asin_grad"):
    """
    do element-wise asin_grad operation between two input tensors

    Parameters:
    ----------
    y : dict of y, include shape and dtype, dtype support float16, float32

    dy : dict of dy, include shape and dtype, dtype support float16, float32

    z : dict of output

    kernel_name : cce kernel name, default value is "asin_grad"

    Returns
    -------
    None
    """

    # get the shape and dtype
    shape_y = y.get("shape")
    shape_dy = dy.get("shape")
    dtype_y = y.get("dtype")
    dtype_dy = dy.get("dtype")

    # kernel name check: should be unique

    # check whether the shape is right
    para_check.check_shape(shape_y, param_name="y")
    para_check.check_shape(shape_dy, param_name="dy")
    if not operator.eq(shape_y, shape_dy):
        error_detail = "shape of y and dy should be same"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "y", "dy", error_detail)
    shape_y, _ = shape_util.refine_shape_axes(shape_y, [])
    shape_dy, _ = shape_util.refine_shape_axes(shape_dy, [])

    # check whether dtypes are fp16,fp32 and whether they are the same
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_y, check_list, param_name="y")
    para_check.check_dtype(dtype_dy, check_list, param_name="dy")
    dtype_y = dtype_y.lower()
    if dtype_y != dtype_dy.lower():
        error_detail = "dtype of y and dy should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "y", "dy", error_detail)

    # get 2 input tensors: data_y, data_dy
    data_y = tvm.placeholder(shape_y, name="data_y", dtype=dtype_y)
    data_dy = tvm.placeholder(shape_y, name="data_dy", dtype=dtype_y)
    res = asin_grad_compute(data_y, data_dy, z, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_y, data_dy, res]}
    tbe.cce_build_code(sch, config)
