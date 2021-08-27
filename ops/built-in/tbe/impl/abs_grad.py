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
abs_grad

  Op_description :
    Computes gradients for abs operation

    # abs_grad(
    #   y,
    #   dy,
    #   z,
    #   kernel_name="cce_abs_grad")

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

SHAPE_SIZE_LIMIT = 2147483648


@tbe_platform.fusion_manager.fusion_manager.register("abs_grad")
def abs_grad_compute(y, dy, z, kernel_name="abs_grad"):
    """
    do abs_grad compute
    Parameters:
    ----------------
    y: input tensor y
    dy: input tensor dy
    z: output dict
    kernel_name: cce kernel name, default value is "abs_grad"
    return: data_dy * sign(data_y)
    ----------------
    """

    dtype = dy.dtype

    if dtype == "float16":
        fp_max = tvm.const(2 ** 15, dtype)
        fp_min = tvm.const(2 ** (-15), dtype)
    else:
        fp_max = tvm.const(2 ** 62, dtype)
        fp_min = tvm.const(2 ** (-127), dtype)
    new_data = tbe.vmuls(y, fp_max)
    abs_data = tbe.vabs(new_data)
    denominator = tbe.vadds(abs_data, fp_min)
    res = tbe.vdiv(new_data, denominator)
    res = tbe.round(res)
    data1_res = tbe.vmul(res, dy)
    return data1_res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def abs_grad(y, dy, z, kernel_name="abs_grad"):
    """
    do element-wise abs_grad operation between two input tensors

    Parameters:
    ----------
    y : dict of y, include shape and dtype, dtype support float16, float32

    dy : dict of dy, include shape and dtype, dtype support float16, float32

    z : dict of z, include shape and dtype, dtype support float16, float32

    kernel_name : cce kernel name, default value is "abs_grad"
    -------
    """

    # get the shape and dtype for input_1,input_2
    shape_y = y.get("shape")
    shape_dy = dy.get("shape")
    dtype_y = y.get("dtype")
    dtype_dy = dy.get("dtype")

    para_check.check_shape(shape_y, param_name="y")
    para_check.check_shape(shape_dy, param_name="dy")
    shape_y, _ = shape_util.refine_shape_axes(shape_y, [])
    shape_dy, _ = shape_util.refine_shape_axes(shape_dy, [])

    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_y, check_list, param_name="y")
    para_check.check_dtype(dtype_dy, check_list, param_name="dy")
    dtype_y = dtype_y.lower()
    dtype_dy = dtype_dy.lower()
    if not operator.eq(shape_y, shape_dy):
        error_detail = "shape of y and dy should be same"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "y", "dy", error_detail)
    if dtype_y != dtype_dy:
        error_detail = "dtype of y and dy should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "y", "dy", error_detail)
    shape_y, _ = shape_util.refine_shape_axes(shape_y, [])
    shape_dy, _ = shape_util.refine_shape_axes(shape_dy, [])

    data_y = tvm.placeholder(shape_y, dtype=dtype_y, name="data1")
    data_dy = tvm.placeholder(shape_dy, dtype=dtype_dy, name="data2")
    res = abs_grad_compute(data_y, data_dy, z, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_y, data_dy, res]}
    tbe.cce_build_code(sch, config)
