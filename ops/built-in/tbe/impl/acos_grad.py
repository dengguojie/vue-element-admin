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
acos_grad

  Op_description :
    Computes gradients for Acos operation

    # acos_grad(
    #   y,
    #   dy,
    #   z,
    #   kernel_name="acos_grad")

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

# newton eqation is x1 = x0(3-a*(x0^2))/2
NUM_MINUS_ONE = -1
NUM_ONE = 1


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("acos_grad")
def acos_grad_compute(y, dy, z, kernel_name="acos_grad"):
    """
    do acos_grad compute with sqrt and div
    Parameters:
    ----------------
    y: input tensor y
    dy: input tensor dy
    z: output dict
    kernel_name: cce kernel name, default value is "acos_grad"
    return: dy * (- 1 / (1 - data_y^2)^1/2)
    ----------------
    """

    dtype = y.dtype
    dtype_1 = dtype
    if dtype == "float16" and \
       tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd", "float32"):
        y = tbe.cast_to(y, "float32")
        dy = tbe.cast_to(dy, "float32")
        dtype = "float32"

    data1_square = tbe.vmul(y, y)
    data1_square = tbe.vmuls(data1_square, tvm.const(NUM_MINUS_ONE, dtype=dtype))
    data1_square = tbe.vadds(data1_square, tvm.const(NUM_ONE, dtype=dtype))

    data1_reciprocal = tbe.vsqrt(data1_square, 1)
    data1_reciprocal = tbe.vdiv(dy, data1_reciprocal)
    res = tbe.vmuls(data1_reciprocal, tvm.const(NUM_MINUS_ONE, dtype=dtype))

    if dtype_1 == "float16":
        res = tbe.cast_to(res, "float16")
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def acos_grad(y, dy, z, kernel_name="acos_grad"):
    """
    do element-wise acos_grad operation between two input tensors

    Parameters:
    ----------
    y : dict of y, include shape and dtype, dtype support float16, float32

    dy : dict of dy, include shape and dtype, dtype support float16, float32

    z : dict of z, include shape and dtype, dtype support float16, float32

    kernel_name : cce kernel name, default value is "acos_grad"
    -------
    """

    # get the shape and dtype for input_1,input_2
    shape_y = y.get("shape")
    shape_dy = dy.get("shape")
    dtype = y.get("dtype")
    dtype1 = dy.get("dtype")

    para_check.check_shape(shape_y, param_name="y")
    para_check.check_shape(shape_dy, param_name="dy")
    shape_y, _ = shape_util.refine_shape_axes(shape_y, [])
    shape_dy, _ = shape_util.refine_shape_axes(shape_dy, [])

    # raise runtimeerror if the input paras are invalid
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype, check_list, param_name="y")
    para_check.check_dtype(dtype1, check_list, param_name="dy")
    dtype = dtype.lower()
    dtype1 = dtype1.lower()
    if not operator.eq(shape_y, shape_dy):
        raise RuntimeError(
            "acos_grad only support input shape while input_shape1 equals"
            " to input_shape2")
    if dtype != dtype1:
        raise RuntimeError(
            "acos_grad only support dtype while input_dtype1 equals"
            " to input_dtype2")
    shape_y, _ = shape_util.refine_shape_axes(shape_y, [])
    shape_dy, _ = shape_util.refine_shape_axes(shape_dy, [])

    data_y = tvm.placeholder(shape_y, dtype=dtype, name="data1")
    data_dy = tvm.placeholder(shape_dy, dtype=dtype, name="data2")

    res = acos_grad_compute(data_y, data_dy, z, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": (data_y, data_dy, res)}
    tbe.cce_build_code(sch, config)
