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
asin

  Op_description :
    Computes acos of x element-wise

    # asin(
    #   x,
    #   y,
    #   kernel_name="cce_asin")

  Supportive_dtype_format :
    ['float16', 'float32']
    ['ALL']

  Constraint :
    [1] All : shape size limit is 2147483648.
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from impl.util import util_compute
from te.utils import para_check
from te.utils import shape_util

NUM_ONE = 1.0
NEG_NUM_ONE = -1.0

HALF_PI = 1.5707963267948966192313216916398

BOUNDARY_1 = 0.70710678118654752440084436210485

# Taylor coefficient
COEF = (1.0,
        0.16666666666666666666666666666667,
        0.075,
        0.04464285714285714285714285714286,
        0.03038194444444444444444444444444,
        0.02237215909090909090909090909091,
        0.01735276442307692307692307692308,
        0.01396484375)

# TAYLOR COUNT
TAYLOR_COUNT = 7


def _taylor_compute(data_x, x_square=None):
    """
    do arcsinx compute use the 15th order taylor expansion when 0 <= x <= BOUNDARY_1
    asin(x) = x + 1/6*x^3 + 3/40*x^5 + 5/112*x^7 + ... + 13!!/(14!!*15)*x^15

    Parameters:
    ----------
    data_x : the placeholder of data input

    x_square : the placeholder of the square of data_x

    Returns : A Tensor. Has the same type as data.
    -------
    """

    if x_square is None:
        x_square = tbe.vmul(data_x, data_x)

    res = tbe.vmuls(x_square, tvm.const(COEF[TAYLOR_COUNT],
                                                x_square.dtype))
    for temp in reversed(range(TAYLOR_COUNT)):
        res = tbe.vadds(res, tvm.const(COEF[temp], x_square.dtype))
        if temp == 0:
            res = tbe.vmul(res, data_x)
        else:
            res = tbe.vmul(x_square, res)

    return res


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("asin")
def asin_compute(x, y, kernel_name="asin"):
    """
    do element-wise asin compute
    asin(x) = | arcsin(sqrt(1-x^2)) - HALF_PI, x belongs to (-1, -2^(-0.5))
              | the 15th order taylor expansion, x belongs to (-2^(-0.5), 2^(-0.5))
              | HALF_PI - arcsin(sqrt(1-x^2)), x belongs to (2^(-0.5), 1)

    Parameters:
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "cce_asin"

    Returns : A Tensor. Has the same type as data_input.
    -------
    """

    shape = x.shape
    dtype = x.dtype


    # Change dtype to float32
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd", "float32"):
        x = tbe.cast_to(x, "float32")

    # Sign mask
    sign = util_compute.sign(x)

    # All positive
    x = tbe.vmul(x, sign)

    # x belongs to (0, 2^(-0.5))
    if tbe_platform.cce_conf.api_check_support("te.lang.cce.vmins", x.dtype):
        choice_1 = tbe.vmins(x, tvm.const(BOUNDARY_1, x.dtype))
    else:
        boundary_mask1 = tbe.broadcast(tvm.const(BOUNDARY_1, x.dtype), shape)
        choice_1 = tbe.vmin(x, boundary_mask1)

    if tbe_platform.cce_conf.api_check_support("te.lang.cce.vsubs", choice_1.dtype):
        choice_1 = tbe.vsubs(choice_1, tvm.const(BOUNDARY_1, choice_1.dtype))
    else:
        boundary_mask1 = tbe.broadcast(tvm.const(BOUNDARY_1, choice_1.dtype), shape)
        choice_1 = tbe.vsub(choice_1, boundary_mask1)

    choice_1 = tbe.vmuls(tbe.floor(choice_1), NEG_NUM_ONE)

    res_1 = _taylor_compute(x)
    res_1 = tbe.vmul(res_1, choice_1)

    # x belongs to (2^(-0.5), 1)
    choice_2 = tbe.vmuls(choice_1, tvm.const(NEG_NUM_ONE, x.dtype))
    choice_2 = tbe.vadds(choice_2, tvm.const(NUM_ONE, x.dtype))

    res_2 = tbe.vmul(x, x)
    res_2 = tbe.vmuls(res_2, tvm.const(NEG_NUM_ONE, x.dtype))
    res_2 = tbe.vadds(res_2, tvm.const(NUM_ONE, x.dtype))
    res_2_sqrt = tbe.vsqrt(res_2)

    res_2 = _taylor_compute(res_2_sqrt, res_2)

    res_2 = tbe.vmuls(res_2, tvm.const(NEG_NUM_ONE, x.dtype))
    res_2 = tbe.vadds(res_2, tvm.const(HALF_PI, x.dtype))
    res_2 = tbe.vmul(res_2, choice_2)

    # Restore sign
    res_1 = tbe.vadd(res_1, res_2)
    res_1 = tbe.vmul(res_1, sign)

    # Restore dtype
    if dtype == "float16":
        res_1 = tbe.cast_to(res_1, "float16")

    return res_1


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def asin(x, y, kernel_name="asin"):
    """
    ----------
    asin(x) = | arcsin(sqrt(1-x^2)) - HALF_PI, x belongs to (-1, 2^(-0.5))
              | the 15th order taylor expansion, x belongs to (-2^(-0.5), 2^(-0.5))
              | HALF_PI - arcsin(sqrt(1-x^2)), x belongs to (2^(-0.5), 1)

    Parameters:
    ----------
    x : the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "asin"

    Returns : None
    -------
    """
    shape_input = x.get("shape")
    dtype_input = x.get("dtype")

    para_check.check_shape(shape_input, param_name="x")
    shape_input, _ = shape_util.refine_shape_axes(shape_input, [])

    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_input, check_list, param_name="x")

    inp_dtype = dtype_input.lower()
    data_input = tvm.placeholder(shape_input, dtype=inp_dtype,
                                 name="data_input")

    res = asin_compute(data_input, y, kernel_name)

    with tvm.target.cce():
        auto_sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": [data_input, res],
              "bool_storage_as_1bit": False}

    tbe.cce_build_code(auto_sch, config)
