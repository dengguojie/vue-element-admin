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
elu
  Op_description :
    do element-wise elu operation.

    # elu(
    #   x,
    #   y,
    #   kernel_name='cce_elu')

  Supportive_dtype_format :
    ["float16", "float32"]
    ['ND', 'NCHW', 'NHWC', 'NC1HWC0']

  Constraint :
    [1] All : shape size limit is 2147483648

"""
import functools

import te.lang.cce as tbe
from te import tvm
from te import platform as tbe_platform
from te.utils import para_check
from impl.util.platform_adapter import error_manager_vector


class Constant:
    """
    The class for constant
    """
    NUM_ZERO = 0.0
    NUM_ONE_NEG = -1.0


def _elu_computer_performance(data, alpha, dtype):
    """
    computer performance
    """
    scalar_one_neg = tvm.const(Constant.NUM_ONE_NEG, dtype)

    negative_data = tbe.vmuls(data, scalar_one_neg)
    negative_data = tbe.vrelu(negative_data)
    negative_data = tbe.vmuls(negative_data, scalar_one_neg)
    positive_data = tbe.vrelu(data)

    exp_res = tbe.vexp(negative_data)
    exp_res = tbe.vadds(exp_res, scalar_one_neg)

    res = tbe.vmuls(exp_res, tvm.const(alpha, dtype))
    res = tbe.vadd(positive_data, res)

    return res


def _elu_computer_precision(data, alpha, dtype):
    """
    computer precision
    """
    scalar_zero = tvm.const(Constant.NUM_ZERO, dtype)
    negative_data = tbe.vmins(data, scalar_zero)
    positive_data = tbe.vmaxs(data, scalar_zero)

    exp_res = tbe.vexp(negative_data)
    exp_res = tbe.vadds(exp_res, tvm.const(Constant.NUM_ONE_NEG, dtype))

    res = tbe.vaxpy(exp_res, positive_data, tvm.const(alpha, dtype))

    return res


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("elu")
def elu_compute(x, y, alpha, kernel_name="elu"):
    """
    do element-wise elu compute
    f(x) = max(min(alpha(e^x - 1), 0), x),  in cloud scene, for all inputs
    f(x) = max(min(alpha(e^x - 1), 0), x),  in mini scene, for x <= TAYLOR_THRESHOLD or x >= 0
    f(x) = fifth taylor computer,    in mini scene, for TAYLOR_THRESHOLD < x < 0

    Parameters:
    ----------
    x: the placeholder of data input

    alpha: float, coefficient when input tensor is less than zero

    y: the dict of output

    kernel_name : cce kernel name, default value is "elu"

    Returns : A Tensor. Has the same type as data_input.
    -------
    """

    data = x
    dtype = data.dtype

    has_improve_precision = False
    if tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp", "float32"):
        has_improve_precision = True
    if dtype.lower() == "float16" and has_improve_precision:
        data = tbe.cast_to(data, "float32")
        cvt_dtype = "float32"
    else:
        cvt_dtype = dtype

    if has_improve_precision:
        res = _elu_computer_precision(data, alpha, cvt_dtype)
    else:
        res = _elu_computer_performance(data, alpha, cvt_dtype)

    if dtype.lower() == "float16" and has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


# pylint: disable=invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def elu(x, y, alpha=1.0, kernel_name="elu"):
    """
    do element-wise elu operation

    Parameters:
    ----------
    x: the dict of input, only support float16, float32

    alpha: float, coefficient when input tensor is less than zero.

    output_res : the dict of output

    kernel_name : cce kernel name, default value is "elu"

    Returns
    -------
    None
    """

    shape_input = x.get("shape")
    dtype_input = x.get("dtype")
    input_dtype = dtype_input.lower()

    para_check.check_shape(shape_input, param_name="x")

    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_input, check_list, param_name="x")

    if not tbe_platform.cce_conf.api_check_support("te.lang.cce.sum", "float32") and dtype_input == "float32":
        error_manager_vector.raise_err_input_dtype_not_supported("elu", "x", "float16", dtype_input)
    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x*y, shape_input)
    data_input = tvm.placeholder(fuseshape, name="data_input", dtype=input_dtype)

    res = elu_compute(data_input, y, alpha, kernel_name)

    with tvm.target.cce():
        auto_sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": [data_input, res],
              "bool_storage_as_1bit": False}
    tbe.cce_build_code(auto_sch, config)
