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

import functools
import te.lang.cce as tbe
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from te.utils import para_check
from te import platform as tbe_platform


NUM_ZERO = 0.0
NUM_ONE_NEG = -1.0
NUM_ONE_POS = 1.0

def _celu_computer_performance(data, alpha, dtype):
    positive_data = tbe.vrelu(data)

    scalar_one_neg = tvm.const(NUM_ONE_NEG, dtype)
    scalar_alpha_rev = tvm.const(NUM_ONE_NEG/alpha, dtype)
    negative_data = tbe.vmuls(data, scalar_one_neg)
    negative_data = tbe.vrelu(negative_data)
    negative_data = tbe.vmuls(negative_data, scalar_alpha_rev)
    
    exp_res = tbe.vexp(negative_data)
    exp_res = tbe.vadds(exp_res, scalar_one_neg)

    res = tbe.vmuls(exp_res, tvm.const(alpha, dtype))
    res = tbe.vadd(positive_data, res)

    return res


def _celu_computer_precision(data, alpha, dtype):
    scalar_zero = tvm.const(NUM_ZERO, dtype)
    negative_data = tbe.vmins(data, scalar_zero)
    scalar_alpha_rev = tvm.const(NUM_ONE_POS/alpha, dtype)
    negative_data = negative_data.vmuls(negative_data, scalar_alpha_rev)

    positive_data = tbe.vmaxs(data, scalar_zero)

    exp_res = tbe.vexp(negative_data)
    exp_res = tbe.vadds(exp_res, tvm.const(NUM_ONE_NEG, dtype))

    res = tbe.vaxpy(exp_res, positive_data, tvm.const(alpha, dtype))

    return res


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@fusion_manager.register("celu")
def celu_compute(x, y, alpha, kernel_name="celu"):
    """
    do element-wise celu compute

    Parameters:
    ----------
    x: the placeholder of data input

    alpha: float, coefficient when input tensor is less than zero

    y: the dict of output

    kernel_name : cce kernel name, default value is "celu"

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
        res = _celu_computer_precision(data, alpha, cvt_dtype)
    else:
        res = _celu_computer_performance(data, alpha, cvt_dtype)

    if dtype.lower() == "float16" and has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


# pylint: disable=invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def celu(x, y, alpha=1.0, kernel_name="celu"):
    """
    do element-wise celu operation

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
        error_info = {}
        error_info['errCode'] = 'E80008'
        error_info['param_name'] = 'x'
        error_info['op_name'] = 'celu'
        error_info['expect_value'] = "float16"
        error_info['real_value'] = dtype_input
        raise RuntimeError(error_info, "In op[%s], the parameter[%s]'s dtype "
                                       "should be [%s], but actually is [%s]."
                           % (error_info['op_name'], error_info['param_name'], \
                              error_info['expect_value'], error_info['real_value']))

    fuseshape = []
    fuseshape.append(functools.reduce(lambda x, y: x*y, shape_input))
    data_input = tvm.placeholder(fuseshape, name="data_input", dtype=input_dtype)

    res = celu_compute(data_input, y, alpha, kernel_name)

    with tvm.target.cce():
        auto_sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": [data_input, res],
              "bool_storage_as_1bit": False}
    tbe.build(auto_sch, config)
