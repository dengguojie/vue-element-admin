# Copyright 2021 Huawei Technologies Co., Ltd
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
dynamic elu
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
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
import te.lang.base as tbe_base
from te.utils import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute

# shape limit 2**31
SHAPE_SIZE_LIMIT = 2147483648
NUM_ZERO = 0.0
NUM_ONE_NEG = -1.0


def _elu_computer_performance(data, scalar_one_neg):
    negative_data = tbe.vmuls(data, scalar_one_neg)
    negative_data = tbe.vrelu(negative_data)
    negative_data = tbe.vmuls(negative_data, scalar_one_neg)
    positive_data = tbe.vrelu(data)
    return negative_data, positive_data


def _elu_computer_precision(data, dtype):
    scalar_zero = tvm.const(NUM_ZERO, dtype)
    negative_data = tbe.vmins(data, scalar_zero)
    positive_data = tbe.vmaxs(data, scalar_zero)
    return negative_data, positive_data

# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@register_operator_compute("Elu", op_mode="dynamic", support_fusion=False)
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

    scalar_one_neg = tvm.const(NUM_ONE_NEG, cvt_dtype)

    if has_improve_precision:
        _negative_data, _positive_data = _elu_computer_precision(data, cvt_dtype)
    else:
        _negative_data, _positive_data = _elu_computer_performance(data, scalar_one_neg)
    exp_res = tbe.vexp(_negative_data)
    exp_res = tbe.vadds(exp_res, scalar_one_neg)
    res = tbe.vaxpy(exp_res, _positive_data, tvm.const(alpha, cvt_dtype))

    if dtype.lower() == "float16" and has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


# pylint: disable=invalid-name
@register_operator("Elu")
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
    input_dtype = x.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="x")
    if not tbe_platform.cce_conf.api_check_support("te.lang.cce.sum", "float32") and input_dtype == "float32":
        error_info = {}
        error_info['errCode'] = 'E80008'
        error_info['param_name'] = 'x'
        error_info['op_name'] = 'elu'
        error_info['expect_value'] = "float16"
        error_info['real_value'] = input_dtype
        raise RuntimeError(error_info, "In op[%s], the parameter[%s]'s dtype "
                                       "should be [%s], but actually is [%s]."
                           % (error_info['op_name'], error_info['param_name'], \
                              error_info['expect_value'], error_info['real_value']))
    schedules, tensors = [], []
    ins = classify([x], Mode.ELEWISE)

    for (_x,) in ins:
        with tbe_base.compute():
            x_shape = shape_util.variable_shape([_x])
            fuseshape = [1]
            fuseshape[0] = functools.reduce(lambda x, y: x * y, x_shape[0])
            data_input = tvm.placeholder(fuseshape, dtype = input_dtype,
                                         name = "data_input")
            res = elu_compute(data_input, y, alpha, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
