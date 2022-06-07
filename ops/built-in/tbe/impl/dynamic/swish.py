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
dynamic swish
'y = x * sigmoid(scale * x)'
"""
import math
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant.
    """
    CONST_FP32_MAX = 3.4e+38
    CONST_FP16_MAX = 65504


def swish_normal(data_input, scale, dtype, exp_support):
    tmp_negative = tbe.vmuls(data_input, tvm.const(-scale, dtype=dtype))

    # avoid data overflow
    if dtype == "float32" and exp_support:
        ln_res = math.log(Constant.CONST_FP32_MAX)
        ln_res = int(ln_res * 10) / 10
    else:
        ln_res = math.log(Constant.CONST_FP16_MAX)
        ln_res = int(ln_res * 1000) / 1000
    tmp_negative = tbe.vmins(tmp_negative, ln_res)

    if dtype == "float32" and not exp_support:
        tmp_negative = tbe.cast_to(tmp_negative, "float16")
    if dtype == "float16" and exp_support:
        tmp_negative = tbe.cast_to(tmp_negative, "float32")
    tmp_exp = tbe.vexp(tmp_negative)
    if not exp_support:
        tmp_exp = tbe.cast_to(tmp_exp, "float32")
    tmp_sum = tbe.vadds(tmp_exp, tvm.const(1.0, dtype="float32"))
    if tmp_sum.dtype == "float16":
        tmp_sum = tbe.cast_to(tmp_sum, "float32")
    if dtype == "float16":
        data_input = tbe.cast_to(data_input, "float32")
        res = tbe.vdiv(data_input, tmp_sum)
        return tbe.cast_to(res, dtype)
    else:
        return tbe.vdiv(data_input, tmp_sum)


def swish_overflow(data_input, scale, dtype):
    data_input = tbe.cast_to(data_input, "float32")
    scale_input = tbe.vmuls(data_input, tvm.const(scale, dtype="float32"))
    abs_scale_input = tbe.vabs(scale_input)
    minus_abs = tbe.vmuls(abs_scale_input, tvm.const(-1.0, dtype="float32"))
    sign_diff = tbe.vadd(scale_input, minus_abs)
    half_sign_diff = tbe.vmuls(sign_diff, tvm.const(0.5, dtype="float32"))

    exp_top = tbe.vexp(half_sign_diff)
    exp_bottom = tbe.vexp(minus_abs)
    one_plus_exp = tbe.vadds(exp_bottom, tvm.const(1.0, dtype="float32"))

    input_mul_exp = tbe.vmul(data_input, exp_top)
    res = tbe.vdiv(input_mul_exp, one_plus_exp)
    if dtype == "float16":
        return tbe.cast_to(res, "float16")
    else:
        return res


# 'pylint: disable=unused-argument,too-many-locals,invalid-name
@register_operator_compute("Swish", op_mode="dynamic", support_fusion=True)
def swish_compute(data_input, y, scale, kernel_name="swish"):
    """
    calculating Swish
    Parameters
    ----------
    data_input : TVM tensor
        the placeholder of input data
    data_output : dict
        shape and dtype of output data, should be same shape and type as input
    scale: float
        scalar of sigmoid, default value is 1.0
    kernel_name : str
        kernel name, default value is "swish"

    Returns
    -------
    output tensor
    """
    dtype = data_input.dtype.lower()
    exp_support = tbe_platform.api_check_support("te.lang.cce.vexp", "float32")
    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") == "Ascend910":
        return swish_overflow(data_input, scale, dtype)
    else:
        return swish_normal(data_input, scale, dtype, exp_support)


@register_operator("Swish")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def swish(x, y, scale=1.0, kernel_name="swish"):
    """
    calculating Swish
    Parameters
    ----------
    x : dict
        dict of x, include keys(shape and dtype)
    y : dict
        shape and dtype of output, should be same shape and type as input
    scale: float
        scalar of sigmoid, default value is 1.0
    kernel_name : str
        kernel name, default value is "swish"

    Returns
    -------
    None
    """
    input_dtype = x.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="x")
    schedules, tensors = [], []
    ins = classify([x], OpPatternMode.ELEWISE)
    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(x_shape[0],
                                         name="data_input",
                                         dtype=input_dtype)
            res = swish_compute(data_input, y, scale, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors,
              "bool_storeage_as_lbit": False}
    tbe.build(schedules, config)
