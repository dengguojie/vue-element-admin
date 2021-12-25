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
celu
"""
import te.lang.cce as tbe
from tbe import tvm
from tbe.common.register import register_op_compute
from tbe.common.utils import para_check
from te import platform as tbe_platform


@register_op_compute("Celu")
# 'pylint:disable=too-many-arguments, too-many-locals
def celu_compute(x, y, alpha=1.0, kernel_name="celu"):
    """
    Implement the operator by referring to  the
            TBE Operator Development Guide.
    scale * (max(0, x) + min(0, alpha * (exp(x/input_scale) - 1)
    x:dict of x, include shape and data_type
    y:dict of y, include shape and data_type
    alpha: attr, alpha of the min
    input_scale: attr, input scale of the input

    """

    high_perf = False
    data_type = x.dtype
    
    if tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp", "float32"):
        high_perf = True

    type_fp16 = "float16"
    type_fp32 = "float32"

    if data_type.lower() == type_fp16 and high_perf:
        compute_type = type_fp32
        x = tbe.cast_to(x, type_fp32)
    else:
        compute_type = data_type

    # min(0, alpha * (exp(x/input_scale)-1))
    x_mul_rec_input_scale = tbe.vmuls(x, tvm.const(1.0/alpha, compute_type))
    exp_x_mul_rec_input_scale = tbe.vexp(x_mul_rec_input_scale)
    exp_x_mul_rec_input_scale_minus1 = tbe.vadds(exp_x_mul_rec_input_scale, tvm.const(-1.0, compute_type))
    alpha_times_exp_minus1 = tbe.vmuls(exp_x_mul_rec_input_scale_minus1, tvm.const(alpha, compute_type))
    vmin_x = tbe.vmins(alpha_times_exp_minus1, tvm.const(0.0, compute_type))

    # max(0, x)
    vmax_x = tbe.vmaxs(x, tvm.const(0.0, compute_type))

    # add min max
    result = tbe.vadd(vmax_x, vmin_x)

    if data_type.lower() == type_fp16 and high_perf:
        result = tbe.cast_to(result, data_type)

    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def celu(x, y, alpha=1.0, kernel_name="celu"):
    """
    Implement the operator by referring to  the
            TBE Operator Development Guide.
    scale * (max(0, x) + min(0, alpha * (exp(x/input_scale) - 1)
    x:dict of x, include shape and data_type
    y:dict of y, include shape and data_type
    notice: Celu formula is above with scale = 1.0 and input_scale=alpha
    """
    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")
    if alpha == 0:
        raise ZeroDivisionError("alpha is zero, zero division error occur!")

    res = celu_compute(data_x, y, alpha=alpha, kernel_name=kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x, res]}
    tbe.build(schedule, config)
