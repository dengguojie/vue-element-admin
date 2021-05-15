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
from topi.cce import util
from te.utils.error_manager import error_manager_vector
from te.utils import para_check
from te import platform as tbe_platform

@fusion_manager.register("Celu")
def celu_compute(x, y, a1, a2, a3, kernel_name="celu"):
    """
    Implement the operator by referring to  the
            TBE Operator Development Guide.
    celu:
    if x >= 0
        y = alpha3 * 3
    else
        y = alpha1 * (exp(x/alpha2)-1)
    x:dict of x, include shape and dtype
    y:dict of y, include shape and dtype
    a1: scalar, alpha1
    a2: scalar, alpha2
    a3: scalar, alpha3

    """

    data = x
    dtype = data.dtype

    rec_a2 = tvm.const(-1/a2, "float32")
    negative_x = tbe.vmuls(data, tvm.const(-1, "float32"))
    vmax_x = tbe.vmaxs(negative_x, tvm.const(0,"float32"))
    div_a2x = tbe.vmuls(vmax_x, rec_a2)
    exp_a2x = tbe.vexp(div_a2x)
    neg_part = tbe.vadds(exp_a2x, tvm.const(-1,"float32"))
    
    pos_part = tbe.vmaxs(x, tvm.const(0,"float32"))

    mul_a1 = tbe.vmuls(neg_part, tvm.const(a1, "float32"))
    mul_a3 = tbe.vmuls(pos_part, tvm.const(a3, "float32"))

    res = tbe.vadd(mul_a1,mul_a3)
    res = tbe.cast_to(res, dtype)

    return res


# pylint: disable=invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT,para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def celu(x, y, alpha1=1.0, alpha2=1.0, alpha3=1.0, kernel_name="celu"):
    """
    Implement the operator by referring to  the
            TBE Operator Development Guide.
    celu:
    if x >= 0
        y = alpha3 * 3
    else
        y = alpha1 * (exp(x/alpha2)-1)
    x:dict of x, include shape and dtype
    y:dict of y, include shape and dtype
    a1: scalar, alpha1
    a2: scalar, alpha2
    a3: scalar, alpha3

    """
    util.check_kernel_name(kernel_name)
    shape_input = x.get("shape")
    dtype_input = x.get("dtype")
    input_dtype = dtype_input.lower()

    para_check.check_shape(shape_input, param_name="x")

    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_input, check_list, param_name="x")

    if alpha2 == 0:
        error_manager_vector.raise_err_input_value_invalid("celu","alpha2","non-zero","zero")

    data_input = tvm.placeholder(shape_input, name="data_input", dtype=input_dtype)

    res = celu_compute(data_input, y, alpha1, alpha2, alpha3, kernel_name)

    with tvm.target.cce():
        auto_sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    tbe.cce_build_code(auto_sch, config)
