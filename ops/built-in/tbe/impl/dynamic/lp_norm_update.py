# Copyright 2020 Huawei Technologies Co., Ltd
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
lp_norm_update
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpPatternMode

_CONST_INF = 2147483647
_CONST_EPSILON_FP16 = 1e-7
_CCE_PLAT = tbe_platform.get_soc_spec('SOC_VERSION')

# pylint: disable=invalid-name,unused-argument,too-many-locals
@register_operator_compute("LpNormUpdate", op_mode="dynamic", support_fusion=True)
def lp_norm_update_compute(x, y, p, kernel_name):
    """
    Compute norm for p = 2.
    For precision considering, separate it from lp_norm_update_compute without using vlog.
    
    Compute norm for p >= 3.
    When p equals other int value, lp_norm_update = pow(sum(pow(abs(input),p)),1/p).
    """
    # extraction can be transformed like x^p =  y --> x = exp(log(y)/p)
    if p == 2:
        res = tbe.vsqrt(x, 1)
    else:
        if "910" in _CCE_PLAT:
            log_sum_x = tbe.vlog(x, 1)
        else:
            log_sum_x = tbe.vlog(x)
        zero_tensor = tbe.vmuls(log_sum_x, tvm.const(0, dtype=log_sum_x.dtype))
        p_tensor = tbe.vadds(zero_tensor, tvm.const(p, dtype=log_sum_x.dtype))
        div_log_x = tbe.vdiv(log_sum_x, p_tensor)
        res = tbe.vexp(div_log_x)
   
    return res


@register_operator("LpNormUpdate")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT, 
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def lp_norm_update(x, y, p=2, epsilon=1e-12, kernel_name="lp_norm_update"):
    """
    Computes norm for p equals 0, 1, 2, -inf, inf, or other integers.
    Parameters
    ----------
    x: tensor
       The input tensor.0
       Required.
    y: tensor
       The output tensor.
       Required.
    p: int, inf, -inf
       The order of norm.
       Optional. Default: 2.
    epsilon: float
             The number used for safe considering as norm usually served as denominator.
             Optional. Default: 1e-7 for fp16, 1e-12 for fp32
    kernel_name: str
                 Kernel name.
                 Optional. Default: "lp_norm_update".
    Returns
    -------
    None
    """
    para_check.check_kernel_name(kernel_name)
    xtype_list = ["float16", "float32"]
    x_type = x.get("dtype").lower()
    x_shape = x.get("shape")
    para_check.check_dtype(x_type, xtype_list)
    para_check.check_shape(x_shape)
    p_inf_list = ("inf", "-inf")

    schedules = []
    tensors = []
    ins = classify([x], OpPatternMode.ELEWISE)
    
    for (_x,) in ins:
        with tbe.compute():
            shape_var_new = shape_util.variable_shape([_x,])[0]
            input_data = tvm.placeholder(shape_var_new, name="input_data", dtype=x_type)

            if (p in p_inf_list) or (p == _CONST_INF) or (p == -_CONST_INF - 1) or (p == 0) or (p == 1):
                res = input_data
            else:
                res = lp_norm_update_compute(input_data, y, p, kernel_name)

            if x_type == "float16" and float(epsilon) <= _CONST_EPSILON_FP16:
                if epsilon == 0.0:
                    std_no = tvm.const(0.0, dtype=x_type)
                else:
                    std_no = tvm.const(_CONST_EPSILON_FP16, dtype=x_type)
            else:
                std_no = tvm.const(float(epsilon), dtype=x_type)
            res = tbe.vmaxs(res, std_no)
        
            tensors.append([input_data, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
