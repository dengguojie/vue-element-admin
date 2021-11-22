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
hardtanh_grad
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-many-arguments,too-many-locals
# 'pylint: disable=pointless-string-statement,no-else-return,unused-argument,invalid-name
@register_operator_compute("HardtanhGrad", op_mode="dynamic", support_fusion=True)
def hardtanh_grad_compute(input_result, input_grad, output_y, min_val=-1.0, max_val=1.0, kernel_name="hardtanh_grad"):
    """
    calculating data

    Parameters
    ----------
    input_result : TVM tensor
        the placeholder of input_x
    input_grad : TVM tensor
        the placeholder of input_y
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    min_val : float
        default to -1.0
    max_val : float
        default to 1.0
    kernel_name : str
        kernel name, default value is "hardtanh_grad"

    Returns
    -------
    output tensor
    """

    """
    Please refer to the TE DSL Manual, And code here with TE DSL.
    """
    inf_fp32_val = 1e-10

    in_data_type = input_result.dtype.lower()
    f_min = tvm.const(min_val, dtype="float32")
    f_max = tvm.const(max_val, dtype="float32")

    max_tensor = tbe.broadcast(f_max, input_result.shape, output_dtype="float32")
    min_tensor = tbe.broadcast(f_min, input_result.shape, output_dtype="float32")

    if in_data_type == "float16":
        in_grad_float = tbe.cast_to(input_grad, "float32")
        in_result_float = tbe.cast_to(input_result, "float32")
        tmp_min = tbe.vmins(in_result_float, f_max)
    else:
        tmp_min = tbe.vmins(input_result, f_max)

    # control value in maximum & minimum
    tmp_max = tbe.vmaxs(tmp_min, f_min)
    if max_val != 0:
        sub_max = tbe.vsub(tmp_max, max_tensor)
    else:
        sub_max = tmp_max
    if min_val != 0:
        sub_min = tbe.vsub(tmp_max, min_tensor)
    else:
        sub_min = tmp_max
    mul_max_min = tbe.vmul(sub_max, sub_min)

    add_inf = tbe.vadds(mul_max_min, tvm.const(inf_fp32_val , dtype="float32"))
    div_res = tbe.vdiv(mul_max_min, add_inf)

    if in_data_type == "float16":
        res_float32 = tbe.vmul(div_res, in_grad_float)
        return tbe.cast_to(res_float32, in_data_type)
    else:
        return tbe.vmul(div_res, input_grad)


@register_operator("HardtanhGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def hardtanh_grad(result, grad, y, min_val, max_val, kernel_name="hardtanh_grad"):
    """
    calculating data

    Parameters
    ----------
    result : dict
        shape and dtype of input
    grad : dict
        shape and dtype of input, should be same shape and type as result
    y : dict
        shape and dtype of input, should be same shape and type as input
    min_val:
        minimum value of the linear region range.
    max_val:
        maximum value of the linear region range.
    kernel_name : str
        kernel name, default value is "hardtanh_grad"

    Returns
    -------
    None
    """

    result_shape = result.get("shape")
    result_dtype = (result.get("dtype")).lower()
    grad_shape = grad.get("shape")
    grad_dtype = (grad.get("dtype")).lower()

    """
    operator check
    """
    para_check.check_kernel_name(kernel_name)

    check_tuple = ("float16", "float32")
    para_check.check_dtype(grad_dtype, check_tuple)
    para_check.check_dtype(result_dtype, check_tuple)

    if grad_dtype != result_dtype:
        raise RuntimeError("grad datatype %s and result datatype %s should be equal!" % (grad_dtype, result_dtype))

    if result_shape != grad_shape:
        raise RuntimeError("grad shape %s and result shape %s should be equal!" % (grad_shape, result_shape))

    """
    operator compute, invoke hardtanh_grad_compute
    """
    ins = classify([result, grad], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_result, _grad) in ins:
        with tbe.compute():
            shape_result, shape_grad = shape_util.variable_shape([_result, _grad])
            data_result = tvm.placeholder(shape_result, name="data_result", dtype=result_dtype)
            data_grad = tvm.placeholder(shape_grad, name="data_grad", dtype=grad_dtype)
            res = hardtanh_grad_compute(data_result, data_grad, y, min_val, max_val, kernel_name)
            tensors.append([data_result, data_grad, res])

        """
        auto schedule
        """
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    """
    operator build
    """
    config = {"name": kernel_name,
              "tensor_list": tensors}

    tbe.build(schedules, config)
