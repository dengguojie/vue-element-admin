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
kl_div
"""
import functools

import te.platform as tbe_platform
from te import tvm
from te.lang import cce as tbe
from te.utils import para_check


# pylint: disable=too-many-arguments,unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("kl_div")
def kl_div_compute(input_x, input_target, output_y, reduction, batch_size, kernel_name="kl_div"):
    """
    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    input_target : TVM tensor
        the placeholder of input_target
    output_y : dict
        shape and dtype of output, should be same shape and type as input_x
    reduction: str
        Specifies the reduction to apply to the output:
        reduction="batchmean" or reduction="sum".
        "batchmean": the sum of the output will be divided by the batchsize
        "sum": the output will be summed
    batch_size: int
        Equal to the first dimension value of the input shape.
    kernel_name : str
        cce kernel name, default value is "kl_div"

    Returns
    ------
    compute result of kl_div
    """
    input_dtype = input_x.dtype
    log_support_fp32 = tbe_platform.api_check_support("te.lang.cce.vlog", "float32")
    if log_support_fp32 and input_dtype == "float32":
        log_target = tbe.vlog(input_target, priority_flag=1)
    else:
        log_target = tbe.vlog(input_target)

    tmp_result = tbe.vsub(log_target, input_x)
    output_pos = tbe.vmul(input_target, tmp_result)

    # max(output_pos, 0)
    target_gt_zero = tbe.vmaxs(input_target, 0)

    if input_dtype == "float16":
        # algrithm : Y = X*1024/(X*1024+ESP_MIN)
        # for float16, add a small number which value is 1.18e-7, so that the
        # divisor is not equal to 0, and for accuracy, multiply by a number
        # which value is 1024.
        mul_big = tbe.vmuls(target_gt_zero, 1024)
        add_espmin = tbe.vadds(mul_big, 1.18e-7)
        y_espmin = tbe.vdiv(mul_big, add_espmin)
    if input_dtype == "float32":
        # algrithm : Y = X/(X*+ESP_MIN)
        # for float32, add a small number which value is 1.18e-38, so that
        # the divisor is not equal to 0.
        add_espmin = tbe.vadds(target_gt_zero, 1.18e-38)
        y_espmin = tbe.vdiv(target_gt_zero, add_espmin)

    output_res = tbe.vmul(y_espmin, output_pos)

    if reduction == "batchmean":
        output_res = tbe.vmuls(output_res, 1.0 / batch_size)
        final_res = tbe.sum(output_res, axis=0)
    elif reduction == "sum":
        final_res = tbe.sum(output_res, axis=0)
    else:
        raise RuntimeError("Reduction method only support batchmean and sum")

    return final_res


def _check_parameter(input_x, input_target):
    """
    Parameters
    ----------
    input_x : dict
        shape and dtype of input_x
    input_target : dict
        shape and dtype of input_target.Shape and dtype must be same as input_x
    Returns
    ------
    None
    """
    shape_x = input_x.get("shape")
    shape_target = input_target.get("shape")
    para_check.check_shape(shape_x, param_name="input_x")
    if list(shape_x) != list(shape_target):
        raise RuntimeError("input_x and input_target must have the same shape.")

    # check input tensor data_type
    dtype_x = input_x.get("dtype").lower()
    dtype_target = input_target.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_x, check_list, param_name="input_x")
    if dtype_x != dtype_target:
        raise RuntimeError("input_x and input_target must have the same dtype.")

    if dtype_x == "float32" and not tbe_platform.api_check_support("te.lang.cce.vmul", "float32"):
        raise RuntimeError("Instric only support float16 while input dtype is float32")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def kl_div(input_x, input_target, output_y, reduction, kernel_name="kl_div"):
    """
    Calcuate Kullback-Leibler divergence.

    output_pos = input_target * (log(input_target) - input_x)
    output = where(input_target > 0, output_pos, zeros)
    reduced = reduce_sum_all(output)
    if reduction = "batchmean":
        final_res = reduce / input.dim[0]
    else:
        final_res = reduced
    Parameters
    ----------
    input_x : dict
        shape and dtype of input_x, dtype only support fp16 and fp32.
    input_target : dict
        shape and dtype of input_target.Shape and dtype must be same as input_x
    output_y : dict
        shape and dtype of output.Dtype must be same as input_x
    reduction: str
        Specifies the reduction to apply to the output:
        reduction="batchmean" or reduction="sum".
        "batchmean": the sum of the output will be divided by the batchsize
        "sum": the output will be summed
    kernel_name : str
        cce kernel name, default value is "kl_div"

    Returns
    ------
    None
    """
    # check input parameter
    _check_parameter(input_x, input_target)

    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype")
    batch_size = shape_x[0]
    shape_one_dim = [functools.reduce(lambda x, y: x * y, shape_x[:])]
    data_x = tvm.placeholder(shape_one_dim, name="data_x", dtype=dtype_x)
    data_target = tvm.placeholder(shape_one_dim, name="data_target", dtype=dtype_x)

    final_res = kl_div_compute(data_x, data_target, output_y, reduction, batch_size, kernel_name=kernel_name)
    with tvm.target.cce():
        auto_sch = tbe.auto_schedule(final_res)

    config = {"name": kernel_name, "tensor_list": (data_x, data_target, final_res)}

    tbe.cce_build_code(auto_sch, config)
