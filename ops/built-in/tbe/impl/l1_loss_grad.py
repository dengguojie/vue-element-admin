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
l1_loss_grad
"""

import functools
import te.lang.cce as tbe
from te import tvm
import te.platform as tbe_platform
from te.utils import shape_util
from te.utils import para_check


# 'pylint: disable=invalid-name,too-many-arguments,unused-argument,too-many-locals
@tbe_platform.fusion_manager.fusion_manager.register("l1_loss_grad")
def l1_loss_grad_compute(grad_out, predict, target, y, reduction="mean", kernel_name="l1_loss_grad"):
    """
    l1_loss_grad_compute
    """
    predict_dtype = predict.dtype.lower()
    if predict_dtype == "float16":
        grad_out = tbe.cast_to(grad_out, "float32", False)
        predict = tbe.cast_to(predict, "float32", False)
        target = tbe.cast_to(target, "float32", False)
    zero_tensor = tbe.vmuls(predict, tvm.const(0, dtype="float32"))
    sign = tbe.vcmpsel(predict, target, "gt", 1.0, -1.0)
    # cast sign data type to "float32" to through data type checking
    sign = tbe.cast_to(sign, "float32")
    # rectify sign to 0 when predict equal to target
    sign = tbe.vcmpsel(predict, target, "eq", zero_tensor, sign)
    sign = tbe.cast_to(sign, "float32")
    grad_shape = shape_util.shape_to_list(grad_out.shape)
    n = functools.reduce(lambda x, y: x * y, grad_shape)
    # if choose "mean", grad_out should divide over n
    if reduction == "mean":
        grad_out = tbe.vmuls(grad_out, tvm.const(1 / n, dtype="float32"))
    # chain multiplication to get the gradient of L1 with respect to weights(grad_out)
    res = tbe.vmul(sign, grad_out)
    if predict_dtype == "float16":
        res = tbe.cast_to(res, "float16", False)

    return res


@para_check.check_op_params(dict, dict, dict, dict, str, str)
def l1_loss_grad(grads, predict, label, y, reduction="mean", kernel_name="l1_loss_grad"):
    """
    Parameters
    ----------
    grads : dict
        shape and dtype of grad_out as input
    predict : dict
        shape and dtype of predict as input, should be same shape and type as grads
    label : dict
        shape and dtype of label as input, should be same shape and type as grads
    y : dict
        shape and dtype of output, should be same shape and type as grads
    reduction: string
        reduction name, default value is "mean"
    kernel_name : str
        kernel name, default value is "l1_loss_grad"

    Returns
    -------
    None
    """
    dtype_list = ["float16", "float32"]
    reduction_list = ["none", "mean", "sum"]
    grads_data_type = grads.get("dtype").lower()
    grads_shape = grads.get("shape")
    predict_data_type = predict.get("dtype").lower()
    predict_shape = predict.get("shape")
    label_data_type = label.get("dtype").lower()
    label_shape = label.get("shape")

    para_check.check_dtype(grads_data_type, dtype_list)
    para_check.check_dtype(predict_data_type, dtype_list)
    para_check.check_dtype(label_data_type, dtype_list)

    para_check.check_shape(grads_shape)
    para_check.check_shape(predict_shape)
    para_check.check_shape(label_shape)

    shape_util.compare_tensor_dict_key(grads, predict, "shape")
    shape_util.compare_tensor_dict_key(grads, label, "shape")
    shape_util.compare_tensor_dict_key(grads, predict, "dtype")
    shape_util.compare_tensor_dict_key(grads, label, "dtype")

    if reduction not in reduction_list:
        raise RuntimeError("reduction should be one of ['none','mean','sum']")
    grads = tvm.placeholder(grads_shape, dtype=grads_data_type, name="grads")
    predict = tvm.placeholder(predict_shape, dtype=predict_data_type, name="predict")
    label = tvm.placeholder(label_shape, dtype=label_data_type, name="label")

    res = l1_loss_grad_compute(grads, predict, label, y, reduction=reduction, kernel_name="l1_loss_grad")

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [grads, predict, label, res]}
    tbe.cce_build_code(schedule, config)
