#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

assignsub
"""
from te import tvm
import te.lang.cce as tbe

from te.utils import shape_util
import te.lang.base as tbe_base
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from te.utils import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@register_operator_compute("AssignSub", op_mode="dynamic", support_fusion=False)
def assign_sub_compute(var, value, output_z, kernel_name="assign_sub"):
    """
    assign_sub compute function

    Parameters
    ----------
    tensor_var : tvm.tensor
        tensor of var
    tensor_value : tvm.tensor
        tensor of value
    out : dict
        dict of out.
    kernel_name : str
        cce kernel name, default value is "assign_sub"

    Returns
    -------
    res : tvm.tensor
        tensor of result
    """
    dtype = value.dtype
    if dtype in ("int8", "uint8"):
        var = tbe.cast_to(var, "float16")
        value = tbe.cast_to(value, "float16")
    res = tbe.vsub(var, value)
    if dtype in ("int8", "uint8"):
        res = tbe.cast_to(res, dtype)
    return res


@register_operator("AssignSub")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def assign_sub(var, value, output_z, kernel_name="assign_sub"):
    """
    Update var by subtracting value from it.

    Parameters:
    ----------
    var : dict
        dict of input_var, include shape and dtype,
        dtype support int8, uint8, int32, float16, float32

    value : dict
        dict of input_value, include shape and dtype,
        dtype support int8, uint8, int32, float16, float32.
        Must have the same shape and dtype as input_var

    out : dict
        dict of out

    kernel_name : str
        cce kernel name, default value is "assign_sub"

    Returns
    -------
    None
    """
    check_list = ["float16", "float32", "int32", "int8", "uint8"]
    dtype_var = var.get("dtype").lower()
    dtype_value = value.get("dtype").lower()

    para_check.check_dtype(dtype_var, check_list, param_name="var")
    para_check.check_dtype(dtype_value, check_list, param_name="value")

    ins = classify([var, value], Mode.ELEWISE)
    schedules, tensors = [], []
    for (data1, data2) in ins:
        with tbe_base.compute():
            var_shape, value_shape = shape_util.variable_shape([data1, data2])

            tensor_var = tvm.placeholder(var_shape, dtype_var, "tensor_var")
            tensor_value = tvm.placeholder(value_shape, dtype_value, "tensor_value")
            res = assign_sub_compute(tensor_var, tensor_value, output_z, kernel_name)
            tensors.append([tensor_var, tensor_value, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    # build
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
