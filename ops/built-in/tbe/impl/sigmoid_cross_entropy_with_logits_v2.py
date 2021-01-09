#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights losserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

sigmoid_cross_entropy_with_logits_v2
"""

import te.lang.cce
from te import tvm
from te import platform as tbe_platform
from te.platform.fusion_manager import fusion_manager
from te.utils import shape_util
from te.utils import para_check
from topi import generic
from topi.cce.util import compare_tensor_dict_key
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

# define a scalar, value = 1
SCALAR_ONE = 1
# define a scalar, value = 0
SCALAR_ZREO = 0


# pylint: disable=redefined-builtin,too-many-arguments,too-many-locals,unused-argument
def op_select_format(predict, target, weight, pos_weight, loss, reduction="mean",
                     kernel_name="sigmoid_cross_entropy_with_logits_v2"):
    """op_select_format.

    Parameters
    ----------
    predict : dict
        shape and dtype of predict
    target : dict
        shape and dtype of target
    weight : dict
        shape and dtype of weight
    pos_weight : dict
        shape and dtype of pos_weight
    loss : dict
        shape and dtype of output, should be same shape and type as input
    reduction: str
        default value is "mean"
    kernel_name : str
        kernel name, default value is "sigmoid_cross_entropy_with_logits_v2"

    Returns
    -------
    None
    """
    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        dtype = ["float16"]
    else:
        dtype = ["float16", "float"]

    dtype_length = len(dtype)
    format_list = ["ND", "NC1HWC0"]
    dtype = dtype * len(format_list)

    format = []
    for data_format in format_list:
        format = format + [data_format] * dtype_length

    dtype_total = ','.join(dtype)
    format_total = ','.join(format)

    if predict.get("dtype").lower() == "float16" and reduction != "none":
        dtype_output = ','.join(["float"] * len(dtype))
    else:
        dtype_output = dtype_total

    input0 = gen_param(
        classify="input0", name="predict", datatype=dtype_total,
        format=format_total)
    input1 = gen_param(
        classify="input1", name="target", datatype=dtype_total,
        format=format_total)
    input2 = gen_param(
        classify="input2", name="weight", datatype=dtype_total,
        format=format_total)
    input3 = gen_param(
        classify="input3", name="pos_weight", datatype=dtype_total,
        format=format_total)
    output0 = gen_param(
        classify="output0", name="loss", datatype=dtype_output,
        format=format_total)

    param_list = [input0, input1, input2, input3, output0]

    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("sigmoid_cross_entropy_with_logits_v2")
def sigmoid_cross_entropy_with_logits_v2_compute(predict,
                                                 target,
                                                 weight,
                                                 pos_weight,
                                                 loss,
                                                 reduction,
                                                 kernel_name):
    """
    calculating data

    Parameters
    ----------
    predict : TVM tensor
        the placeholder of predict
    target : TVM tensor
        the placeholder of target
    weight : TVM tensor
        the placeholder of weight
    pos_weigth : TVM tensor
        the placeholder of pos_weight
    loss : dict
        dict of loss, include keys(shape and dtype)
    reduction: str
        default value is "mean"
    kernel_name : str
        kernel name, default value is "sigmoid_cross_entropy_with_logits_v2"

    Returns
    -------
    output tensor
    """
    predict_dtype = predict.dtype

    is_support_float32 = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vmul", "float32")

    if is_support_float32:
        if predict_dtype == "float16":
            predict = te.lang.cce.cast_to(predict, "float32")

        if target.dtype == "float16":
            target = te.lang.cce.cast_to(target, "float32")

        if weight is not None and weight.dtype == "float16":
            weight = te.lang.cce.cast_to(weight, "float32")

        if pos_weight is not None and pos_weight.dtype == "float16":
            pos_weight = te.lang.cce.cast_to(pos_weight, "float32")

    shape_predict = te.lang.cce.util.shape_to_list(predict.shape)
    const_zero = tvm.const(SCALAR_ZREO, dtype=predict.dtype)
    const_one = tvm.const(SCALAR_ONE, dtype=predict.dtype)
    const_zero_broadcast = te.lang.cce.broadcast(const_zero, shape_predict)
    const_one_broadcast = te.lang.cce.broadcast(const_one, shape_predict)

    # info: max(-predict,0)
    reversed_predict = te.lang.cce.vsub(const_zero_broadcast, predict)
    max_predict_zero = te.lang.cce.vmaxs(reversed_predict, const_zero)

    # info: max_val=max(-predict,0)
    # info: ln(1+exp(-x))=max_val+np.log(np.exp(-max_val)+np.exp(-predict-max_val)))
    reversed_max_predict_zero = te.lang.cce.vsub(const_zero_broadcast, max_predict_zero)
    exp_reversed_max_predict_zero = te.lang.cce.vexp(reversed_max_predict_zero)
    sub_reversed_max_predict_zero = te.lang.cce.vsub(reversed_max_predict_zero, predict)
    exp_sub_reversed_max_predict_zero = te.lang.cce.vexp(sub_reversed_max_predict_zero)
    add_reversed_predict = te.lang.cce.vadd(exp_reversed_max_predict_zero, exp_sub_reversed_max_predict_zero)
    log_reversed_predict = te.lang.cce.vlog(add_reversed_predict, priority_flag=1)
    add_max_predict = te.lang.cce.vadd(log_reversed_predict, max_predict_zero)

    # info: (1-target)*predict
    sub_target = te.lang.cce.vsub(const_one_broadcast, target)
    mul_predict_target = te.lang.cce.vmul(sub_target, predict)

    if pos_weight is not None:
        # info: log_weight=(pos_weight - 1)*target+1
        # info: loss=(1-target)*predict+(log_weight*(max_val+np.log(np.exp(-max_val)+np.exp(-predict-max_val))))
        pos_weight = te.lang.cce.broadcast(pos_weight, shape_predict)
        sub_pos_weight = te.lang.cce.vsub(pos_weight, const_one_broadcast)
        mul_pos_weight = te.lang.cce.vmul(sub_pos_weight, target)
        add_pos_weight = te.lang.cce.vadds(mul_pos_weight, const_one)
        mul_pos_weight_predict = te.lang.cce.vmul(add_pos_weight, add_max_predict)
        loss = te.lang.cce.vadd(mul_predict_target, mul_pos_weight_predict)
    else:
        # info: loss=(1-target)*predict+max_val+np.log(np.exp(-max_val)+np.exp(-predict-max_val))
        loss = te.lang.cce.vadd(mul_predict_target, add_max_predict)

    if weight is not None:
        weight = te.lang.cce.broadcast(weight, shape_predict)
        loss = te.lang.cce.vmul(loss, weight)

    if predict_dtype == "float16" and reduction == "none":
        loss = te.lang.cce.cast_to(loss, "float16")

    return loss


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def sigmoid_cross_entropy_with_logits_v2(
        predict, target, weight, pos_weight, loss, reduction="mean",
        kernel_name="sigmoid_cross_entropy_with_logits_v2"):
    """
    calculating data

    Parameters
    ----------
    predict : dict
        shape and dtype of predict
    target : dict
        shape and dtype of target
    weight : dict
        shape and dtype of weight
    pos_weight : dict
        shape and dtype of pos_weight
    loss : dict
        shape and dtype of output, should be same shape and type as input
    reduction: str
        default value is "mean"
    kernel_name : str
        kernel name, default value is "sigmoid_cross_entropy_with_logits_v2"

    Returns
    -------
    None
    """
    check_list = ("float16", "float32")

    shape_predict = predict.get("shape")
    dtype_predict = predict.get("dtype").lower()
    para_check.check_shape(shape_predict, param_name="predict")
    para_check.check_dtype(dtype_predict, check_list, param_name="predict")

    shape_target = target.get("shape")
    dtype_target = target.get("dtype").lower()
    para_check.check_shape(shape_target, param_name="target")
    para_check.check_dtype(dtype_target, check_list, param_name="target")

    compare_tensor_dict_key(predict, target, "shape")

    if reduction not in ("mean", "sum", "none"):
        raise RuntimeError("{} is not a valid value for reduction".format(reduction))

    data_weight = None
    if weight is not None:
        shape_weight = weight.get("shape")
        dtype_weight = weight.get("dtype").lower()
        para_check.check_shape(shape_weight, param_name=weight)
        para_check.check_dtype(dtype_weight, check_list, param_name="weight")
        _, shape_weight, _ = shape_util.broadcast_shapes(shape_predict, shape_weight,
                                                         param_name_input1="predict",
                                                         param_name_input2="weight")
        data_weight = tvm.placeholder(shape_weight,
                                      name="data_weight",
                                      dtype=dtype_weight)

    data_pos_weight = None
    if pos_weight is not None:
        shape_pos_weight = pos_weight.get("shape")
        dtype_pos_weight = pos_weight.get("dtype").lower()
        para_check.check_shape(shape_pos_weight, param_name=weight)
        para_check.check_dtype(dtype_pos_weight, check_list, param_name="pos_weight")
        _, shape_pos_weight, _ = shape_util.broadcast_shapes(shape_predict, shape_pos_weight,
                                                             param_name_input1="predict",
                                                             param_name_input2="pos_weight")
        data_pos_weight = tvm.placeholder(shape_pos_weight,
                                          name="data_pos_weight",
                                          dtype=dtype_pos_weight)

    data_predict = tvm.placeholder(shape_predict,
                                   name="data_predict",
                                   dtype=dtype_predict)
    data_target = tvm.placeholder(shape_target,
                                  name="data_target",
                                  dtype=dtype_target)

    loss = sigmoid_cross_entropy_with_logits_v2_compute(data_predict,
                                                        data_target,
                                                        data_weight,
                                                        data_pos_weight,
                                                        loss,
                                                        reduction,
                                                        kernel_name)
    tensor_list = [data_predict, data_target]
    if data_weight is not None:
        tensor_list.append(data_weight)

    if data_pos_weight is not None:
        tensor_list.append(data_pos_weight)

    tensor_list.append(loss)

    with tvm.target.cce():
        sch = generic.auto_schedule(loss)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(sch, config)
