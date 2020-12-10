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


# pylint: disable=too-many-arguments,invalid-name,unused-argument,too-many-locals,too-many-return-statements
def is_support_5hd(predict, target, weight, pos_weight, reduction):
    """is_support_5hd.

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

    Returns
    -------
    bool
    """
    predict_shape = predict.get("ori_shape")
    predict_format = predict.get("ori_format")
    target_shape = target.get("ori_shape")
    target_format = target.get("ori_format")

    if predict_shape != target_shape \
            or predict_format != target_format:
        return False

    if weight is not None:
        weight_shape = weight.get("ori_shape")
        weight_format = weight.get("ori_format")
        #broadcast
        if predict_shape != weight_shape \
                or predict_format != weight_format:
            return False

    if pos_weight is not None:
        pos_weight_shape = pos_weight.get("ori_shape")
        pos_weight_format = pos_weight.get("ori_format")
        #broadcast
        if predict_shape != pos_weight_shape \
                or predict_format != pos_weight_format:
            return False

    if reduction in ("none", "sum"):
        return True

    c0_num = 16
    if len(predict_shape) == 4:
        if predict_format == "NCHW":
            dim_c = predict_shape[1]
        elif predict_format == "NHWC":
            dim_c = predict_shape[3]
        else:
            return False

        if dim_c % c0_num == 0:
            return True

    return False


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
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
        dtype = ["float16"]
    else:
        dtype = ["float16", "float"]

    dtype_length = len(dtype)
    format = ["ND"] * dtype_length

    if is_support_5hd(predict, target, weight, pos_weight, reduction):
        dtype = dtype * dtype_length
        format = format + ["NC1HWC0"] * dtype_length

    dtype_total = ','.join(dtype)
    format_total = ','.join(format)

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
        classify="output0", name="loss", datatype=dtype_total,
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
    # info: log(1+exp(-x)) == max(predict,0)+log(1+exp(-abs(x))
    const_zero = tvm.const(SCALAR_ZREO, dtype=predict.dtype)
    const_one = tvm.const(SCALAR_ONE, dtype=predict.dtype)
    # info: max(predict,0)
    max_predict_zero = te.lang.cce.vmaxs(predict, const_zero)
    # info: log(1+exp(-abs(x))
    abs_predict = te.lang.cce.vabs(predict)
    const_zero_broadcast = te.lang.cce.broadcast(const_zero, shape_predict)
    reverse_abs_predict = te.lang.cce.vsub(const_zero_broadcast, abs_predict)
    exp_predict = te.lang.cce.vexp(reverse_abs_predict)
    adds_exp_predict = te.lang.cce.vadds(exp_predict, const_one)
    log_exp_predict = te.lang.cce.vlog(adds_exp_predict, priority_flag=1)
    log_exp_predict_res = te.lang.cce.vadd(max_predict_zero, log_exp_predict)

    if pos_weight is not None:
        pos_weight = te.lang.cce.broadcast(pos_weight, shape_predict)
        mul_pos_weight_target = te.lang.cce.vmul(pos_weight, target)
        sub_pos_weight = te.lang.cce.vsub(mul_pos_weight_target, target)
        adds_pos_weight = te.lang.cce.vadds(sub_pos_weight, const_one)
        mul_pos_weight_log = te.lang.cce.vmul(adds_pos_weight, log_exp_predict_res)
        mul_pos_weight_target_predict = te.lang.cce.vmul(mul_pos_weight_target, predict)
        loss = te.lang.cce.vsub(mul_pos_weight_log, mul_pos_weight_target_predict)
    else:
        mul_res = te.lang.cce.vmul(predict, target)
        loss = te.lang.cce.vsub(log_exp_predict_res, mul_res)

    if weight is not None:
        weight = te.lang.cce.broadcast(weight, shape_predict)
        loss = te.lang.cce.vmul(loss, weight)

    if reduction != "none":
        axis = list(range(len(shape_predict)))
        if reduction == "mean":
            reduce_elts = 1.0
            for i in axis:
                reduce_elts *= shape_predict[i]
            cof = reduce_elts ** (-1)
            loss = te.lang.cce.vmuls(loss, cof)

        loss = te.lang.cce.sum(loss, axis)

    if predict_dtype == "float16":
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
