#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this
file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

fusion common function for dynamic
"""

from __future__ import absolute_import

from te import tvm
from te.platform import operation


def extract_dict(input_x):
    """
    :param input_x:
    :return:
    """
    if isinstance(input_x, tvm.tensor.Tensor):
        return {"shape": input_x.shape,
                "range": [(1, 1)] * len(input_x.shape)
                }
    return input_x


def create_placeholder(input_x, shape_x):
    """
    :param input_x:
    :param shape_x:
    :return:
    """
    if isinstance(input_x, tvm.tensor.Tensor):
        return input_x
    dtype = input_x.get("dtype").lower()
    return tvm.placeholder(shape_x, dtype=dtype)


def normalize_shape(inputs: list):
    """
    :param inputs:
    :return:
    """
    var_t = tvm.expr.Var
    expr_t = tvm.expr.BinaryOpExpr

    def get_var(_i):
        for _input in inputs:
            dim_i = _input["shape"][_i]
            if isinstance(dim_i, (var_t, expr_t)):
                return dim_i
        for _input in inputs:
            dim_i = _input["shape"][_i]
            range_i = _input["range"][_i]
            if dim_i == -1:
                return operation.var("dim_" + str(_i), range_i)

    shapes, ranges = [], []
    for input_i in inputs:
        shapes.append(input_i["shape"])
        ranges.append(input_i["range"])

    d_shapes = [[] for _ in shapes]
    for i in range(len(shapes[0])):
        _var = get_var(i)
        for d_shape, shape in zip(d_shapes, shapes):
            if isinstance(shape[i], (var_t, expr_t)):
                pass
            if shape[i] == -1:
                d_shape.append(_var)
            else:
                d_shape.append(shape[i])
    return d_shapes


def check_fusion_input(inputs: list):
    """
    :param inputs:
    :return:
    """
    tensor_t = tvm.tensor.Tensor

    for i, input_i in enumerate(inputs):
        if not isinstance(input_i, (tensor_t, dict)):
            raise RuntimeError("The input must be a tensor or dict!")

    if len(list(filter(lambda x: isinstance(x, tensor_t), inputs))) > 1:
        raise RuntimeError("The input tensor number must be less than 1!")
