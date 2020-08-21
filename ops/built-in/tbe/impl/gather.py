#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

gather
"""
from topi.cce import util
from impl.gather_v2_d import gather_v2_d
from te.utils.op_utils import *

# pylint: disable=locally-disabled,unused-argument,invalid-name
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, OPTION_ATTR_BOOL, KERNEL_NAME)
def gather(x, indices, y, validate_indices=True, kernel_name="gather"):
    """Gather slices from `params` according to `indices`.`indices` must be an
    integertensor of any dimension (usually 0-D or 1-D).Produces an output
    tensor with shape `indices.shape + params.shape[1:]`.

    Parameters
    ----------
    x: dict
        dict with keys(shape and dtype) of x
    indices: dict
        dict with keys(shape and dtype) of indices
    y: dict
        dict with keys(shape and dtype) of output
    validate_indices: bool
        An optional `bool`. Defaults to `True`
    kernel_name: str
        kernel name, default value is "pow"

    Returns
    -------
    None
    """
    x_shape = x.get("shape")
    x_dtype = x.get("dtype").lower()
    indices_shape = indices.get("shape")
    indices_dtype = indices.get("dtype").lower()

    check_shape(x_shape, param_name="x")
    check_shape(indices_shape, param_name="indices")
    dtype_list = ("int8", "int16", "int32", "int64", "uint8",
                  "uint16", "uint32", "uint64", "float16", "float32")
    check_dtype(indices_dtype, ("int32", "int64"), param_name="indices")
    check_dtype(x_dtype, dtype_list, param_name="x")

    gather_v2_d(x, indices, y, 0, kernel_name)
