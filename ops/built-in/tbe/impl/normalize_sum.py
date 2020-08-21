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

normalize_sum
"""

import te.lang.cce
from te import tvm, platform as tbe_platform
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import *


# pylint: disable=locally-disabled,unused-argument,invalid-name
@fusion_manager.register("normalize_sum")
def normalize_sum_compute(x1, y, data_format, across_spatial=True,
                          kernel_name="normalize_sum"):
    """
    calculating data

    Parameters
    ----------
    x1 : TVM tensor
        the placeholder of x1
    y : dict
        dict of y, include keys(shape and dtype, format)
    data_format: str
        the format of x1
    across_spatial: bool
        indicates whether reduction should cross spatial locations.
        Default(True)
    kernel_name : str
        kernel name, default value is "normalize_sum"

    Returns
    -------
    output tensor
    """

    # set intermediate dtype
    if not tbe_platform.cce_conf.api_check_support("te.lang.cce.sum", "float32"):
        # hisi es, cs
        intermediate_dtype = "float16"
        dtype_cast_mapping = {"int8": "float16"}
    else:
        intermediate_dtype = "float32"
        dtype_cast_mapping = {"int8": "float16", "float16": "float32"}

    x1_cast = x1
    if intermediate_dtype != x1.dtype:
        while x1_cast.dtype in dtype_cast_mapping:
            x1_cast = te.lang.cce.cast_to(x1_cast,
                                          dtype_cast_mapping[x1_cast.dtype])

    x1_cast_sqr = te.lang.cce.vmul(x1_cast, x1_cast)

    if across_spatial:
        x1_cast_sqr_sum = te.lang.cce.sum(x1_cast_sqr, axis=[1, 2, 3],
                                          keepdims=True)
    elif data_format == "NCHW":
        x1_cast_sqr_sum = te.lang.cce.sum(x1_cast_sqr, axis=[1], keepdims=True)
    elif data_format == "NHWC":
        x1_cast_sqr_sum = te.lang.cce.sum(x1_cast_sqr, axis=[3], keepdims=True)

    return x1_cast_sqr_sum


# pylint: disable=locally-disabled,invalid-name
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, OPTION_ATTR_BOOL, KERNEL_NAME)
def normalize_sum(x1, y, across_spatial=True, kernel_name="normalize_sum"):
    """
    calculating data

    Parameters
    ----------
    x1 : dict
        shape and dtype, format of input 1
    y : dict
        shape and dtype, format of output, should be same format as input 1
    across_spatial: bool
        indicates whether reduction should cross spatial locations.
        Default(True)
    kernel_name : str
        kernel name, default value is "normalize_sum"

    Returns
    -------
    None
    """

    shape_1 = x1.get("shape")
    dtype_1 = x1.get("dtype").lower()
    data_format = x1.get("format")

    check_shape(shape_1, param_name="x1")

    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
        # hisi es, cs
        check_dtype(dtype_1, ("int8", "float16",), param_name="x1")
    else:
        check_dtype(dtype_1, ("int8", "float16", "float32",), param_name="x1")

    check_format(data_format, ("NCHW", "NHWC"), param_name="x1")

    check_shape(shape_1, min_rank=4, max_rank=4, param_name="x1")

    data_x1 = tvm.placeholder(shape_1, name="data_1", dtype=dtype_1)
    res = normalize_sum_compute(data_x1, y, data_format, across_spatial,
                                kernel_name)

    # pylint: disable=no-member
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "need_build": True,
              "name": kernel_name,
              "tensor_list": [data_x1, res]}

    te.lang.cce.cce_build_code(sch, config)
