"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

renorm
"""

import te.lang.cce as tbe
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils import para_check


# 'pylint: disable=not-use-list-comprehension,unused-argument,too-many-locals
@fusion_manager.register("renorm")
def renorm_compute(input_x, output_y, p, dim, maxnorm, kernel_name="renorm"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    p: float
        specify L_p norm
    dim: int
        the processed dim
    maxnorm: float
        threshold for comparison
    kernel_name : str
        kernel name, default value is "renorm"

    Returns
    -------
    None
    """
    dtype = input_x.dtype
    shape = input_x.shape
    dims = len(shape)
    need_cast = False
    if dtype == "float16":
        need_cast = True
    if need_cast:
        input_x = tbe.cast_to(input_x, "float32")
    const_maxnorm = tvm.const(maxnorm, "float32")
    ext = tvm.const(1e-7, "float32")

    shape_list = []
    for i in range(dims):
        if i != dim:
            shape_list = shape_list + [i]
    if p == 1:
        x_sum = tbe.sum(tbe.vabs(input_x), shape_list, keepdims=True)
        x_l1norm = tbe.vmins(x_sum, const_maxnorm)
        ratio = tbe.vdiv(x_l1norm, tbe.vadds(x_sum, ext))
    elif p == 2:
        x_square = tbe.vmul(input_x, input_x)
        x_square_sum = tbe.sum(x_square, shape_list, keepdims=True)
        x_l2norm_sqrt = tbe.vsqrt(x_square_sum)
        x_l2norm = tbe.vmins(x_l2norm_sqrt, const_maxnorm)
        ratio = tbe.vdiv(x_l2norm, tbe.vadds(x_l2norm_sqrt, ext))
    else:
        if p == 0:
            zero_scalar = tvm.const(0, 'float32')
            one_scalar = tvm.const(1, 'float32')
            tmp = tbe.vcmpsel(input_x, one_scalar, 'ne', one_scalar, zero_scalar)
            x_tmp = tbe.cast_to(tbe.sum(tmp, shape_list, keepdims=True), 'float32')
        else:
            p_log = tbe.vlog(tbe.vabs(input_x))
            p_mul = tbe.vmuls(p_log, p)
            x_sum = tbe.vexp(p_mul)
            x_psum = tbe.sum(x_sum, shape_list, keepdims=True)
            p_log_v = tbe.vlog(x_psum)
            p_mul_v = tbe.vmuls(p_log_v, 1 / p)
            x_tmp = tbe.vexp(p_mul_v)
        x_lpnorm_p = tbe.vmins(x_tmp, const_maxnorm)
        x_tmp_ext = tbe.vadds(x_tmp, ext)
        ratio = tbe.vdiv(x_lpnorm_p, x_tmp_ext)

    if need_cast:
        ratio = tbe.cast_to(ratio, "float16")

    return ratio


# 'pylint: disable=not-use-list-comprehension,too-many-arguments,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_FLOAT, para_check.KERNEL_NAME)
def renorm(input_x, output_y, p, dim, maxnorm, kernel_name="renorm"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    p: float
        specify L_p norm
    dim: int
        the processed dim
    maxnorm: float
        threshold for comparison
    kernel_name : str
        kernel name, default value is "renorm"

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")
    input_dtype = dtype.lower()

    para_check.check_shape_rule(shape)
    para_check.check_shape_size(shape)
    para_check.check_kernel_name(kernel_name)
    check_tuple = ("float16", "float32")
    if input_dtype not in check_tuple:
        raise RuntimeError("Only support %s while dtype is %s" % (",".join(check_tuple), input_dtype))
    if p < 0:
        raise RuntimeError("Only support p >= 0 while p is {}".format(p))
    dims = len(shape)
    ne_dims = dims * -1
    if dim < ne_dims or dim > (dims - 1):
        raise RuntimeError("only support {} =< dim <= {} while dim is {}".format(ne_dims, dims - 1, dim))
    if dim < 0:
        dim = dim + dims

    data_input = tvm.placeholder(shape, name="data_input", dtype=input_dtype)
    res = renorm_compute(data_input, output_y, p, dim, maxnorm, kernel_name)
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input, res], "bool_storage_as_1bit": False}

    tbe.build(schedule, config)
