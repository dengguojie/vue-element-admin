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

pass_through
"""

from topi.cce import util
from impl import pass_through_forward
from impl import pass_through_backward
from te.utils.op_utils import *

MINI_STRIDE = 1


def check_param(in_dic, out_dic, stride, reverse, kernel_name):
    """
    check validation of input param

    Parameters
    ----------
    in_dic : dict
        shape/dtype/fmt of input
    out_dic : dict
        shape/dtype/fmt of output, should be same shape and type as input
    stride : int32
        c/h/w stride
    reverse : bool
        forward or reverse flag
    kernel_name : str
        kernel name, default value is "pass_through"

    Returns
    -------
    None
    """
    shape_in = in_dic.get("shape")
    dtype_in = in_dic.get("dtype")
    fmt_in = in_dic.get("format")
    shape_out = out_dic.get("shape")
    dtype_out = out_dic.get("dtype")
    fmt_out = out_dic.get("format")

    check_shape(shape_in, param_name="in_dic")
    check_shape(shape_out,param_name="out_dic")
    check_dtype(dtype_in.lower(), ["float16", "float32",
                                   "int8", "uint8",
                                   "int16", "uint16",
                                   "int32", "uint32",
                                   "int64", "uint64"],param_name="in_dic")
    check_dtype(dtype_out.lower(), ["float16", "float32",
                                    "int8", "uint8",
                                    "int16", "uint16",
                                    "int32", "uint32",
                                    "int64", "uint64"],param_name="out_dic")

    if fmt_in.lower() != "nhwc" or fmt_out.lower() != "nhwc":
        error_Info = {}
        error_Info['errCode'] = 'E80015'
        error_Info['op_name'] = 'pass_through'
        error_Info['param_name1'] = 'fmt_in'
        error_Info['param_name2'] = 'fmt_out'
        error_Info['expect_value'] = 'NHWC'
        error_Info['real_value1'] = fmt_in.lower()
        error_Info['real_value2'] = fmt_out.lower()
        raise ValueError(error_Info,"In op[%s], the format of [%s]/[%s] must be [%s], "
                        "but actually is [%s]/[%s]."%(error_Info['op_name'],\
                        error_Info['param_name1'],error_Info['param_name2'],
                        error_Info['expect_value'],error_Info['real_value1'],error_Info['real_value2']))

    if stride < MINI_STRIDE:
        error_Info = {}
        error_Info['errCode'] = 'E81007'
        error_Info['param_name'] = 'stride'
        error_Info['op_name'] = 'pass_through'
        error_Info['real_value'] = stride
        raise ValueError(error_Info,"In op[%s], the parameter [%s] must be greater than 0, "
                        "but actually is [%s]."%(error_Info['op_name'],error_Info['param_name'], \
                        error_Info['real_value']))
    if reverse is True:
        if (shape_in[3] % (stride * stride)) != 0:
            error_Info = {}
            error_Info['errCode'] = 'E81008'
            error_Info['param_name'] = 'C'
            error_Info['op_name'] = 'pass_through'
            error_Info['real_value'] = shape_in[3]
            error_Info['expect_value'] = stride * stride
            raise ValueError(error_Info,"In op[%s], the parameter [%s] must be "
                        "times of stride**2[%s], but actually is [%s]."
                         %(error_Info['op_name'],error_Info['param_name'], \
                        error_Info['expect_value'],error_Info['real_value']))

    else:
        if (shape_in[1] % stride != 0) or (shape_in[2] % stride != 0):
            error_Info = {}
            error_Info['errCode'] = 'E81008'
            error_Info['op_name'] = 'pass_through'
            error_Info['param_value1'] = shape_in[1]
            error_Info['param_value2'] = shape_in[2]
            raise ValueError(error_Info,"In op[%s], the parameter w/h must be "
                             "times of stride[%s], but actually is [%s]/[%s]."
                             %(error_Info['op_name'], stride,error_Info['param_value2'],\
                               error_Info['param_value1']))

@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, OPTION_ATTR_INT,
                 OPTION_ATTR_BOOL, KERNEL_NAME)
def pass_through(in_dic, out_dic, stride, reverse, kernel_name="pass_through"):
    """
    pass_through ops interface

    Parameters
    ----------
    in_dic : dict
        shape/dtype/fmt of input
    out_dic : dict
        shape/dtype/fmt of output, should be same shape and type as input
    stride : int32
        c/h/w stride
    reverse : bool
        forward or reverse flag
    kernel_name : str
        kernel name, default value is "pass_through"

    Returns
    -------
    tik instance
    """

    check_param(in_dic, out_dic, stride, reverse, kernel_name)

    if reverse is False:
        tik_instance, input_gm, output_gm = \
            pass_through_backward.pass_through_backward_func(in_dic, stride)
    else:
        tik_instance, input_gm, output_gm = \
            pass_through_forward.pass_through_forward_func(in_dic, stride)

    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=input_gm,
                          outputs=output_gm)
    return tik_instance
