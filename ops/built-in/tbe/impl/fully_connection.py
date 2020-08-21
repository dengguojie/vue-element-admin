#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
copyright 2019 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License == distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

fully_connection
"""
from __future__ import absolute_import
import te.lang.cce
from te import tvm
from te.utils.op_utils import *
from topi.cce import util
from te.platform.fusion_manager import fusion_manager
from topi import generic
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

NoneType = type(None)


# pylint: disable=too-many-arguments,unused-argument,invalid-name,redefined-outer-name
# pylint: disable=too-many-boolean-expressions,too-many-locals,unused-variable
def op_select_format(x, w, b, offset_w, y, num_output, transpose, axis, offset_x,
                     kernel_name="fully_connection"):
    """
    select format dynamically
    """
    if axis == 1:
        input0 = gen_param(classify="input0", name="x",
                           datatype="float16, int8",
                           format="NC1HWC0, NC1HWC0")
        input1 = gen_param(classify="input1", name="w",
                           datatype="float16, int8",
                           format="FRACTAL_Z, FRACTAL_Z")
        input2 = gen_param(classify="input2", name="b",
                           datatype="float16,int32",
                           format="NC1HWC0,NC1HWC0")
        input3 = gen_param(classify="input3", name="offset_w",
                           datatype="int8,int8",
                           format="ND,ND")
        output0 = gen_param(classify="output0", name="y",
                            datatype="float16,int32",
                            format="NC1HWC0,NC1HWC0")
    else:
        input0 = gen_param(classify="input0", name="x",
                           datatype="float16, int8",
                           format="FRACTAL_NZ, FRACTAL_NZ")
        input1 = gen_param(classify="input1", name="w",
                           datatype="float16, int8",
                           format="FRACTAL_Z, FRACTAL_Z")
        input2 = gen_param(classify="input2", name="b",
                           datatype="float16,int32",
                           format="NC1HWC0,NC1HWC0")
        input3 = gen_param(classify="input3", name="offset_w",
                           datatype="int8,int8",
                           format="ND,ND")
        output0 = gen_param(classify="output0", name="y",
                            datatype="float16,int32",
                            format="FRACTAL_NZ, FRACTAL_NZ")

    param_list = [input0, input1, input2, input3, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


# pylint: disable=locally-disabled,too-many-arguments, too-many-locals
# pylint: disable=too-many-statements, invalid-name, unused-argument
def fully_connection_check_rule(x, w, b, offset_w, y, num_output, transpose, axis, offset_x,
                                kernel_name="fully_connection"):
    """check input params"""
    # x info
    shape_x = x.get('shape')
    dtype_x = x.get('dtype')
    format_x = x.get('format')

    if axis == 2:
        km_shape = shape_x[1] * shape_x[-1]
    else:
        km_shape = shape_x[1] * shape_x[2] * shape_x[3] * shape_x[4]


    if shape_x[-1] not in (16, 32):
        raise RuntimeError("for axis = 1: no_quant x C0 must be 16!, quant x C0 must be 32! \
                            for axis = 2: the last dim must be 16!")

    check_dtype(dtype_x, ['float16', 'int8'], param_name="x")
    check_format(format_x, ('NC1HWC0', 'FRACTAL_NZ'), param_name="x")

    # w info
    shape_w = w.get('shape')
    dtype_w = w.get('dtype')
    format_w = w.get('format')

    check_dtype(dtype_w, ['float16', 'int8'], param_name="w")

    check_format(format_w, ["FRACTAL_Z"], param_name="w")
    # format shape info
    if dtype_x == 'float16' and (shape_w[2] != 16 or shape_w[3] != 16):
        raise RuntimeError("for no quant, w last two dims must be 16!")
    if dtype_x == 'int8' and (shape_w[2] != 16 or shape_w[3] != 32):
        raise RuntimeError("for quant, w last two dims must be 16 and 32!")

    kn_shape = shape_w[0] * shape_w[3]
    n_shape = shape_w[1] * shape_w[2]

    # Check shape
    if km_shape != kn_shape:
        raise RuntimeError("KM must equal to KN!")

    # b info
    if b is not None:
        shape_b = b.get('shape')
        dtype_b = b.get('dtype')
        format_b = b.get('format')
        b_size = shape_b[1] * shape_b[4]

        # Check info
        check_dtype(dtype_b, ['float16', 'int32'], param_name="b")
        check_format(format_b, ('NC1HWC0'), param_name="b")
        if b_size != n_shape:
            raise RuntimeError("For bias, the C1*C0 must equal to aligned_Cout!")

    # axis info
    if axis not in (1, 2):
        raise RuntimeError("axis only support 1, 2 when reduce from channel!")


@fusion_manager.register("fully_connection")
def fully_connection_compute(x, w, b, offset_w, y, num_output, transpose, axis, offset_x,
                             kernel_name="fully_connection"):
    """
    x : the tensor of input x

    w: the tensor of intput w

    b: the tensor of bias

    offset_w : unused

    num_output : output neuron number

    transpose: is weight transpose

    axis: the beginning reduce axis, reduce axis is [axis, last_axis]

    offset_x: unused

    Returns:
    -------
    None
    """
    if axis == 2:
        format_a = 'FRACTAL_NZ'
        format_b = 'fractal'
        trans_a=True
    else:
        format_a = 'ND'
        format_b = 'fractal'
        trans_a=False

    quantize_params = None
    if offset_w is not None:
        raise RuntimeError("For FullyConnection, tensor offset_w must be None!")

    result = te.lang.cce.matmul(tensor_a=x, tensor_b=w, trans_a=trans_a, trans_b=transpose,
                                format_a=format_a, format_b=format_b, alpha_num=1.0, beta_num=0.0,
                                dst_dtype='float16', tensor_bias=b, quantize_params=quantize_params)
    return result


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, OPTION_INPUT, OPTION_INPUT,
                REQUIRED_OUTPUT, REQUIRED_ATTR_INT, OPTION_ATTR_BOOL, REQUIRED_ATTR_INT, REQUIRED_ATTR_INT, KERNEL_NAME)
def fully_connection(x, w, b, offset_w, y, num_output, transpose, axis, offset_x,
                     kernel_name="fully_connection"):
    """
    x : the dict of input x

    w: the dict of intput w

    b: the dict of bias

    offset_w : unused

    num_output : output neuron number

    transpose: is weight transpose

    axis: the beginning reduce axis, reduce axis is [axis, last_axis]

    offset_x: unused

    Returns:
    -------
    None
    """
    # Check params
    fully_connection_check_rule(x, w, b, offset_w, y, num_output, transpose, axis, offset_x,
                                kernel_name="fully_connection")

    # x info
    shape_x = x.get('shape')
    dtype_x = x.get('dtype')

    if axis == 2:
        shape_x_final = (shape_x[0], shape_x[1], shape_x[2], shape_x[3], shape_x[4])
    else:
        shape_x_final = (shape_x[0], shape_x[1] * shape_x[2] * shape_x[3] * shape_x[4])


    tensor_x = tvm.placeholder(shape_x_final, dtype=dtype_x, name='tensor_a')

    # w info
    shape_w = w.get('shape')
    dtype_w = w.get('dtype')
    tensor_w = tvm.placeholder(shape_w, dtype=dtype_w, name='tensor_b')

    # b info
    if b is not None:
        shape_b = b.get('shape')
        dtype_b = b.get('dtype')
        shape_bias = (shape_b[1] * shape_b[4],)
        tensor_b = tvm.placeholder(shape_bias, dtype=dtype_b, name='tensor_bias')
    else:
        tensor_b = None

    # offset_w info
    if offset_w is None:
        tensor_offset_w = None
    else:
        raise RuntimeError("offset_w must be None!")

    # Compute
    result = fully_connection_compute(tensor_x, tensor_w, tensor_b, tensor_offset_w, y,
                                      num_output, False, axis, offset_x)

    # Schedule
    with tvm.target.cce():
        schedule = generic.auto_schedule(result)

    # CCE build
    if b is not None:
        tensor_list = [tensor_x, tensor_w, tensor_b, result]
    else:
        tensor_list = [tensor_x, tensor_w, result]

    config = {"print_ir": False, "need_build": True, "need_print": True,
              "name": kernel_name, "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(schedule, config)
