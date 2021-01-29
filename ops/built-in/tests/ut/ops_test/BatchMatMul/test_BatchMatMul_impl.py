"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

BatchMatmul ut case
"""
from op_test_frame.ut import OpUT
import sys
import sys
import time
import unittest
import functools

import te.platform as tbe_platform
from batchmatmul_fusion_case import batchmatmul_ut_fusion_case
from tbe.dsl.static_schedule import cce_build_code
from te.tvm.target import cce
from topi.generic import auto_schedule
from te import tvm
from te.utils import shape_util
from impl.batch_matmul import batch_matmul_compute_self
from impl.batch_matmul import _get_input_shape
from impl.batch_matmul import _get_bias
from impl.reduce_sum_d import reduce_sum_d_compute
from te.utils import para_check


ut_case = OpUT("BatchMatMul", "impl.batch_matmul", "batch_matmul")

case1 = {"params": [{"shape": (3, 96, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (3,96, 32),"ori_format": "NHWC"}, #x
                    {"shape": (3, 64, 96), "dtype": "float16", "format": "NHWC", "ori_shape": (3,64, 96),"ori_format": "NHWC"},
                    {"shape": (64,), "dtype": "float16", "format": "NHWC", "ori_shape": (64,),"ori_format": "NHWC"},
                    {"shape": (3, 96, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (3,96, 32),"ori_format": "NHWC"},
                    True,True
                    ],
         "case_name": "BatchMatmul_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (3, 32, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 32, 96),"ori_format": "NHWC"}, #x
                    {"shape": (3, 96, 64), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96, 64),"ori_format": "NHWC"}, #h
                    {"shape": (96,), "dtype": "float32", "format": "NHWC", "ori_shape": (96,),"ori_format": "NHWC"},
                    {"shape": (3, 96, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96, 96),"ori_format": "NHWC"},
                    True,True,
                    ],
         "case_name": "BatchMatmul_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (3, 32, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 32, 128),"ori_format": "NHWC"}, #x
                    {"shape": (3, 128, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 128, 32),"ori_format": "NHWC"},
                    {"shape": (128, ), "dtype": "float16", "format": "NHWC", "ori_shape": (128,),"ori_format": "NHWC"},
                    {"shape": (3, 32, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 32, 128),"ori_format": "NHWC"},
                    True,True,
                    ],
         "case_name": "BatchMatmul_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (3, 112, 64), "dtype": "float32", "format": "ND", "ori_shape": (3, 112, 64),"ori_format": "ND"}, #x
                    {"shape": (3, 64, 112), "dtype": "float32", "format": "ND", "ori_shape": (3, 64, 112),"ori_format": "ND"}, #h
                    {"shape": (64,), "dtype": "float32", "format": "ND", "ori_shape": (64,),"ori_format": "ND"},
                    {"shape": (3, 112, 64), "dtype": "float32", "format": "ND", "ori_shape": (3, 112, 64),"ori_format": "ND"},
                    True,True,
                    ],
         "case_name": "BatchMatmul_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (112, 64), "dtype": "float32", "format": "ND", "ori_shape": (112, 64),"ori_format": "ND"},
                    {"shape": (64, 112), "dtype": "float32", "format": "ND", "ori_shape": (64, 112),"ori_format": "ND"},
                    None,
                    {"shape": (112, 64), "dtype": "float32", "format": "ND", "ori_shape": (112, 64),"ori_format": "ND"},
                    True,True,
                    ],
         "case_name": "BatchMatmul_5",
         "expect": "success",
         "support_expect": True}


# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)

def get_batchmatmul_node(case):
    """
    get out put node of batchmatmul
    """
    input_x,input_y,bias,output_z,trans_a,trans_b = case[1:7]

    shape_a = input_x.get("ori_shape")
    shape_b = input_y.get("ori_shape")
    shape_a_length = len(shape_a)
    shape_b_length = len(shape_b)
    if shape_a is not None:
        if shape_a_length < 2:
            shape_a = input_x.get("shape")

    if shape_b is not None:
        if shape_b_length < 2:
            shape_b = input_y.get("shape")
    shape_bias = ()
    if bias is not None and bool(bias):
        shape_bias = bias.get("shape")
        shape_bias = list(shape_bias)
        if input_x.get("format") == "FRACTAL_NZ":
            shape_bias = _get_bias(shape_bias)

    src_dtype = input_x.get("dtype").lower()
    dst_dtype = output_z.get("dtype").lower()
    is_fractal = False

    shape_a = list(shape_a)
    shape_b = list(shape_b)
    if input_x.get("format") == "FRACTAL_NZ":
        shape_a = _get_input_shape(shape_a)
        shape_b = _get_input_shape(shape_b)

    trans_a_local = trans_a
    trans_b_local = trans_b

    if input_x.get("format") == "FRACTAL_NZ":
        batch_axis = shape_a[:(len(shape_a) - 2)]
        shape_a = batch_axis + [shape_a[len(shape_a) - 1], shape_a[len(shape_a) - 2]]
        trans_a_local = bool(1 - trans_a)

    if input_y.get("format") == "FRACTAL_NZ":
        batch_axis = shape_b[:(len(shape_b) - 2)]
        shape_b = batch_axis + [shape_b[len(shape_b) - 1], shape_b[len(shape_b) - 2]]
        trans_b_local = bool(1 - trans_b)

    inp_src_dtype = src_dtype.lower()
    m_shape = shape_a[len(shape_a) - 2]
    km_shape = shape_a[len(shape_a) - 1]
    kn_shape = shape_b[len(shape_b) - 2]
    n_shape = shape_b[len(shape_b) - 1]

    if inp_src_dtype == "float16":
        block_reduce = tbe_platform.BLOCK_REDUCE

    block_in = tbe_platform.BLOCK_IN
    block_out = tbe_platform.BLOCK_OUT

    if trans_a and km_shape == 1:
        block_in = tbe_platform.BLOCK_VECTOR

    if not trans_a and m_shape == 1:
        block_in = tbe_platform.BLOCK_VECTOR

    if trans_b and kn_shape == 1:
        block_out = tbe_platform.BLOCK_VECTOR

    if not trans_b and n_shape == 1:
        block_out = tbe_platform.BLOCK_VECTOR

    if trans_a:
        shape_a_dup = (m_shape // block_reduce, km_shape // block_in, block_reduce, block_in)
    else:
        shape_a_dup = (m_shape // block_in, km_shape // block_reduce, block_in, block_reduce)

    if trans_b:
        shape_b_dup = (kn_shape // block_out, n_shape // block_reduce, block_reduce, block_out)
    else:
        shape_b_dup = (kn_shape // block_reduce, n_shape // block_out, block_out, block_reduce)

    if input_x.get("format") == "FORMAT_FRACTAL_Z":
        shape_a_dup = (shape_a_dup[0], shape_a_dup[1], shape_a_dup[2], shape_a_dup[3])
        format_a = "fractal"
    elif input_x.get("format") == "FRACTAL_NZ":
        shape_a_dup = (shape_a_dup[0], shape_a_dup[1], shape_a_dup[2], shape_a_dup[3])
        format_a = "FRACTAL_NZ"
    else:
        shape_a_dup = (shape_a[len(shape_a) - 2], shape_a[len(shape_a) - 1])
        format_a = "ND"

    if input_y.get("format") == "FORMAT_FRACTAL_Z":
        shape_b_dup = (shape_b_dup[0], shape_b_dup[1], shape_b_dup[2], shape_b_dup[3])
        format_b = "fractal"
    elif input_y.get("format") == "FRACTAL_NZ":
        shape_b_dup = (shape_b_dup[0], shape_b_dup[1], shape_b_dup[2], shape_b_dup[3])
        format_b = "FRACTAL_NZ"
    else:
        shape_b_dup = (shape_b[len(shape_b) - 2], shape_b[len(shape_b) - 1])
        format_b = "ND"

    batch_shape_a = None
    if len(shape_a) > 2:
        batch_shape_a = functools.reduce(lambda x, y: x * y, shape_a[:-2])

    batch_shape_b = None
    if len(shape_b) > 2:
        batch_shape_b = functools.reduce(lambda x, y: x * y, shape_b[:-2])

    if len(shape_a) >= len(shape_b):
        batch_shape = batch_shape_a
    else:
        batch_shape = batch_shape_b

    if batch_shape is not None and batch_shape >= 1:
        if is_fractal:
            if batch_shape_a is not None:
                shape_a_dup = (batch_shape_a,) + shape_a_dup
            if batch_shape_b is not None:
                shape_b_dup = (batch_shape_b,) + shape_b_dup
        else:
            if batch_shape_a is not None:
                shape_a_dup = (batch_shape_a,) + shape_a_dup
            if batch_shape_b is not None:
                shape_b_dup = (batch_shape_b,) + shape_b_dup

    tensor_bias = None
    shape_bias_length = len(shape_bias)
    if shape_bias_length <= 2:
        shape_bias_dup = shape_bias
    else:
        shape_bias_dup = (shape_bias[len(shape_bias) - 2], shape_bias[len(shape_bias) - 1])
        bias_batch_size = functools.reduce(lambda x, y: x * y, shape_bias[:-2])
        shape_bias_dup = (bias_batch_size,) + shape_bias_dup

    tensor_a = tvm.placeholder(shape_a_dup, name='tensor_a',
                               attrs={'format': format_a},
                               dtype=inp_src_dtype)
    tensor_b = tvm.placeholder(shape_b_dup, name='tensor_b',
                               attrs={'format': format_b},
                               dtype=inp_src_dtype)

    if shape_bias_length > 0:
        tensor_bias = tvm.placeholder(shape_bias_dup, name='tensor_bias',
                                      dtype=dst_dtype)
    result = batch_matmul_compute_self(tensor_a, tensor_b, tensor_bias,
                                       output_z, trans_a, trans_b)
    tensor_list = [tensor_a, tensor_b, result]

    if shape_bias_length > 0:
        tensor_list = [tensor_a, tensor_b, tensor_bias, result]

    return result, tensor_list  


def test_batchmatmul_fusion(fusion_case):
    def test_batchmatmul_fusion_ub(test_args):
        with cce():
            outs, tensor_list = get_batchmatmul_node(fusion_case)
            y = fusion_case[7]
            axis = fusion_case[8]
            keepdims = None
            x = fusion_case[4]
            shape = x.get("shape")
            dtype = x.get("dtype")
            format_x = x.get("format")
            format_y = y.get("format")
            format_ori_y = y.get("ori_format")
            dtype_lower = dtype.lower()

            axis_d = []
            shape_len = len(shape)
            if not axis:
                for i, _ in enumerate(shape):
                    axis_d.append(i)
            else:
                axis_d = list(axis)
            axis_d = shape_util.axis_check(shape_len, axis_d)
            # 5HD Special param for 5hd schedule
            is_nz_nd = False
            if format_x == "FRACTAL_NZ" and format_ori_y == format_y:
                is_nz_nd = True
            is_5hdc = para_check.check_and_init_5hdc_reduce_support(x, axis)

            if not keepdims and not is_5hdc:
                shape, axis_d = shape_util.shape_refine(list(shape), axis_d, keepdims)
                shape, axis_d = shape_util.simplify_axis_shape(shape, axis_d)

            res = reduce_sum_d_compute(outs, y, axis_d, keepdims,
                                        is_5hdc=is_5hdc, is_nz_nd=is_nz_nd)
            if is_5hdc:
                res.ori_shape = x["ori_shape"]
                res.ori_format = x["ori_format"]

            tensor_list.append(res)
            sch = auto_schedule(res)
            config = {
                "print_ir":False,
                "need_build":True,
                "name":"batchmatmul_reducesum_fusion",
                "tensor_list":tensor_list,
            }
            cce_build_code(sch, config)

    return test_batchmatmul_fusion_ub


for fusion_case in batchmatmul_ut_fusion_case:
    print("==========add case for batchmamtul fusion===============")
    print("the fusion_case is ", fusion_case)
    ut_case.add_cust_test_func(
        fusion_case[0], test_func=test_batchmatmul_fusion(fusion_case)
    )
if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
    exit(0)
