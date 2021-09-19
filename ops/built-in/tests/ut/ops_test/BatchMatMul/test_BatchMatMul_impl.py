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
from op_test_frame.common import precision_info
import numpy as np
import sys
import time
import unittest
import functools
from te import tvm
from te.lang.cce import cce_build_code
from te.tvm.target import cce
from tbe.dsl import auto_schedule

from impl.batch_matmul import get_op_support_info
from impl.batch_matmul import batch_matmul_compute
from impl.batch_matmul import check_supported
from impl.confusion_transpose_d import confusion_transpose_d_compute
from impl.add import add_compute
from impl.relu import relu_compute
from impl.div import div_compute
from impl.fused_mul_add import fusion_mul_add_compute
from te.platform.cce_conf import te_set_version

from batchmatmul_fusion_case import batchmatmul_ut_fusion_case
from test_BatchMatMul_fusion import test_batchmatmul_fusion


ut_case = OpUT("BatchMatMul", "impl.batch_matmul", "batch_matmul")

case1 = {"params": [{"shape": (3, 96, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (3,96, 32),"ori_format": "NHWC"}, #x
                    {"shape": (3, 64, 96), "dtype": "float16", "format": "NHWC", "ori_shape": (3,64, 96),"ori_format": "NHWC"},
                    {"shape": (64,), "dtype": "float16", "format": "NHWC", "ori_shape": (64,),"ori_format": "NHWC"},
                    {"shape": (3, 96, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (3,96, 32),"ori_format": "NHWC"},
                    True,True
                    ],
         "case_name": "BatchMatmul_v1_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (3, 32, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 32, 96),"ori_format": "NHWC"}, #x
                    {"shape": (3, 96, 64), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96, 64),"ori_format": "NHWC"}, #h
                    {"shape": (96,), "dtype": "float32", "format": "NHWC", "ori_shape": (96,),"ori_format": "NHWC"},
                    {"shape": (3, 96, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96, 96),"ori_format": "NHWC"},
                    True,True,
                    ],
         "case_name": "BatchMatmul_v1_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (3, 32, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 32, 128),"ori_format": "NHWC"}, #x
                    {"shape": (3, 128, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 128, 32),"ori_format": "NHWC"},
                    {"shape": (128, ), "dtype": "float16", "format": "NHWC", "ori_shape": (128,),"ori_format": "NHWC"},
                    {"shape": (3, 32, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 32, 128),"ori_format": "NHWC"},
                    True,True,
                    ],
         "case_name": "BatchMatmul_v1_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (3, 112, 64), "dtype": "float32", "format": "ND", "ori_shape": (3, 112, 64),"ori_format": "ND"}, #x
                    {"shape": (3, 64, 112), "dtype": "float32", "format": "ND", "ori_shape": (3, 64, 112),"ori_format": "ND"}, #h
                    {"shape": (64,), "dtype": "float32", "format": "ND", "ori_shape": (64,),"ori_format": "ND"},
                    {"shape": (3, 112, 64), "dtype": "float32", "format": "ND", "ori_shape": (3, 112, 64),"ori_format": "ND"},
                    True,True,
                    ],
         "case_name": "BatchMatmul_v1_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (112, 64), "dtype": "float32", "format": "ND", "ori_shape": (112, 64),"ori_format": "ND"},
                    {"shape": (64, 112), "dtype": "float32", "format": "ND", "ori_shape": (64, 112),"ori_format": "ND"},
                    None,
                    {"shape": (112, 64), "dtype": "float32", "format": "ND", "ori_shape": (112, 64),"ori_format": "ND"},
                    True,True,
                    ],
         "case_name": "BatchMatmul_v1_5",
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape": (8, 4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (8, 4, 32, 64),"ori_format": "ND"},
                    {"shape": (8, 1, 1, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (8, 1, 64, 16),"ori_format": "ND"},
                    None,
                    {"shape": (8, 4, 1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (8, 4, 32, 16),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_v1_6",
         "expect": "success",
         "support_expect": True}

case7 = {"params": [{"shape": (2, 4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2, 4, 32, 64),"ori_format": "ND"},
                    {"shape": (2, 4, 1, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2, 4, 64, 16),"ori_format": "ND"},
                    None,
                    {"shape": (1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 16),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_v1_7",
         "expect": "success",
         "support_expect": True}

case8 = {"params": [{"shape": (2, 4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2, 4, 32, 64),"ori_format": "ND"},
                    {"shape": (4, 1, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4, 64, 16),"ori_format": "ND"},
                    None,
                    {"shape": (1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 16),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_v1_8",
         "expect": "success",
         "support_expect": True}

case9 = {"params": [{"shape": (4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4, 32, 64),"ori_format": "ND"},
                    {"shape": (4, 1, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4, 64, 16),"ori_format": "ND"},
                    None,
                    {"shape": (4, 1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4, 32, 16),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_v1_9",
         "expect": "success",
         "support_expect": True}

case10 = {"params": [{"shape": (0, 10), "ori_shape": (0, 10), "dtype": "float16", "format": "ND", "ori_format": "ND"},
                     {"shape": (10, 20), "ori_shape": (10, 20), "dtype": "float16", "format": "ND", "ori_format": "ND"},
                     None,
                     {"shape": (0, 20), "ori_shape": (0, 20), "dtype": "float16", "format": "ND", "ori_format": "ND"},
                     False, False,
                     ],
         "case_name": "BatchMatmul_v1_10",
         "expect": "success",
         "support_expect": True}

case11 = {"params": [{"shape": (1, 2708, 2708), "dtype": "float32", "format": "ND", "ori_shape": (1, 2708, 2708),"ori_format": "ND"}, #x
                    {"shape": (1, 2708, 7), "dtype": "float32", "format": "ND", "ori_shape": (1, 2708, 7),"ori_format": "ND"}, #h
                    {"shape": (7,), "dtype": "float32", "format": "ND", "ori_shape": (7,),"ori_format": "ND"},
                    {"shape": (1, 2708, 7), "dtype": "float32", "format": "ND", "ori_shape": (1, 2708, 7),"ori_format": "ND"},
                    False, False,
                    ],
         "case_name": "BatchMatmul_v1_11",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case("Ascend920A", case9)
print("==========add case for batchmamtul fusion===============")
for fusion_case in batchmatmul_ut_fusion_case:
   ut_case.add_cust_test_func(["Ascend910", "Ascend310", "Ascend710"],
                              test_func=test_batchmatmul_fusion(fusion_case))

def test_split_batch_matmul(test_arg):
    x1 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16", "shape": (16, 1, 2, 16, 16), "ori_shape": (16, 32, 16)}
    x2 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16", "shape": (16, 1, 2, 16, 16), "ori_shape": (16, 32, 16)}
    get_op_support_info(x1, x2, trans_a=True)

def test_split_batch_matmul_1(test_arg):
    x1 = {"format": "ND","ori_format": "ND", "dtype": "float16", "shape": (16, 16, 32), "ori_shape": (16, 16, 32)}
    x2 = {"format": "ND","ori_format": "ND", "dtype": "float16", "shape": (16, 32), "ori_shape": (16, 32)}
    get_op_support_info(x1, x2, trans_b=True)
ut_case.add_cust_test_func(test_func=test_split_batch_matmul)
ut_case.add_cust_test_func(test_func=test_split_batch_matmul_1)

def test_batchmatmul_confusion_transpose_910(test_arg):
    te_set_version("Ascend910")
    with cce():
        x1 = tvm.placeholder((24*16, 32, 32, 16, 16), name="x1", attrs={'format': "FRACTAL_NZ", "ori_shape": (24, 16, 512, 512)}, dtype="float16")
        x2 = tvm.placeholder((24*16, 4, 32, 16, 16), name="x2", attrs={'format': "FRACTAL_NZ", "ori_shape": (24, 16, 512, 64)}, dtype="float16")
        output_y = {"shape": (24*16, 4, 32, 16, 16), "dtype": "float16", "ori_shape": (24, 16, 512, 64), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = batch_matmul_compute(x1, x2, None, output_y)
        y = {"shape": (64, 768, 16, 16), "ori_shape": (12288, 1024), "dtype": "float16", "format": "FRACTAL_NZ", "ori_format": "ND"}
        out = confusion_transpose_d_compute(matmul_out, y, [0, 2, 1, 3], (12288, 1024), True)
        tensor_list = [x1, x2, out]
        sch = auto_schedule(out)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "batch_matmul_confusion_transpose_910",
            "tensor_list": tensor_list,
        }
        cce_build_code(sch, config)
    te_set_version("Ascend310")

def test_batchmatmul_confusion_transpose_710(test_arg):
    te_set_version("Ascend710")
    with cce():
        x1 = tvm.placeholder((8*12, 8, 8, 16, 16), name="x1", attrs={'format': "FRACTAL_NZ", "ori_shape": (8, 12, 128, 128)}, dtype="float16")
        x2 = tvm.placeholder((8*12, 4, 8, 16, 16), name="x2", attrs={'format': "FRACTAL_NZ", "ori_shape": (8, 12, 128, 64)}, dtype="float16")
        output_y = {"shape": (8*12, 4, 8, 16, 16), "dtype": "float16", "ori_shape": (8, 12, 128, 64), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = batch_matmul_compute(x1, x2, None, output_y)
        y = {"shape": (48, 64, 16, 16), "ori_shape": (1024, 768), "dtype": "float16", "format": "FRACTAL_NZ", "ori_format": "ND"}
        out = confusion_transpose_d_compute(matmul_out, y, [0, 2, 1, 3], (1024, 768), True)
        tensor_list = [x1, x2, out]
        sch = auto_schedule(out)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "batch_matmul_confusion_transpose_710",
            "tensor_list": tensor_list,
        }
        cce_build_code(sch, config)
    te_set_version("Ascend310")

def test_batchmatmul_add(test_arg):
    with cce():
        x1 = tvm.placeholder((16, 20, 6, 16, 16), name="x1", attrs={'format': "FRACTAL_NZ", "ori_shape": (16, 96, 320)}, dtype="float16")
        x2 = tvm.placeholder((20, 20, 16, 16), name="x2", attrs={'format': "FRACTAL_NZ", "ori_shape": (320, 320)}, dtype="float16")
        output_y = {"shape": (16, 20, 6, 16, 16), "dtype": "float16", "ori_shape": (16, 96, 320), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = batch_matmul_compute(x1, x2, None, output_y)

        add_tensor = tvm.placeholder((320, ), name='tensor_add', dtype="float16", attrs={"format": "ND", "ori_shape": (320,)})
        out = add_compute(matmul_out, add_tensor, {})

        tensor_list = [x1, x2, add_tensor, out]
        sch = auto_schedule(out)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "batch_matmul_add",
            "tensor_list": tensor_list,
        }
        cce_build_code(sch, config)

def test_batchmatmul_add_add(test_arg):
    with cce():
        x1 = tvm.placeholder((16, 20, 6, 16, 16), name="x1", attrs={'format': "FRACTAL_NZ", "ori_shape": (16, 96, 320)}, dtype="float16")
        x2 = tvm.placeholder((20, 20, 16, 16), name="x2", attrs={'format': "FRACTAL_NZ", "ori_shape": (320, 320)}, dtype="float16")
        output_y = {"shape": (16, 20, 6, 16, 16), "dtype": "float16", "ori_shape": (16, 96, 320), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = batch_matmul_compute(x1, x2, None, output_y)

        add_tensor = tvm.placeholder((320, ), name='tensor_add', dtype="float16", attrs={"format": "ND", "ori_shape": (320,)})
        add_tensor1 = tvm.placeholder((16, 20, 6, 16, 16), name='tensor_add1', dtype="float16", attrs={"format": "FRACTAL_NZ", "ori_shape": (16, 320, 320)})
        out = add_compute(matmul_out, add_tensor, {})
        out = add_compute(out, add_tensor1, {})

        tensor_list = [x1, x2, add_tensor, add_tensor1, out]
        sch = auto_schedule(out)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "batch_matmul_add_add",
            "tensor_list": tensor_list,
        }
        cce_build_code(sch, config)

def test_batchmatmul_add_relu(test_arg):
    with cce():
        x1 = tvm.placeholder((16, 20, 6, 16, 16), name="x1", attrs={'format': "FRACTAL_NZ", "ori_shape": (16, 96, 320)}, dtype="float16")
        x2 = tvm.placeholder((128, 20, 16, 16), name="x2", attrs={'format': "FRACTAL_NZ", "ori_shape": (320, 2048)}, dtype="float16")
        output_y = {"shape": (16, 128, 6, 16, 16), "dtype": "float16", "ori_shape": (16, 96, 2048), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = batch_matmul_compute(x1, x2, None, output_y)

        add_tensor = tvm.placeholder((2048, ), name='tensor_add', dtype="float16", attrs={"format": "ND", "ori_shape": (2048, )})
        out = add_compute(matmul_out, add_tensor, {})
        out = relu_compute(out, {})

        tensor_list = [x1, x2, add_tensor, out]
        sch = auto_schedule(out)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "batch_matmul_add_relu",
            "tensor_list": tensor_list,
        }
        cce_build_code(sch, config)

def test_batchmatmul_div_fused_mul_add(test_arg):
    with cce():
        x1 = tvm.placeholder((16*4, 5, 6, 16, 16), name="x1", attrs={'format': "FRACTAL_NZ", "ori_shape": (16, 4, 96, 80)}, dtype="float16")
        x2 = tvm.placeholder((16*4, 6, 5, 16, 16), name="x2", attrs={'format': "FRACTAL_NZ", "ori_shape": (16, 4, 80, 96)}, dtype="float16")
        output_y = {"shape": (16*4, 6, 6, 16, 16), "dtype": "float16", "ori_shape": (16,4, 96, 96), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = batch_matmul_compute(x1, x2, None, output_y)

        div_tensor = tvm.placeholder((1, 1, 1 ,1 ,1, 1), name='tensor_div', dtype="float16", attrs={"format": "ND", "ori_shape": (1,)})
        mul_tensor = tvm.placeholder((16, 1, 6, 6 ,16, 16), name='tensor_mul', dtype="float16", attrs={"format": "FRACTAL_NZ", "ori_shape": (16, 1, 96, 96)})
        add_tensor =  tvm.placeholder((16, 1, 6, 6 ,16, 16), name='tensor_add', dtype="float16", attrs={"format": "FRACTAL_NZ", "ori_shape": (16, 1, 96, 96)})
        out = div_compute(matmul_out, div_tensor, {})
        out = fusion_mul_add_compute(out, mul_tensor, add_tensor, {})

        tensor_list = [x1, x2, div_tensor, mul_tensor, add_tensor, out]
        sch = auto_schedule(out)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "batch_matmul_div_fused_mul_add",
            "tensor_list": tensor_list,
        }
        cce_build_code(sch, config)


def test_op_check_supported(test_arg):
    def _test_supported(case):
        input_x, input_y, bias, output_z, trans_a, trans_b = case["params"]
        try:
            check_supported(input_x, input_y, bias, output_z, trans_a, trans_b, kernel_name="batch_matmul")
        except RuntimeError:
            print("The case is not supported!")
            pass

    _test_supported(case1)
    _test_supported(case2)
    _test_supported(case3)
    _test_supported(case4)
    _test_supported(case5)
    _test_supported(case10)

def test_op_select_format(test_arg):
    from impl.batch_matmul import op_select_format
    # static shape
    op_select_format({"shape": (3, 2, 4), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 4), "ori_format": "ND"},
                     {"shape": (3, 4, 5), "dtype": "float16", "format": "ND", "ori_shape": (4, 5), "ori_format": "ND"},
                     )
    op_select_format({"shape": (3, 2, 4), "dtype": "float", "format": "ND", "ori_shape": (3, 2, 4), "ori_format": "ND"},
                     {"shape": (1, 4, 5), "dtype": "float", "format": "ND", "ori_shape": (1, 4, 5), "ori_format": "ND"},
                     )

ut_case.add_cust_test_func(test_func=test_op_select_format)
ut_case.add_cust_test_func(test_func=test_batchmatmul_confusion_transpose_910)
ut_case.add_cust_test_func(test_func=test_batchmatmul_confusion_transpose_710)
#ut_case.add_cust_test_func(test_func=test_batchmatmul_add)
#ut_case.add_cust_test_func(test_func=test_batchmatmul_add_add)
#ut_case.add_cust_test_func(test_func=test_batchmatmul_add_relu)
#ut_case.add_cust_test_func(test_func=test_batchmatmul_div_fused_mul_add)
ut_case.add_cust_test_func(test_func=test_op_check_supported)


def reshape_nd_to_nz(tensor1, shape):
    if len(tensor1.shape) < 2:
        return
    j_outer, i_outer, i_inner, j_inner = shape[-4:]
    batch_shapes = shape[:-4]
    batch_len = len(batch_shapes)

    reshape_tensor1_size = list(batch_shapes) + [i_outer, i_inner, j_outer, j_inner]
    transpose_idx = [x for x in range(batch_len)] + \
                    [batch_len+2, batch_len+0, batch_len+1, batch_len+3]
    tensor1 = np.reshape(tensor1, reshape_tensor1_size).transpose(transpose_idx)
    return tensor1


def reshape_nz_to_nd(tensor1):
    shape = tensor1.shape
    if len(shape) < 4:
        return
    batch_shapes = shape[:-4]
    batch_len = len(batch_shapes)
    j_outer, i_outer, i_inner, j_inner = shape[-4:]

    transpose_idx = [x for x in range(batch_len)] + \
                    [batch_len+1, batch_len+2, batch_len+0, batch_len+3]
    reshape_tensor1_size = list(batch_shapes) + [i_outer*i_inner, j_outer*j_inner]
    tensor1 = np.transpose(tensor1, transpose_idx).reshape(reshape_tensor1_size)
    return tensor1

def calc_expect_func(input_x, input_y, bias, output_z, trans_a, trans_b):
    a = reshape_nz_to_nd(input_x["value"])
    b = reshape_nz_to_nd(input_y["value"])
    if bias:
        res = np.matmul(a, b) + bias["value"]
    else:
        res = np.matmul(a, b)
    res = reshape_nd_to_nz(res, output_z["shape"]).astype(output_z["dtype"])
    return res

precision_case1 = {"params": [{"shape": (16,1,1,16, 2,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16,1,1,16, 16,32), "ori_format": "ND", "param_type": "input"},
                              {"shape": (4,6,1, 1,2,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4,6,1, 32,16), "ori_format": "ND", "param_type": "input"},
                              None,
                              {"shape": (16,4,6,16, 1,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16,4,6,16, 16,16), "ori_format": "ND", "param_type": "output"},
                              False, False],
                              "expect": "success",
                              "calc_expect_func": calc_expect_func,
                              "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

ut_case.add_precision_case("Ascend910", precision_case1)


case15 = {"params": [{"shape": (4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4, 32, 64),"ori_format": "ND"},
                     {"shape": (4, 4, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4, 64, 49),"ori_format": "ND"},
                     {"shape": (49,), "dtype": "float32", "format": "ND", "ori_shape": (49,),"ori_format": "ND"},
                     {"shape": (4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (
                         4, 32, 49), "ori_format": "ND"},
                     False, False,
                    ],
         "case_name": "BatchMatmul_v2_15",
         "expect": "success",
         "support_expect": True}
ut_case.add_case(["Ascend920A"], case15)

if __name__ == '__main__':
    ut_case._case_info_map = {}
    ut_case.run("Ascend910")
    # ut_case.run(["Ascend310", "Ascend710", "Ascend910A", "Ascend920A"],
    #             simulator_mode="pv", simulator_lib_path="../../Ascend/toolkit/tools/simulator")
