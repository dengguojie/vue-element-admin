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
import time
import unittest
import functools
from te import tvm
from te.lang.cce import cce_build_code
from te.tvm.target import cce
from topi.generic import auto_schedule

from impl.batch_matmul import get_op_support_info
from impl.batch_matmul import batch_matmul_compute
from impl.confusion_transpose_d import confusion_transpose_d_compute
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


# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
print("==========add case for batchmamtul fusion===============")
for fusion_case in batchmatmul_ut_fusion_case:
    ut_case.add_cust_test_func(["Ascend910", "Ascend310", "Ascend710"],
                               test_func=test_batchmatmul_fusion(fusion_case))

def test_split_batch_matmul(test_arg):
    x1 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16", "shape": (16, 1, 2, 16, 16), "ori_shape": (16, 32, 16)}
    x2 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16", "shape": (16, 2, 16, 16), "ori_shape": (16, 32, 16)}
    get_op_support_info(x1, x2, trans_a=True)

    x1 = {"format": "ND","ori_format": "ND", "dtype": "float16", "shape": (16, 16, 32), "ori_shape": (16, 16, 32)}
    x2 = {"format": "ND","ori_format": "ND", "dtype": "float16", "shape": (16, 32), "ori_shape": (16, 32)}
    get_op_support_info(x1, x2, None, trans_b=True)
ut_case.add_cust_test_func(test_func=test_split_batch_matmul)

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

ut_case.add_cust_test_func(test_func=test_batchmatmul_confusion_transpose_910)
ut_case.add_cust_test_func(test_func=test_batchmatmul_confusion_transpose_710)

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
    exit(0)
