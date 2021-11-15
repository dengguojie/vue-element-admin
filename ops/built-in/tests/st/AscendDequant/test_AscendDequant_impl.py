#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from te import tvm

from op_test_frame.ut import OpUT
from impl.ascend_dequant import _matmul_compute
ut_case = OpUT("AscendDequant", None, None)

case1 = {"params": [{"shape": (1,1,1,1,16), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,1,1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,1,1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NC1HWC0"}],
         "case_name": "ascend_dequant_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,2,4,4,16), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (1,2,4,4,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,1,1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,2,4,4,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,2,4,4,16),"ori_format": "NC1HWC0"}],
         "case_name": "ascend_dequant_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,4,4,16,16), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (1,4,4,16,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,1,1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,4,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,4,4,16,16),"ori_format": "NC1HWC0"}],
         "case_name": "ascend_dequant_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2,1,1,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (2,4,4),"ori_format": "NC1HWC0"},
                    {"shape": (1,1,1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (2,1,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1,4,4,16,16),"ori_format": "NC1HWC0"}],
         "case_name": "ascend_dequant_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

def test_matmul_dequant_compute(test_arg):
    x = tvm.placeholder((4, 4, 16, 16), name="matmul_input", attrs={'format': "FRACTAL_NZ", "ori_shape": (64, 64)}, dtype="int32")
    x_shape = (4, 4, 16, 16)
    deq_scale = tvm.placeholder((1, 4, 1, 1, 16), name="deq_tensor", attrs={'format': "NC1HWC0", "ori_shape": (1, 64, 1, 1)}, dtype="uint64")
    _matmul_compute(x, x_shape, deq_scale, False, False, (64, 64), 0, True)

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_cust_test_func(test_func=test_matmul_dequant_compute)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)


