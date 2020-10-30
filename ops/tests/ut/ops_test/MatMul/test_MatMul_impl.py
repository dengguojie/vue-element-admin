#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("MatMul", None, None)

case1 = {"params": [{"shape": (6, 2,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 96),"ori_format": "ND"},
                    {"shape": (4, 6,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96, 64),"ori_format": "ND"},
                    {"shape": (64, ), "dtype": "float16", "format": "ND", "ori_shape": (64, ),"ori_format": "ND"},
                    None,
                    {"shape": (4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 64),"ori_format": "ND"},
                    False, False],
         "case_name": "MatMul_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2, 6,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96, 32),"ori_format": "ND"},
                    {"shape": (6, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 96),"ori_format": "ND"},
                    {"shape": (64,), "dtype": "float16", "format": "ND", "ori_shape": (64, ),"ori_format": "ND"},
                    None,
                    {"shape": (4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 64),"ori_format": "ND"},
                    True, True],
         "case_name": "MatMul_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (6, 2,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 96),"ori_format": "ND"},
                    {"shape": (6, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 96),"ori_format": "ND"},
                    {"shape": (64, ), "dtype": "float16", "format": "ND", "ori_shape": (64, ),"ori_format": "ND"},
                    None,
                    {"shape": (4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 64),"ori_format": "ND"},
                    False, True],
         "case_name": "MatMul_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

if __name__ == '__main__':
    ut_case.run("Ascend910")
