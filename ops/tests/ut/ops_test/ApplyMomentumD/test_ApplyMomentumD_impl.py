#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("ApplyMomentumD", None, None)

case1 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}],
         "case_name": "apply_momentum_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}],
         "case_name": "apply_momentum_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (0,), "dtype": "float32", "format": "ND", "ori_shape": (0,),"ori_format": "ND"},
                    {"shape": (0,), "dtype": "float32", "format": "ND", "ori_shape": (0,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (0,), "dtype": "float32", "format": "ND", "ori_shape": (0,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (0,), "dtype": "float32", "format": "ND", "ori_shape": (0,),"ori_format": "ND"},
                    {"shape": (0,), "dtype": "float32", "format": "ND", "ori_shape": (0,),"ori_format": "ND"}],
         "case_name": "apply_momentum_d_3",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (0.77,), "dtype": "float32", "format": "ND", "ori_shape": (0.77,),"ori_format": "ND"},
                    {"shape": (0.77,), "dtype": "float32", "format": "ND", "ori_shape": (0.77,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (0.77,), "dtype": "float32", "format": "ND", "ori_shape": (0.77,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (0.77,), "dtype": "float32", "format": "ND", "ori_shape": (0.77,),"ori_format": "ND"},
                    {"shape": (0.77,), "dtype": "float32", "format": "ND", "ori_shape": (0.77,),"ori_format": "ND"}],
         "case_name": "apply_momentum_d_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)


if __name__ == '__main__':
    ut_case.run("Ascend910")



