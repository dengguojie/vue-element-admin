#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("SplitVD", "impl.split_v_d", "split_v_d")

case1 = {"params": [{"shape": (1024, 1024, 256), "dtype": "uint16", "format": "NCHW", "ori_shape": (1024, 1024, 256),"ori_format": "NCHW"},
                    [], [1024], -3, 1],
         "case_name": "split_d_v_1",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (1024, 1024, 256), "dtype": "int32", "format": "NCHW", "ori_shape": (1024, 1024, 256),"ori_format": "NCHW"},
                    [], [512, 512], -2, 2],
         "case_name": "split_d_v_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (3, 1, 16, 16), "dtype": "int32", "format": "FRACTAL_NZ",
                     "ori_shape": (3, 1, 16, 16),"ori_format": "FRACTAL_NZ"},
                    [{"shape": (2, 1, 16, 16), "dtype": "int32", "format": "FRACTAL_NZ",
                     "ori_shape": (2, 1, 16, 16),"ori_format": "FRACTAL_NZ"},
                     {"shape": (1, 1, 16, 16), "dtype": "int32", "format": "FRACTAL_NZ",
                     "ori_shape": (1, 1, 16, 16),"ori_format": "FRACTAL_NZ"}], [2, 1], 0, 2],
         "case_name": "split_d_v_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (3, 1, 16, 16), "dtype": "int32", "format": "FRACTAL_NZ",
                     "ori_shape": (16, 48),"ori_format": "ND"},
                    [{"shape": (2, 1, 16, 16), "dtype": "int32", "format": "FRACTAL_NZ",
                     "ori_shape": (16, 32),"ori_format": "ND"},
                     {"shape": (1, 1, 16, 16), "dtype": "int32", "format": "FRACTAL_NZ",
                     "ori_shape": (16, 16),"ori_format": "ND"}], [32, 16], 1, 2],
         "case_name": "split_d_v_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)

if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)