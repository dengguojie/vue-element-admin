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

case5 = {"params": [{"shape": (9973, 256), "dtype": "float16", "format": "ND",
                     "ori_shape": (9973, 256),"ori_format": "ND"},
                    [{"shape": (9973, 80), "dtype": "float16", "format": "ND",
                     "ori_shape": (9973, 80),"ori_format": "ND"},
                     {"shape": (9973, 80), "dtype": "float16", "format": "ND",
                     "ori_shape": (9973, 80),"ori_format": "ND"},
                     {"shape": (9973, 80), "dtype": "float16", "format": "ND",
                     "ori_shape": (9973, 80),"ori_format": "ND"},
                     {"shape": (9973, 1), "dtype": "float16", "format": "ND",
                     "ori_shape": (9973, 1),"ori_format": "ND"},
                     {"shape": (9973, 1), "dtype": "float16", "format": "ND",
                     "ori_shape": (9973, 1),"ori_format": "ND"},
                     {"shape": (9973, 1), "dtype": "float16", "format": "ND",
                     "ori_shape": (9973, 1),"ori_format": "ND"},
                     {"shape": (9973, 13), "dtype": "float16", "format": "ND",
                     "ori_shape": (9973, 13),"ori_format": "ND"},], [80, 80, 80, 1, 1, 1, 13], -1, 7],
         "case_name": "split_d_v_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case6 = {"params": [{"shape": (16, 52, 52, 3, 85), "dtype": "float16", "format": "ND",
                     "ori_shape": (16, 52, 52, 3, 85),"ori_format": "ND"},
                     [{"shape": (16, 52, 52, 3, 1), "dtype": "float16", "format": "ND",
                     "ori_shape": (16, 52, 52, 3, 1),"ori_format": "ND"},
                     {"shape": (16, 52, 52, 3, 2), "dtype": "float16", "format": "ND",
                     "ori_shape": (16, 52, 52, 3, 2),"ori_format": "ND"},
                     {"shape": (16, 52, 52, 3, 2), "dtype": "float16", "format": "ND",
                     "ori_shape": (16, 52, 52, 3, 2),"ori_format": "ND"},
                     {"shape": (16, 52, 52, 3, 80), "dtype": "float16", "format": "ND",
                     "ori_shape": (16, 52, 52, 3, 80),"ori_format": "ND"},], [1,2,2,80],-1, 4],
         "case_name": "split_d_v_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [{"shape": (1, 1, 13, 3, 85), "dtype": "float16", "format": "ND",
                     "ori_shape": (1, 1, 13, 3, 85),"ori_format": "ND"},
                     [{"shape": (1, 1, 13, 3, 1), "dtype": "float16", "format": "ND",
                     "ori_shape": (1, 1, 13, 3, 1),"ori_format": "ND"},
                     {"shape": (1, 1, 13, 3, 2), "dtype": "float16", "format": "ND",
                     "ori_shape": (1, 1, 13, 3, 2),"ori_format": "ND"},
                     {"shape": (1, 1, 13, 3, 2), "dtype": "float16", "format": "ND",
                     "ori_shape": (1, 1, 13, 3, 2),"ori_format": "ND"},
                     {"shape": (1, 1, 13, 3, 80), "dtype": "float16", "format": "ND",
                     "ori_shape": (1, 1, 13, 3, 80),"ori_format": "ND"},], [1,2,2,80],-1, 4],
         "case_name": "split_d_v_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case8 = {"params": [{"shape": (6, 52, 52, 3, 85), "dtype": "float32", "format": "ND",
                     "ori_shape": (6, 52, 52, 3, 85),"ori_format": "ND"},
                     [{"shape": (6, 52, 52, 3, 2), "dtype": "float32", "format": "ND",
                     "ori_shape": (6, 52, 52, 3, 2),"ori_format": "ND"},
                     {"shape": (6, 52, 52, 3, 2), "dtype": "float32", "format": "ND",
                     "ori_shape": (6, 52, 52, 3, 2),"ori_format": "ND"},
                     {"shape": (6, 52, 52, 3, 1), "dtype": "float32", "format": "ND",
                     "ori_shape": (6, 52, 52, 3, 1),"ori_format": "ND"},
                     {"shape": (6, 52, 52, 3, 80), "dtype": "float32", "format": "ND",
                     "ori_shape": (6, 52, 52, 3, 80),"ori_format": "ND"},], [2,2,1,80],4, 4],
         "case_name": "split_d_v_8",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case8)

if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)