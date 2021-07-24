#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT


ut_case = OpUT("SplitV", "impl.dynamic.split_v", "split_v")


case1 = {"params": [{"shape": (-1, -1, -1), "dtype": "int32", "format": "ND",
                     "ori_shape": (2, 32, 128), "ori_format": "ND", "range": [[1, 10000]] * 3},
                    {"shape": (1,), "dtype": "int32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND", "range": [[1, 1]]},
                    {"shape": (1,), "dtype": "int32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND", "range": [[1, 1]]},
                    [{"shape": (-1, -1, -1), "dtype": "float32", "format": "ND",
                     "ori_shape": (2, 32, 128), "ori_format": "ND", "range": [[1, 10000]] * 3}],
                    1],
         "case_name": "split_v_1",
         "expect": RuntimeError,
         "support_expect": True}

case2 = {"params": [{"shape": (-1, -1, -1), "dtype": "float16", "format": "ND",
                     "ori_shape": (2, 32, 128), "ori_format": "ND", "range": [[1, 10000]] * 3},
                    {"shape": (1,), "dtype": "int64", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND", "range": [[1, 1]]},
                    {"shape": (1,), "dtype": "int32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND", "range": [[1, 1]]},
                    [{"shape": (-1, -1, -1), "dtype": "float16", "format": "ND",
                     "ori_shape": (2, 32, 128), "ori_format": "ND", "range": [[1, 10000]] * 3}],
                    1],
         "case_name": "split_v_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (-1, -1), "dtype": "float16", "format": "ND",
                     "ori_shape": (18720, 3), "ori_format": "ND", "range": [[1, 100000]] * 2},
                    {"shape": (1,), "dtype": "int64", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND", "range": [[1, 1]]},
                    {"shape": (1,), "dtype": "int32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND", "range": [[1, 1]]},
                    [{"shape": (-1, -1), "dtype": "float16", "format": "ND",
                     "ori_shape": (18720, 1), "ori_format": "ND", "range": [[1, 10000]] * 2}] * 3,
                    3],
         "case_name": "split_v_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (6, -1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (-1, 96), "ori_format": "ND", "range": [(6, 6), (1, None), (16, 16), (16, 16)]},
                    {"shape": (2,), "dtype": "int32", "format": "ND",
                     "ori_shape": (2,), "ori_format": "ND", "range": [[2, 2]]},
                    {"shape": (1,), "dtype": "int32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND", "range": [[1, 1]]},
                    [{"shape": (2, -1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (-1, 32), "ori_format": "ND", "range": [(2, 2), (2, None), (16, 16), (16, 16)]},
                     {"shape": (4, -1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                      "ori_shape": (-1, 64), "ori_format": "ND", "range": [(4, 4), (2, None), (16, 16), (16, 16)]}
                     ],
                    2],
         "case_name": "split_v_4",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
