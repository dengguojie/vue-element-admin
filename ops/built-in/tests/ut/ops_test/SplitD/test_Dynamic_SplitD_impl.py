#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT


ut_case = OpUT("SplitD", "impl.dynamic.split_d", "split_d")


case1 = {"params": [{"shape": (-1, -1, -1), "dtype": "int32", "format": "ND",
                     "ori_shape": (2, 32, 128), "ori_format": "ND", "range": [[1, 10000]] * 3},
                    [{"shape": (-1, -1, -1), "dtype": "float32", "format": "ND",
                     "ori_shape": (2, 16, 128), "ori_format": "ND", "range": [[1, 10000]] * 3}],
                    1, 2],
         "case_name": "split_d_1",
         "expect": RuntimeError,
         "support_expect": True}

case2 = {"params": [{"shape": (-1, -1, -1), "dtype": "float16", "format": "ND",
                     "ori_shape": (2, 32, 128), "ori_format": "ND", "range": [[1, 10000]] * 3},
                    [{"shape": (-1, -1, -1), "dtype": "float16", "format": "ND",
                     "ori_shape": (2, 32, 128), "ori_format": "ND", "range": [[1, 10000]] * 3}],
                    1, 2],
         "case_name": "split_d_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (-1, -1), "dtype": "float16", "format": "ND",
                     "ori_shape": (18720, 3), "ori_format": "ND", "range": [[1, 100000]] * 2},
                    [{"shape": (-1, -1), "dtype": "float16", "format": "ND",
                     "ori_shape": (18720, 1), "ori_format": "ND", "range": [[1, 10000]] * 2}] * 3,
                    -1, 3],
         "case_name": "split_d_3",
         "expect": "success",
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
