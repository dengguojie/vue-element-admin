#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import te
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


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)


if __name__ == '__main__':
    with te.op.dynamic():
        ut_case.run("Ascend910")
    exit(0)
