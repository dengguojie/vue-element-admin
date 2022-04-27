#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
select v2 dynamic test UT
'''
from op_test_frame.ut import OpUT

ut_case = OpUT("SelectV2", "impl.dynamic.select_v2", "select_v2")

case1 = {
    "params": [
        {
            "shape": (-1, -1),
            "dtype": "int8",
            "format": "ND",
            "ori_shape": (2, 4),
            "ori_format": "ND",
            "range": [(1, 10), (1, 10)]
        }, {
            "shape": (-1, -1),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2, 4),
            "ori_format": "ND",
            "range": [(1, 10), (1, 10)]
        }, {
            "shape": (-1, -1),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2, 4),
            "ori_format": "ND",
            "range": [(1, 10), (1, 10)]
        }, {
            "shape": (-1, -1),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2, 4),
            "ori_format": "ND",
            "range": [(1, 10), (1, 10)]
        }
    ],
    "case_name": "SelectV2_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [
        {
            "shape": (-1, -1),
            "dtype": "int8",
            "format": "ND",
            "ori_shape": (2, 4),
            "ori_format": "ND",
            "range": [(1, 10), (1, 10)]
        }, {
            "shape": (-1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4),
            "ori_format": "ND",
            "range": [(1, 10), (1, 10)]
        }, {
            "shape": (-1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4),
            "ori_format": "ND",
            "range": [(1, 10), (1, 10)]
        }, {
            "shape": (-1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4),
            "ori_format": "ND",
            "range": [(1, 10), (1, 10)]
        }
    ],
    "case_name": "SelectV2_1",
    "expect": "success",
    "support_expect": True
}


def test_import_lib(test_arg):
    import sys
    import importlib
    importlib.reload(sys.modules.get("impl.dynamic.binary_query_register"))


ut_case.add_case("Ascend910A", case1)
ut_case.add_case("Ascend910A", case2)
ut_case.add_case("Ascend310", case1)
ut_case.add_case("Ascend310", case2)
ut_case.add_cust_test_func(test_func=test_import_lib)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    ut_case.run("Ascend310")
