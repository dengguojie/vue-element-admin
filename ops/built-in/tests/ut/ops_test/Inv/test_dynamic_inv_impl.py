#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("Inv", "impl.dynamic.inv", "inv")

case1 = {
    "params": [
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND","range":[(1, 100)]},
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND","range":[(1, 100)]},
    ],
    "case_name": "Inv_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [
        {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (2, 3),"ori_format": "ND","range":[(1, 100), (1, 100)]},
        {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (2, 3),"ori_format": "ND","range":[(1, 100), (1, 100)]},
    ],
    "case_name": "Inv_2",
    "expect": "success",
    "support_expect": True
}
case3 = {
    "params": [
        {"shape": (-1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (2, 3, 4),"ori_format": "ND","range":[(1, 100), (1, 100), (1, 100)]},
        {"shape": (-1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (2, 3, 4),"ori_format": "ND","range":[(1, 100), (1, 100), (1, 100)]},
    ],
    "case_name": "Inv_3",
    "expect": "success",
    "support_expect": True
}
case4 = {
    "params": [
        {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND","range":[(1, 100)]},
        {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND","range":[(1, 100)]},
    ],
    "case_name": "Inv_4",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case("Ascend910A", case1)
ut_case.add_case("Ascend910A", case2)
ut_case.add_case("Ascend910A", case3)
ut_case.add_case("Ascend910A", case4)


def test_import_libs(test_arg):
    import sys
    import importlib
    importlib.reload(sys.modules.get("impl.dynamic.binary_query_register"))

ut_case.add_cust_test_func(test_func=test_import_libs)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
