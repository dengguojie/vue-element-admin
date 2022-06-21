#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("HardSigmoid", "impl.dynamic.hard_sigmoid", "hard_sigmoid")

case1 = {
    "params": [
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),
         "ori_format": "ND", "range": [(1, 100)]},
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),
         "ori_format": "ND", "range": [(1, 100)]},
    ],
    "case_name": "HardSigmoid_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [
        {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
         "ori_format": "ND", "range": [(1, 100)]},
        {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
         "ori_format": "ND", "range": [(1, 100)]},
    ],
    "case_name": "HardSigmoid_2",
    "expect": "success",
    "support_expect": True
}

case3 = {
    "params": [
        {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
        {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
    ],
    "case_name": "HardSigmoid_3",
    "expect": "success",
    "support_expect": True
}

# 'pylint: disable=unused-argument
def test_import_lib(test_arg):
    import sys
    import importlib
    importlib.reload(sys.modules.get("impl.dynamic.binary_query_register"))
    importlib.reload(sys.modules.get("impl.util.util_attr_common"))

ut_case.add_case(["Ascend910A", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend310"], case2)
ut_case.add_case(["Ascend910A", "Ascend310"], case3)
ut_case.add_cust_test_func(test_func=test_import_lib)

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend310"])
