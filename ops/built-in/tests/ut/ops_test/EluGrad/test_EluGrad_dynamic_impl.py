#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("EluGrad", "impl.dynamic.elu_grad", "elu_grad")

case1 = {
    "params": [
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32), "ori_format": "ND",
         "range": [(1, 100)]},
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32), "ori_format": "ND",
         "range": [(1, 100)]},
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32), "ori_format": "ND",
         "range": [(1, 100)]}
    ],
    "case_name": "EluGrad_dynamic_1",
    "expect": "success",
    "support_expect": True
}
case2 = {
    "params": [
        {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (15, 32), "ori_format": "ND",
         "range": [(1, 100)]},
        {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (15, 32), "ori_format": "ND",
         "range": [(1, 100)]},
        {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (15, 32), "ori_format": "ND",
         "range": [(1, 100)]}
    ],
    "case_name": "EluGrad_dynamic_1",
    "expect": "success",
    "support_expect": True
}
case3 = {
    "params": [
        {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
        {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
        {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"}
    ],
    "case_name": "EluGrad_dynamic_3",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend910A", "Ascend610", "Ascend310P3", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend610", "Ascend310P3", "Ascend310"], case2)
ut_case.add_case(["Ascend910A", "Ascend610", "Ascend310P3", "Ascend310"], case3)

if __name__ == "__main__":
    ut_case.run(["Ascend910A", "Ascend610", "Ascend310P3", "Ascend310"])
