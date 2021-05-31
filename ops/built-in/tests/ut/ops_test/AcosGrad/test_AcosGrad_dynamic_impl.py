#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("AcosGrad", "impl.dynamic.acos_grad", "acos_grad")

case1 = {
    "params": [
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32), "ori_format": "ND",
         "range": [(1, 100)]},
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32), "ori_format": "ND",
         "range": [(1, 100)]},
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32), "ori_format": "ND",
         "range": [(1, 100)]}
    ],
    "case_name": "AcosGrad_dynamic_1",
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
    "case_name": "AcosGrad_dynamic_1",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend910A", "Ascend610", "Ascend710", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend610", "Ascend710", "Ascend310"], case2)

if __name__ == "__main__":
    ut_case.run(["Ascend910A", "Ascend610", "Ascend710", "Ascend310"])
