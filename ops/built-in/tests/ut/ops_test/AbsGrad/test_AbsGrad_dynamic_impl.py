#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("AbsGrad", "impl.dynamic.abs_grad", "abs_grad")

case1 = {
    "params": [
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (2, 4),"ori_format": "ND","range":[(1, 100)]},
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (2, 4),"ori_format": "ND","range":[(1, 100)]},
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (2, 4),"ori_format": "ND","range":[(1, 100)]},
    ],
    "case_name": "AbsGrad_dynamic_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [
        {"shape": (-1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (2, 4, 4),"ori_format": "ND","range":[(1, 100)] * 3},
        {"shape": (-1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (2, 4, 4),"ori_format": "ND","range":[(1, 100)] * 3},
        {"shape": (-1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (2, 4, 4),"ori_format": "ND","range":[(1, 100)] * 3},
    ],
    "case_name": "AbsGrad_dynamic_2",
    "expect": "success",
    "support_expect": True
}

case3 = {
    "params": [
        {"shape": (2, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (2, 4, 4),"ori_format": "ND","range":[(1, 100)] * 3},
        {"shape": (2, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (2, 4, 4),"ori_format": "ND","range":[(1, 100)] * 3},
        {"shape": (2, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (2, 4, 4),"ori_format": "ND","range":[(1, 100)] * 3},
    ],
    "case_name": "AbsGrad_dynamic_3",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend910A", "Ascend610", "Ascend310P3"], case1)
ut_case.add_case(["Ascend910A", "Ascend610", "Ascend310P3"], case2)
ut_case.add_case(["Ascend910A", "Ascend610", "Ascend310P3"], case3)

if __name__ == "__main__":
    ut_case.run(["Ascend910A", "Ascend610", "Ascend310P3"])
