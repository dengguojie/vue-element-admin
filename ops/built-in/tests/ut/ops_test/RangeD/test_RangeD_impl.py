#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("RangeD", None, None)

case1 = {"params": [{"shape": (5,), "dtype": "int32", "format": "ND", "ori_shape": (5,),"ori_format": "ND"},
                    {"shape": (5,), "dtype": "int32", "format": "ND", "ori_shape": (5,),"ori_format": "ND"},
                    1.0, 6.0, 1.0,
                    ],
         "case_name": "range_d_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (8192,), "dtype": "float32", "format": "ND", "ori_shape": (8192,),"ori_format": "ND"},
                    {"shape": (8192,), "dtype": "float32", "format": "ND", "ori_shape": (8192,),"ori_format": "ND"},
                    2.2, 8194.2, 1.0,
                    ],
         "case_name": "range_d_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (6,), "dtype": "int32", "format": "ND", "ori_shape": (6,),"ori_format": "ND"},
                    {"shape": (6,), "dtype": "int32", "format": "ND", "ori_shape": (6,),"ori_format": "ND"},
                    5.0, -1.0, -1.0
                    ],
         "case_name": "range_d_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (2, 2), "dtype": "int32", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND"},
                    {"shape": (2, 2), "dtype": "int32", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND"},
                    10.0, 20.0, 2.0,
                    ],
         "case_name": "range_d_4",
         "expect": RuntimeError,
         "support_expect": True}

case5 = {"params": [{"shape": (2,), "dtype": "float32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    {"shape": (2,), "dtype": "float32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    10.0, 10.0, 1.0,
                    ],
         "case_name": "range_d_5",
         "expect": RuntimeError,
         "support_expect": True}

case6 = {"params": [{"shape": (2,), "dtype": "float32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    {"shape": (2,), "dtype": "float32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    10.0, 20.0, 0.0,
                    ],
         "case_name": "range_d_6",
         "expect": RuntimeError,
         "support_expect": True}

case7 = {"params": [{"shape": (2,), "dtype": "float32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    {"shape": (2,), "dtype": "float32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    20.0, 10.0, 1.0,
                    ],
         "case_name": "range_d_7",
         "expect": RuntimeError,
         "support_expect": True}

case8 = {"params": [{"shape": (2,), "dtype": "float32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    {"shape": (2,), "dtype": "float32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    10.0, 20.0, -1.0,
                    ],
         "case_name": "range_d_8",
         "expect": RuntimeError,
         "support_expect": True}

ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case6)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case7)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case8)

def calc_expect_func(x, y, start, limit, delta):
    x_value = x.get("value")
    x_dtype = x.get("dtype")
    if x_dtype == "int32":
        start = int(start)
        delta = int(delta)
    y = x_value*delta+start
    return y

ut_case.add_precision_case("all", {"params": [{"shape": (6,), "dtype": "float32", "format": "ND", "ori_shape": (6,),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (6,), "dtype": "float32", "format": "ND", "ori_shape": (6,),"ori_format": "ND", "param_type": "output"},
                                              5.0, -1.0, -1.0
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (6,), "dtype": "int32", "format": "ND", "ori_shape": (6,),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (6,), "dtype": "int32", "format": "ND", "ori_shape": (6,),"ori_format": "ND", "param_type": "output"},
                                              5.0, -1.0, -1.0
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })