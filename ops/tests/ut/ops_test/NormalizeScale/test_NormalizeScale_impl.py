#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = OpUT("NormalizeScale", None, None)

def normalize_scale_cce(shape_x1, dtype1, shape_x2, dtype2, shape_x3, dtype3, across_spatial=True, channel_shared=True, eps=1e-10, data_format="NCHW", case_name="normalize_scale"):

    return {"params": [{"shape": shape_x1, "dtype": dtype1, "format": data_format,"ori_shape":shape_x1, "ori_format":data_format},
                       {"shape": shape_x2, "dtype": dtype2, "format": "ND","ori_shape":shape_x2, "ori_format":"ND"},
                       {"shape": shape_x3, "dtype": dtype3, "format": data_format,"ori_shape":shape_x3, "ori_format":data_format},
                       {"shape": shape_x1, "dtype": dtype1, "format": data_format,"ori_shape":shape_x1, "ori_format":data_format},
                       across_spatial, channel_shared, eps],
            "case_name": case_name,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}


case1 = normalize_scale_cce((2, 3, 2, 3), "float16", (1,), "float16", (2, 1, 1, 1), "float32", True, True, 1e-10, "NCHW",
                            "normalize_scale_1")
case2 = normalize_scale_cce((2, 2, 3, 3), "float16", (1,), "float16", (2, 1, 1, 1), "float32", True, True, 1e-10, "NHWC",
                            "normalize_scale_2")
case3 = normalize_scale_cce((2, 3, 2, 3), "float16", (3,), "float16", (2, 1, 1, 1), "float32", True, False, 1e-10, "NCHW",
                            "normalize_scale_3")
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

def calc_expect_func(x1, x2, x3, y, across_spatial=True,
                     channel_shared=True, eps=1e-10):
    x1_value = x1.get("value")
    x2_value = x2.get("value")
    x3_value = x3.get("value")

    x1_sqr_sum = x3_value+eps
    x1_scaled = x1_value*x2_value
    x1_sqr_sum_sqrt = x1_sqr_sum**0.5
    res = x1_scaled/x1_sqr_sum_sqrt
    return res


ut_case.add_precision_case("all", {"params": [{"shape": (2, 3, 2, 3), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 3, 2, 3),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (2, 1, 1, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 1, 1, 1),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (2, 3, 2, 3), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 3, 2, 3),"ori_format": "NCHW", "param_type": "output"},
                                              True,True,1e-10],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (2, 3, 2, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (2, 3, 2, 3),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (3,),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (2, 1, 1, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 1, 1, 1),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (2, 3, 2, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (2, 3, 2, 3),"ori_format": "NCHW", "param_type": "output"},
                                              True,True,1e-10],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (2, 2, 3, 3), "dtype": "int8", "format": "NCHW", "ori_shape": (2, 2, 3, 3),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (1,), "dtype": "int8", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (2, 1, 1, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 1, 1, 1),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (2, 2, 3, 3), "dtype": "int8", "format": "NCHW", "ori_shape": (2, 2, 3, 3),"ori_format": "NCHW", "param_type": "output"},
                                              True,True,1e-10],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })


# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
