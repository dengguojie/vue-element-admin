#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = OpUT("PadD", None, None)

case1 = {"params": [{"shape": (32, 128, 1024), "dtype": "float16", "format": "NCHW", "ori_shape": (32, 128, 1024),"ori_format": "NCHW"},
                    {"shape": (32, 128, 1024), "dtype": "float16", "format": "NCHW", "ori_shape": (32, 128, 1024),"ori_format": "NCHW"},
                    [[0, 0],[0, 384],[0, 0]]],
         "case_name": "pad_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2, 2, 1024*240), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1024*240),"ori_format": "NCHW"},
                    {"shape": (2, 2, 1024*240), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1024*240),"ori_format": "NCHW"},
                    [[0, 0],[7, 7],[0, 7]]],
         "case_name": "pad_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (2,), "dtype": "float32", "format": "NCHW", "ori_shape": (2,),"ori_format": "NCHW"},
                    {"shape": (2,), "dtype": "float32", "format": "NCHW", "ori_shape": (2,),"ori_format": "NCHW"},
                    [[0,3]]],
         "case_name": "pad_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2,2,9), "dtype": "float32", "format": "NCHW", "ori_shape": (2,2,9),"ori_format": "NCHW"},
                    {"shape": (2,2,9), "dtype": "float32", "format": "NCHW", "ori_shape": (2,2,9),"ori_format": "NCHW"},
                    [[0, 0],[9, 7],[0, 0]]],
         "case_name": "pad_d_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (2, 2, 1027), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    {"shape": (2, 2, 1027), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    [[0, 0],[0, 7],[0, 7]]],
         "case_name": "pad_d_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (2, 2, 1027), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    {"shape": (2, 2, 1027), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    [[0, 0],[0, 16],[0, 0]]],
         "case_name": "pad_d_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (1, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (1, 1),"ori_format": "NCHW"},
                    {"shape": (1, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (1, 1),"ori_format": "NCHW"},
                    [[0, 0],[0, 0]]],
         "case_name": "pad_d_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case8 = {"params": [{"shape": (1, 1, 304, 64, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 304, 64, 8),"ori_format": "NHWC"},
                    {"shape": (1, 5, 304, 64, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 304, 64, 72),"ori_format": "NHWC"},
                    [[0, 0],[0, 0],[0, 0],[32, 32]]],
         "case_name": "pad_d_8",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case8)


def test_op_select_format(test_arg):
    """
    test_op_select_format
    """
    from impl.pad_d import op_select_format
    op_select_format({"shape": (1, 304, 64, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 304, 64, 8),"ori_format": "NHWC"},
                     {"shape": (1, 304, 64, 72), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 304, 64, 72),"ori_format": "NHWC"},
                     [[0, 0],[0, 0],[0, 0],[32, 32]],
                     "test_pad_op_select_format_case_1")

ut_case.add_cust_test_func(test_func=test_op_select_format)


def calc_expect_func(x, res, paddings):
    x_shape = x.get("shape")
    x_value = x.get("value")

    dtype = x["dtype"]
    if dtype == "fp16" or dtype == "float16":
        s_type = np.float16
    elif dtype == "fp32" or dtype == "float32":
        s_type = np.float32
    elif dtype == "int32":
        s_type = np.int32
    elif dtype == "int8":
        s_type = np.int8
    elif dtype == "uint8":
        s_type = np.uint8
    else:
        raise RuntimeError("unsupported dtype:%s " % dtype)

    res = np.pad(x_value, paddings).astype(s_type)
    return res

ut_case.add_precision_case("all", {"params": [{"shape": (2,), "dtype": "float32", "format": "NCHW", "ori_shape": (2,),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (5,), "dtype": "float32", "format": "NCHW", "ori_shape": (5,),"ori_format": "NCHW", "param_type": "output"},
                                              [[0,3]]],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (2, 2, 1027), "dtype": "float16", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (2, 18, 1043), "dtype": "float16", "format": "NCHW", "ori_shape": (2, 18, 1043),"ori_format": "NCHW", "param_type": "output"},
                                              [[0, 0],[0, 16],[0, 16]]],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (2, 2, 1024*240), "dtype": "int32", "format": "NCHW", "ori_shape": (2, 2, 1024*240),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (2, 16, 245767), "dtype": "int32", "format": "NCHW", "ori_shape": (2, 16, 245767),"ori_format": "NCHW", "param_type": "output"},
                                              [[0, 0],[7, 7],[0, 7]]],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW", "param_type": "output"},
                                              [[0, 0],]],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (1, 304, 64, 8), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 304, 64, 8),"ori_format": "NHWC", "param_type": "input"},
                                              {"shape": (1, 304, 64, 72), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 304, 64, 72),"ori_format": "NHWC", "param_type": "output"},
                                              [[0, 0],[0, 0],[0, 0],[32, 32]]],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (1, 304, 64, 8), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 304, 64, 8),"ori_format": "NHWC", "param_type": "input"},
                                              {"shape": (1, 304, 64, 72), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 304, 64, 72),"ori_format": "NHWC", "param_type": "output"},
                                              [[0, 0],[0, 0],[0, 0],[32, 32]]],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("all", {"params": [{"shape": (120, 10), "dtype": "float16", "format": "ND", "ori_shape": (120, 10), "ori_format": "ND", "param_type": "input"},
                                              {"shape": (1592, 1482), "dtype": "float16", "format": "ND", "ori_shape": (1592, 1482), "ori_format": "ND", "param_type": "output"},
                                              [[689, 783], [689, 783]]],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)

