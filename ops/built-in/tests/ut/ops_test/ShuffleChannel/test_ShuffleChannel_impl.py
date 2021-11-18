#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import os

ut_case = OpUT("ShuffleChannel", None, None)

case1 = {"params": [{"shape": (1, 4, 3, 4), "dtype": "int32", "format": "NCHW", 'ori_shape': (1, 4, 3, 4), 'ori_format': "NCHW"},
                    {"shape": (1, 4, 3, 4), "dtype": "int32", "format": "NCHW", 'ori_shape': (1, 4, 3, 4), 'ori_format': "NCHW"},
                    3],
         "case_name": "ShuffleChannel_1",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1, 4, 3, 4, 5), "dtype": "int32", "format": "NCHW", 'ori_shape': (1, 4, 3, 4, 5), 'ori_format': "NCHW"},
                    {"shape": (1, 4, 3, 4, 5), "dtype": "int32", "format": "NCHW", 'ori_shape': (1, 4, 3, 4, 5), 'ori_format': "NCHW"},
                    2],
         "case_name": "ShuffleChannel_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1, 4, 3, 4), "dtype": "float64", "format": "NCHW", 'ori_shape': (1, 4, 3, 4), 'ori_format': "NCHW"},
                    {"shape": (1, 4, 3, 4), "dtype": "float64", "format": "NCHW", 'ori_shape': (1, 4, 3, 4), 'ori_format': "NCHW"},
                    2],
         "case_name": "ShuffleChannel_3",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (1, 4, 3, 4), "dtype": "float32", "format": "NCHW", 'ori_shape': (1, 4, 3, 4), 'ori_format': "NCHW"},
                    {"shape": (1, 5, 3, 4), "dtype": "float32", "format": "NCHW", 'ori_shape': (1, 5, 3, 4), 'ori_format': "NCHW"},
                    3],
         "case_name": "ShuffleChannel_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 4, 3), "dtype": "int8", "format": "NCHW", 'ori_shape': (1, 4, 3), 'ori_format': "NCHW"},
                    {"shape": (1, 4, 3), "dtype": "int8", "format": "NCHW", 'ori_shape': (1, 4, 3), 'ori_format': "NCHW"},
                    2],
         "case_name": "ShuffleChannel_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (1, 4), "dtype": "int8", "format": "NCHW", 'ori_shape': (1, 4), 'ori_format': "NCHW"},
                    {"shape": (1, 4), "dtype": "int8", "format": "NCHW", 'ori_shape': (1, 4), 'ori_format': "NCHW"},
                    2],
         "case_name": "ShuffleChannel_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (1, 4, 3, 4), "dtype": "int8", "format": "NCHW", 'ori_shape': (1, 4, 3, 4), 'ori_format': "NCHW"},
                    {"shape": (1, 4, 3, 4), "dtype": "int8", "format": "NCHW", 'ori_shape': (1, 4, 3, 4), 'ori_format': "NCHW"},
                    2],
         "case_name": "ShuffleChannel_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case8 = {"params": [{"shape": (99, 3, 1, 4), "dtype": "int8", "format": "NCHW", 'ori_shape': (99, 3, 1, 4), 'ori_format': "NCHW"},
                    {"shape": (99, 3, 1, 4), "dtype": "int8", "format": "NCHW", 'ori_shape': (99, 3, 1, 4), 'ori_format': "NCHW"},
                    3],
         "case_name": "ShuffleChannel_8",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case9 = {"params": [{"shape": (9973, 2, 17, 3), "dtype": "int8", "format": "NCHW", 'ori_shape': (9973, 2, 17, 3), 'ori_format': "NCHW"},
                    {"shape": (9973, 2, 17, 3), "dtype": "int8", "format": "NCHW", 'ori_shape': (9973, 2, 17, 3), 'ori_format': "NCHW"},
                    1],
         "case_name": "ShuffleChannel_9",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case("all", case1)
ut_case.add_case("all", case2)
ut_case.add_case("all", case3)
ut_case.add_case("all", case4)
ut_case.add_case("all", case5)
ut_case.add_case("all", case6)
ut_case.add_case("all", case7)
ut_case.add_case("all", case8)
ut_case.add_case("all", case9)


def calc_expect_func(input_x, input_y, group, kernel_name="shuffle_channel", impl_mode="high_performance"):
    input_tensor = input_x.get("value")
    shape = input_tensor.shape
    batch = shape[0]
    channel = shape[1]
    x_len = 1
    data_len = shape[2] * shape[3]
    for i in shape:
        x_len = x_len * i
    input_tensor = input_tensor.reshape(x_len, )
    out_tensor = np.array([0.0] * x_len)
    for c in range(batch * channel):
        j = (c % channel) // group
        i = (c % channel) % group
        index = (c // channel) * channel + i * (channel // group) + j
        start = c * data_len
        out_tensor[start:start + data_len] = input_tensor[index * data_len:index * data_len + data_len]
    return out_tensor.reshape(input_y["shape"])

ut_case.add_precision_case("all", {"params": [{"shape": (1, 3, 2, 7807), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3, 2, 7807),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (1, 3, 2, 7807), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3, 2, 7807),"ori_format": "NCHW", "param_type": "output"},
                                              3],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (10, 10, 13, 69), "dtype": "int32", "format": "NCHW", "ori_shape": (10, 10, 13, 69),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (10, 10, 13, 69), "dtype": "int32", "format": "NCHW", "ori_shape": (10, 10, 13, 69),"ori_format": "NCHW", "param_type": "output"},
                                              2],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (32, 32, 16, 5), "dtype": "float32", "format": "NCHW", "ori_shape": (32, 32, 16, 5),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (32, 32, 16, 5), "dtype": "float32", "format": "NCHW", "ori_shape": (32, 32, 16, 5),"ori_format": "NCHW", "param_type": "output"},
                                              2],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })


# ============ auto gen ["Ascend910"] test cases end =================

def test_shuffle_channel_001(test_arg):
    from te.platform.cce_conf import te_set_version
    from impl.shuffle_channel import shuffle_channel
    te_set_version("SD3403")
    shuffle_channel(
        {
            "shape": (1, 4, 3),
            "dtype": "int16",
            "format": "NCHW",
            'ori_shape': (1, 4, 3),
            'ori_format': "NCHW"
        }, {
            "shape": (1, 4, 3),
            "dtype": "int16",
            "format": "NCHW",
            'ori_shape': (1, 4, 3),
            'ori_format': "NCHW"
        }, 2)


def test_get_op_support_info_001(test_arg):
    from impl.shuffle_channel import get_op_support_info
    get_op_support_info(
        {
            "shape": (1, 4, 3),
            "dtype": "int16",
            "format": "NCHW",
            'ori_shape': (1, 4, 3),
            'ori_format': "NCHW"
        }, None)
    get_op_support_info(
        {
            "shape": (1, 4, 3),
            "dtype": "int16",
            "format": "NHWC",
            'ori_shape': (1, 4, 3),
            'ori_format': "NCHW"
        }, None)


ut_case.add_cust_test_func(test_func=test_shuffle_channel_001)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_001)

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
