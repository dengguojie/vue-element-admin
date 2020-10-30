#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from op_test_frame.ut import OpUT

ut_case = OpUT("ShuffleChannel", None, None)


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
    return out_tensor


def gen_shuffle_channel_precision_case(shape, dtype, expect="success", group=1, kernel_name="shuffle_channel", impl_mode="high_performance"):
    return {"params": [{"dtype": dtype, "shape": shape, "format": "NCHW","ori_shape": shape, "ori_format": "NCHW", "param_type": "input", "value_range": [-10, 10]},
                       {"dtype": dtype, "shape": shape, "format": "NCHW", "ori_shape": shape, "ori_format": "NCHW", "param_type": "output", "value_range": [-10, 10]},
                       group],
            "case_name": kernel_name,
            "expect": expect,
            "calc_expect_func": calc_expect_func}


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


# ut_case.add_precision_case("all",
#                            gen_shuffle_channel_precision_case((32, 32, 16, 5), "float16", "success", 2, kernel_name="shuffle_channel_1"))
#
# ut_case.add_precision_case("all",
#                            gen_shuffle_channel_precision_case((10, 10, 13, 69), "float16", "success", 2, kernel_name="shuffle_channel_2"))
#
# ut_case.add_precision_case("all",
#                            gen_shuffle_channel_precision_case((1, 3, 2, 7807), "float16", "success", 3, kernel_name="shuffle_channel_3"))


if __name__ == '__main__':
    ut_case.run("Ascend910")
