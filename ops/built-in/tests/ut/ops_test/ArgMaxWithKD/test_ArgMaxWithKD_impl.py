"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

ArgMaxWithKd ut case
"""
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import os
ut_case = OpUT("ArgMaxWithKD", "impl.arg_max_with_kd", "arg_max_with_kd")

case1 = {"params": [{"shape": (5, 8,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"}, #x
                    {"shape": (5, 8,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"}, #h
                    {"shape": (5, 8,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"}, #h
                    10000, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (3000, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (3000, 1),"ori_format": "NCHW"}, #x
                    {"shape": (3000, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (3000, 1),"ori_format": "NCHW"}, #h
                    {"shape": (3000, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (3000, 1),"ori_format": "NCHW"}, #h
                    10000, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (2,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (2,16,16),"ori_format": "NCHW"}, #x
                    {"shape": (2,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (2,16,16),"ori_format": "NCHW"}, #h
                    {"shape": (2,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (2,16,16),"ori_format": "NCHW"}, #h
                    10000, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (2,10,1028,1,16), "dtype": "float16", "format": "NCHW", "ori_shape": (2,10,1028,1,16),"ori_format": "NCHW"}, #x
                    {"shape": (2,10,1028,1,16), "dtype": "float16", "format": "NCHW", "ori_shape": (2,10,1028,1,16),"ori_format": "NCHW"},
                    {"shape": (2,10,1028,1,16), "dtype": "float16", "format": "NCHW", "ori_shape": (2,10,1028,1,16),"ori_format": "NCHW"}, #h
                    10000, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (2,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (2,16,16),"ori_format": "NCHW"}, #x
                    {"shape": (2,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (2,16,16),"ori_format": "NCHW"},
                    {"shape": (2,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (2,16,16),"ori_format": "NCHW"}, #h
                    10000, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_5",
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape": (2, 3, 2, 16, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 2,16,16),"ori_format": "NCHW"}, #x
                    {"shape": (2, 3, 2, 16, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 2,16,16),"ori_format": "NCHW"}, #h
                    {"shape": (2, 3, 2, 16, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 2,16,16),"ori_format": "NCHW"}, #h
                    0, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_6",
         "expect": "success",
         "support_expect": True}

case7 = {"params": [{"shape": (10240, 2, 8, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (10240, 2, 8, 1),"ori_format": "NCHW"}, #x
                    {"shape": (10240, 2, 8, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (10240, 2, 8, 1),"ori_format": "NCHW"}, #h
                    {"shape": (10240, 2, 8, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (10240, 2, 8, 1),"ori_format": "NCHW"}, #h
                    1, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_7",
         "expect": "success",
         "support_expect": True}

case8 = {"params": [{"shape": (16, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"}, #x
                    {"shape": (16, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"}, #h
                    {"shape": (16, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"}, #h
                    1, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_8",
         "expect": "success",
         "support_expect": True}

case9 = {"params": [{"shape": (16, 64, 256, 256), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 64, 256, 256),"ori_format": "NCHW"}, #x
                    {"shape": (16, 64, 256, 256), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 64, 256, 256),"ori_format": "NCHW"}, #h
                    {"shape": (16, 64, 256, 256), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 64, 256, 256),"ori_format": "NCHW"}, #h
                    1, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_9",
         "expect": "success",
         "support_expect": True}

case10 = {"params": [{"shape": (16, 256, 256, 256), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 256, 256, 256),"ori_format": "NCHW"}, #x
                    {"shape": (16, 256, 256, 256), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 256, 256, 256),"ori_format": "NCHW"}, #h
                    {"shape": (16, 256, 256, 256), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 256, 256, 256),"ori_format": "NCHW"}, #h
                    1, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_10",
         "expect": "success",
         "support_expect": True}

case11 = {"params": [{"shape": (16, 256, 30, 2), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 256, 30, 2),"ori_format": "NCHW"}, #x
                    {"shape": (16, 256, 30, 2), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 256, 30, 2),"ori_format": "NCHW"}, #h
                    {"shape": (16, 256, 30, 2), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 256, 30, 2),"ori_format": "NCHW"}, #h
                    1, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_11",
         "expect": "success",
         "support_expect": True}

case12 = {"params": [{"shape": (16, 256, 30, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 256, 30, 2),"ori_format": "NCHW"}, #x
                    {"shape": (16, 256, 30, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 256, 30, 2),"ori_format": "NCHW"}, #h
                    {"shape": (16, 256, 30, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 256, 30, 2),"ori_format": "NCHW"}, #h
                    1, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_12",
         "expect": RuntimeError,
         "support_expect": True}

case13 = {"params": [{"shape": (16, 1, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 1, 16, 16),"ori_format": "NCHW"}, #x
                    {"shape": (16, 1, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 1, 16, 16),"ori_format": "NCHW"}, #h
                    {"shape": (16, 1, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 1, 16, 16),"ori_format": "NCHW"}, #h
                    1, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_13",
         "expect": "success",
         "support_expect": True}

case14 = {"params": [{"shape": (10240, 2, 8, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (10240, 2, 8, 1),"ori_format": "NCHW"}, #x
                    {"shape": (10240, 2, 8, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (10240, 2, 8, 1),"ori_format": "NCHW"}, #h
                    {"shape": (10240, 2, 8, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (10240, 2, 8, 1),"ori_format": "NCHW"}, #h
                    10000, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_14",
         "expect": "success",
         "support_expect": True}

case15 = {"params": [{"shape": (16, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"}, #x
                    {"shape": (16, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"}, #h
                    {"shape": (16, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"}, #h
                    10000, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_15",
         "expect": "success",
         "support_expect": True}

case16 = {"params": [{"shape": (16, 16, 1, 1), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHD"}, #x
                    {"shape": (16, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"}, #h
                    {"shape": (16, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"}, #h
                    10000, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_16",
         "expect": RuntimeError,
         "support_expect": True}

case17 = {"params": [{"shape": (16, 16, 1, 1), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"}, #x
                    {"shape": (16, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"}, #h
                    {"shape": (16, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"}, #h
                    10000, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_17",
         "expect": RuntimeError,
         "support_expect": True}

case18 = {"params": [{"shape": (16, 16, 1, 1, 1), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16, 16, 1, 1, 1),"ori_format": "NCHW"}, #x
                    {"shape": (16, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"}, #h
                    {"shape": (16, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"}, #h
                    10000, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_18",
         "expect": RuntimeError,
         "support_expect": True}

case19 = {"params": [{"shape": (16, 16, 1, 1, 1), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"}, #x
                    {"shape": (16, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"}, #h
                    {"shape": (16, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"}, #h
                    10000, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_19",
         "expect": RuntimeError,
         "support_expect": True}

case20 = {"params": [{"shape": (16, 16, 1, 1, 1), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"}, #x
                    {"shape": (16, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"}, #h
                    {"shape": (16, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"}, #h
                    1, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_20",
         "expect": RuntimeError,
         "support_expect": True}

case21 = {"params": [{"shape": (5, 8,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"}, #x
                    {"shape": (5, 8,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"}, #h
                    {"shape": (5, 8,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"}, #h
                    10000, False, 2,
                    ],
         "case_name": "ArgMaxWithKd_21",
         "expect": RuntimeError,
         "support_expect": True}

case22 = {"params": [{"shape": (5, 8,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"}, #x
                    {"shape": (5, 8,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"}, #h
                    {"shape": (5, 8,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"}, #h
                    2147483649, False, 1,
                    ],
         "case_name": "ArgMaxWithKd_22",
         "expect": RuntimeError,
         "support_expect": True}

case23 = {"params": [{"shape": (5, 8,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"}, #x
                    {"shape": (5, 8,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"}, #h
                    {"shape": (5, 8,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"}, #h
                    3, True, 1,
                    ],
         "case_name": "ArgMaxWithKd_23",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case1)
ut_case.add_case(["Ascend310P3"], case2)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case3)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case4)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case5)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case6)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case7)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case8)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case9)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case10)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case11)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case13)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case14)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case15)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case12)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case16)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case17)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case18)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case19)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case20)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case21)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case22)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310P3"], case23)

#precision cases
def naive_arg_top_k(data, top_k, axis):
    """
    perform topK based on np.argsort
    :param data: to be sorted
    :param K: select and sort the top K items
    :param axis: dimension to be sorted.
    :return:
    """
    # ascending
    full_sort = np.argsort(data, axis=axis).astype("int32")
    # take the top_k
    top_k_list = range(-top_k, 0)
    full_sort = full_sort.take(top_k_list, axis=axis)
    # make it descending
    return np.flip(full_sort, axis=axis)

def calc_expect_func(input_x, out1, out2, axis=10000, out_max_val=False, top_k=1):
    shape_x = input_x['shape']
    x = input_x['value']
    if axis == 10000:
        x = x.reshape(shape_x[0], -1)
        axis = 1
    indices = naive_arg_top_k(x, top_k, axis=axis)
    if out_max_val:
        values = np.take_along_axis(x, indices, axis=axis)

    return indices, values

ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (2,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (2,16,16),"ori_format": "NCHW", "param_type": "input", "value_range":[1,100]}, #x
                                                    {"shape": (2, 1), "dtype": "int32", "format": "NCHW", "ori_shape": (2, 1),"ori_format": "NCHW", "param_type": "output"}, #h
                                                    {"shape": (2, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (2, 1),"ori_format": "NCHW", "param_type": "output"}, #h
                                                    10000, True, 1,
                                                    ],
                                         "expect": "success",
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})

def test_arg_max_with_kd_001(test_arg):
    from te.platform.cce_conf import te_set_version
    from impl.arg_max_with_kd import arg_max_with_kd
    te_set_version("Ascend310P3")
    arg_max_with_kd({"shape": (5, 8,16,16), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"},
                    {"shape": (5, 8,16,16), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"},
                    {"shape": (5, 8,16,16), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"},
                    10000, False, 1)
    arg_max_with_kd({"shape": (2,16,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,16,16),"ori_format": "NCHW"},
                    {"shape": (2,16,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,16,16),"ori_format": "NCHW"},
                    {"shape": (2,16,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,16,16),"ori_format": "NCHW"},
                    10000, False, 1)
    arg_max_with_kd({"shape": (2,10,1028,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,10,1028,1,16),"ori_format": "NCHW"},
                    {"shape": (2,10,1028,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,10,1028,1,16),"ori_format": "NCHW"},
                    {"shape": (2,10,1028,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,10,1028,1,16),"ori_format": "NCHW"},
                    10000, False, 1)
    arg_max_with_kd({"shape": (2, 3, 2, 16, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2, 2,16,16),"ori_format": "NCHW"},
                    {"shape": (2, 3, 2, 16, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2, 2,16,16),"ori_format": "NCHW"},
                    {"shape": (2, 3, 2, 16, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2, 2,16,16),"ori_format": "NCHW"},
                    0, False, 1)
    arg_max_with_kd({"shape": (16, 16, 1, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"},
                    {"shape": (16, 16, 1, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"},
                    {"shape": (16, 16, 1, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"},
                    1, False, 1)
    arg_max_with_kd({"shape": (5, 8, 16, 16), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"},
                    {"shape": (5, 8, 16, 16), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"},
                    {"shape": (5, 8, 16, 16), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"},
                    3, True, 1)
    arg_max_with_kd({"shape": (16, 1, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 1, 16, 16),"ori_format": "NCHW"},
                    {"shape": (16, 1, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 1, 16, 16),"ori_format": "NCHW"},
                    {"shape": (16, 1, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 1, 16, 16),"ori_format": "NCHW"},
                    1, True, 1)
    arg_max_with_kd({"shape": (10240, 2, 8, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (10240, 2, 8, 1),"ori_format": "NCHW"},
                    {"shape": (10240, 2, 8, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (10240, 2, 8, 1),"ori_format": "NCHW"},
                    {"shape": (10240, 2, 8, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (10240, 2, 8, 1),"ori_format": "NCHW"},
                    1, False, 1)
    arg_max_with_kd({"shape": (16, 256, 30, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 256, 30, 2),"ori_format": "NCHW"},
                    {"shape": (16, 256, 30, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 256, 30, 2),"ori_format": "NCHW"},
                    {"shape": (16, 256, 30, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 256, 30, 2),"ori_format": "NCHW"},
                    1, False, 1)
    arg_max_with_kd({"shape": (16, 256, 256, 256), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 256, 256, 256),"ori_format": "NCHW"},
                    {"shape": (16, 256, 256, 256), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 256, 256, 256),"ori_format": "NCHW"},
                    {"shape": (16, 256, 256, 256), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 256, 256, 256),"ori_format": "NCHW"},
                    1, True, 1)
    arg_max_with_kd({"shape": (16, 64, 256, 256), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 64, 256, 256),"ori_format": "NCHW"},
                    {"shape": (16, 64, 256, 256), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 64, 256, 256),"ori_format": "NCHW"},
                    {"shape": (16, 64, 256, 256), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 64, 256, 256),"ori_format": "NCHW"},
                    1, True, 1,)
    arg_max_with_kd({"shape": (16, 16, 1, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"},
                    {"shape": (16, 16, 1, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"},
                    {"shape": (16, 16, 1, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"},
                    1, True, 1)
    arg_max_with_kd({"shape": (16, 16, 1, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"},
                    {"shape": (16, 16, 1, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"},
                    {"shape": (16, 16, 1, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 16, 1, 1),"ori_format": "NCHW"},
                    10000, True, 1)
    arg_max_with_kd({"shape": (2,16,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,16,16),"ori_format": "NCHW"},
                    {"shape": (2,16,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,16,16),"ori_format": "NCHW"},
                    {"shape": (2,16,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,16,16),"ori_format": "NCHW"},
                    10000, True, 1)
    arg_max_with_kd({"shape": (2,10,1028,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,10,1028,1,16),"ori_format": "NCHW"},
                    {"shape": (2,10,1028,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,10,1028,1,16),"ori_format": "NCHW"},
                    {"shape": (2,10,1028,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,10,1028,1,16),"ori_format": "NCHW"},
                    10000, True, 1)
    arg_max_with_kd({"shape": (10240, 2, 8, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (10240, 2, 8, 1),"ori_format": "NCHW"}, #x
                    {"shape": (10240, 2, 8, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (10240, 2, 8, 1),"ori_format": "NCHW"}, #h
                    {"shape": (10240, 2, 8, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (10240, 2, 8, 1),"ori_format": "NCHW"}, #h
                    10000, False, 1)
    arg_max_with_kd({"shape": (5, 8,16,15), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"}, #x
                    {"shape": (5, 8,16,15), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"}, #h
                    {"shape": (5, 8,16,15), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"}, #h
                    3, True, 1,)
    arg_max_with_kd({"shape": (5, 8,16,15), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"}, #x
                    {"shape": (5, 8,16,15), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"}, #h
                    {"shape": (5, 8,16,15), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 8,16,16),"ori_format": "NCHW"}, #h
                    3, True, 1,)
    te_set_version("Ascend310P3")
ut_case.add_cust_test_func(test_func=test_arg_max_with_kd_001)

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
