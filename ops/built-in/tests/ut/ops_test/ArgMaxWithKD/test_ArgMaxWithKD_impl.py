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
from tbe.common.platform import set_current_compile_soc_info
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
         "expect": "success",
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


# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend710"], case2)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case5)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case6)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case7)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case8)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case9)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case10)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case11)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case13)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case14)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case15)
set_current_compile_soc_info('Ascend710')
ut_case.add_case(["Ascend710"], case12)


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
if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
