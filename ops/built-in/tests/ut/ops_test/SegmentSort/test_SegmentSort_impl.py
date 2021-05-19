# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("SegmentSort", None, None)


shape_0 = (246016, )
shape_1 = (2048, )
shape_2 = (32, 7952, 8)
data_type = "float16"
data_format = "NCHW"
k_num = 7936
case1 = {
    "params": [
        {"shape": shape_0, "dtype": data_type, "format": data_format, "ori_shape": shape_0, "ori_format": data_format},
        {"shape": shape_1, "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format},
        {"shape": shape_2, "dtype": data_type, "format": data_format, "ori_shape": shape_2, "ori_format": data_format},
        k_num,
    ],
    "case_name": "segment_sort_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

shape_0 = (104857600, )
shape_2 = (32, 3277584, 8)
k_num = 1808412
case2 = {
    "params": [
        {"shape": shape_0, "dtype": data_type, "format": data_format, "ori_shape": shape_0, "ori_format": data_format},
        {"shape": shape_1, "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format},
        {"shape": shape_2, "dtype": data_type, "format": data_format, "ori_shape": shape_2, "ori_format": data_format},
        k_num,
    ],
    "case_name": "segment_sort_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}


ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
