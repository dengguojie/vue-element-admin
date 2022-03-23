# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("SingleMerge", None, None)

k_num = 1808412
shape_0 = (2, 1808432, 8)
shape_1 = (k_num, )


data_type = "float16"
data_format = "NCHW"

case1 = {
    "params": [
        {"shape": shape_0, "dtype": data_type, "format": data_format, "ori_shape": shape_0, "ori_format": data_format},
        {"shape": shape_1, "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format},
        {"shape": shape_1, "dtype": "int32", "format": data_format, "ori_shape": shape_1, "ori_format": data_format},
        k_num,
    ],
    "case_name": "single_merge_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
