# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT
from impl.multi_merge import op_select_format

ut_case = OpUT("MultiMerge", None, None)

shape_0 = (32, 3277584, 8)
shape_1 = (8, 1808432, 8)

k_num = 1808412
data_type = "float16"
data_format = "NCHW"

case1 = {
    "params": [
        {"shape": shape_0, "dtype": data_type, "format": data_format, "ori_shape": shape_0, "ori_format": data_format},
        {"shape": shape_1, "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format},
        {"shape": (1, ), "dtype": "int32", "format": data_format, "ori_shape": (1, ), "ori_format": data_format},
        k_num,
        False
    ],
    "case_name": "multi_merge_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}
ut_case.add_case(["Ascend910A"], case1)
k_num = 1808412
shape_0 = (2, 1808432, 8)
shape_1 = (k_num, )


data_type = "float16"
data_format = "NCHW"

case2 = {
    "params": [
        {"shape": shape_0, "dtype": data_type, "format": data_format, "ori_shape": shape_0, "ori_format": data_format},
        {"shape": shape_1, "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format},
        {"shape": shape_1, "dtype": "int32", "format": data_format, "ori_shape": shape_1, "ori_format": data_format},
        k_num,
        True,
    ],
    "case_name": "multi_merge_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

ut_case.add_case(["Ascend910A"], case2)


def test_op_select_format(_):
    params = case2.get("params")
    op_select_format(*params)


ut_case.add_cust_test_func("Ascend910A", test_op_select_format)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
