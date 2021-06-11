# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT
import numpy as np

ut_case = OpUT("NormalizeBatch", None, None)


def get_params(n_num, c_num, d_num, normalize_type, epsilon):
    data_format = "ND"
    data_type = "float32"
    int_type = "int32"
    shape_0 = (n_num, c_num, d_num)
    shape_1 = (n_num,)
    input_x = {"shape": shape_0, "format": data_format, "dtype": data_type,
               "ori_shape": shape_0, "ori_format": data_format}
    seq_len = {"shape": shape_1, "format": data_format, "dtype": int_type,
               "ori_shape": shape_1, "ori_format": data_format}
    output_y = {"shape": shape_0, "format": data_format, "dtype": data_type,
                "ori_shape": shape_0, "ori_format": data_format}
    params = [input_x, seq_len, output_y, normalize_type, epsilon]
    return params


case1 = {
    "params": get_params(8, 2, 1024, "per_feature", 0.00001),
    "case_name": "normalize_batch_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case2 = {
    "params": get_params(3, 2, 60000, "per_feature", 0.00001),
    "case_name": "normalize_batch_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case3 = {
    "params": get_params(8, 2, 1024, "all_features", 0.00001),
    "case_name": "normalize_batch_3",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case4 = {
    "params": get_params(3, 2, 60000, "all_features", 0.00001),
    "case_name": "normalize_batch_4",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case5 = {
    "params": get_params(8, 2, 1025, "per_feature", 0.00001),
    "case_name": "normalize_batch_5",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case6 = {
    "params": get_params(3, 2, 60001, "per_feature", 0.00001),
    "case_name": "normalize_batch_6",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case7 = {
    "params": get_params(8, 2, 1025, "all_features", 0.00001),
    "case_name": "normalize_batch_7",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case8 = {
    "params": get_params(3, 2, 60001, "all_features", 0.00001),
    "case_name": "normalize_batch_8",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
ut_case.add_case(["Ascend910A"], case6)
ut_case.add_case(["Ascend910A"], case7)
ut_case.add_case(["Ascend910A"], case8)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
