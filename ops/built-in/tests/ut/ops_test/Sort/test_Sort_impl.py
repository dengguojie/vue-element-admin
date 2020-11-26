# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("Sort")

ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10), "shape": (2, 10),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10), "shape": (2, 10),
                "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10), "shape": (2, 10),
                "param_type": "output"},
               -1, False],
    "case_name": "test0",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10), "shape": (2, 10),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10), "shape": (2, 10),
                "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10), "shape": (2, 10),
                "param_type": "output"},
               -1, True],
    "case_name": "test1",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

# ut_case.add_case("all", {
#     "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10), "shape": (2, 10),
#                 "param_type": "input"},
#                {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10), "shape": (2, 10),
#                 "param_type": "output"},
#                {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10), "shape": (2, 10),
#                 "param_type": "output"},
#                0, True],
#     "case_name": "test2",
#     "expect": "RuntimeError",
#     "format_expect": ["ND"],
#     "support_expect": True})

# ut_case.add_case("all", {
#     "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10000), "shape": (2, 10000),
#                 "param_type": "input"},
#                {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10000), "shape": (2, 10000),
#                 "param_type": "output"},
#                {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10000), "shape": (2, 10000),
#                 "param_type": "output"},
#                -1, False],
#     "case_name": "test3",
#     "expect": "RuntimeError",
#     "format_expect": ["ND"],
#     "support_expect": True})


# [TODO] coding expect function here
def calc_expect_func1(x, y1, y2):  # descend
    x = x.get("value")
    y1 = np.sort(x)
    y2 = np.argsort(x)
    y1 = y1[:, ::-1]
    y2 = y2[:, ::-1]
    return y1, y2


def calc_expect_func(x, y1, y2):  # ascend
    x = x.get("value")
    y1 = np.sort(x)
    y2 = np.argsort(x)
    return y1, y2



# [TODO] coding cases here
# ut_case.add_precision_case("all", {
#     "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 480), "shape": (2, 480),
#                 "param_type": "input"},
#                {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 480), "shape": (2, 480),
#                 "param_type": "output"},
#                {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 480), "shape": (2, 480),
#                 "param_type": "output"},
#                -1, True],
#     "case_name": "test4",
#     "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
#     "calc_expect_func": calc_expect_func1
# })

# [TODO] coding cases here
# ut_case.add_precision_case("all", {
#     "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 6400), "shape": (2, 6400),
#                 "param_type": "input"},
#                {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 6400), "shape": (2, 6400),
#                 "param_type": "output"},
#                {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 6400), "shape": (2, 6400),
#                 "param_type": "output"}, -1, False],
#     "case_name": "test5",
#     "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
#     "calc_expect_func": calc_expect_func
# })

# ut_case.add_precision_case("all", {
#     "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10), "shape": (2, 10),
#                 "param_type": "input"},
#                {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10), "shape": (2, 10),
#                 "param_type": "output"},
#                {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10), "shape": (2, 10),
#                 "param_type": "output"}, -1, False],
#     "case_name": "test6",
#     "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
#     "calc_expect_func": calc_expect_func
# })

if __name__ == '__main__':
    ut_case.run("Ascend910")
