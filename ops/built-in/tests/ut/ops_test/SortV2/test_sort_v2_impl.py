# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("SortV2")

ut_case.add_case(["Ascend910A","Ascend920A"], {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10), "shape": (2, 10),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10), "shape": (2, 10),
                "param_type": "output"},
               -1, False],
    "case_name": "test0",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

ut_case.add_case(["Ascend910A","Ascend920A"], {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 40), "shape": (2, 40),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 40), "shape": (2, 40),
                "param_type": "output"},
               -1, True],
    "case_name": "test1",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

ut_case.add_case(["Ascend910A","Ascend920A"], {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 400), "shape": (2, 400),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 400), "shape": (2, 400),
                "param_type": "output"},
               -1, False],
    "case_name": "test2",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

ut_case.add_case(["Ascend910A","Ascend920A"], {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 4000), "shape": (2, 4000),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 4000), "shape": (2, 4000),
                "param_type": "output"},
               -1, True],
    "case_name": "test3",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

ut_case.add_case(["Ascend910A","Ascend920A"], {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 40000), "shape": (2, 40000),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 40000), "shape": (2, 40000),
                "param_type": "output"},
               -1, True],
    "case_name": "test4",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend920A"])
    exit(0)