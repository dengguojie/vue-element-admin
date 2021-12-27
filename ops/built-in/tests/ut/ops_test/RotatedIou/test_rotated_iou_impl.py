# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import OpUT

ut_case = OpUT("RotatedIou")

ut_case.add_case(["Ascend910A","Ascend710"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (8, 5, 16), "shape": (8, 5, 16),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (8, 5, 3), "shape": (8, 5, 3),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (8, 16, 3), "shape": (8, 16, 3),
                "param_type": "output"},
                False,'iou',True,0,0],
    "case_name": "test0",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

ut_case.add_case(["Ascend910A","Ascend710"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 5, 16), "shape": (1, 5, 16),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 5, 3), "shape": (1, 5, 3),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 16, 3), "shape": (1, 16, 3),
                "param_type": "output"},
                True,'iou',True,0,0],
    "case_name": "test1",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

ut_case.add_case(["Ascend910A","Ascend710"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 5, 1), "shape": (1, 5, 1),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 5, 1), "shape": (1, 5, 1),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 1), "shape": (1, 1, 1),
                "param_type": "output"},
                True,'iou',True,0,0],
    "case_name": "test2",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

ut_case.add_case(["Ascend910A","Ascend710"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (8, 5, 65472), "shape": (8, 5, 65472),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (8, 5, 60), "shape": (8, 5, 60),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (8, 65472, 60), "shape": (8, 65472, 60),
                "param_type": "output"},
                True,'iou',True,0,0],
    "case_name": "test3",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

ut_case.add_case(["Ascend910A","Ascend710"], {
    "params": [{"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 5, 1), "shape": (1, 5, 1),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 5, 1), "shape": (1, 5, 1),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 1), "shape": (1, 1, 1),
                "param_type": "output"},
                True,'iou',True,0,0],
    "case_name": "test4",
    "expect": "failed",
    "format_expect": ["ND"],
    "support_expect": True})

ut_case.add_case(["Ascend910A","Ascend710"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (8, 5, 6000), "shape": (8, 5, 6000),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (8, 5, 3000), "shape": (8, 5, 3000),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (8, 6000, 3000), "shape": (8, 6000, 3000),
                "param_type": "output"},
                True,'iou',True,0,0],
    "case_name": "test5",
    "expect": "failed",
    "format_expect": ["ND"],
    "support_expect": True})

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)