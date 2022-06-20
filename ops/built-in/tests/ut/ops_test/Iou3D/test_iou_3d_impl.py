# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import OpUT

ut_case = OpUT("Iou3D", "impl.iou_3d", "iou_3d")

ut_case.add_case(["Ascend910A","Ascend310p3"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (8, 7, 16), "shape": (8, 7, 16),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (8, 7, 3), "shape": (8, 7, 3),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (8, 16, 3), "shape": (8, 16, 3),
                "param_type": "output"}],
    "case_name": "test_shape_with_8_7_16_8_7_3",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

ut_case.add_case(["Ascend910A","Ascend710"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 7, 16), "shape": (1, 7, 16),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 7, 3), "shape": (1, 7, 3),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 16, 3), "shape": (1, 16, 3),
                "param_type": "output"}],
    "case_name": "test_shape_with_1_7_16_1_7_3",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

ut_case.add_case(["Ascend910A","Ascend710"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 7, 1), "shape": (1, 7, 1),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 7, 1), "shape": (1, 7, 1),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 1), "shape": (1, 1, 1),
                "param_type": "output"}],
    "case_name": "test_shape_with_1_7_1_1_7_1",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

ut_case.add_case(["Ascend910A","Ascend710"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (8, 7, 67472), "shape": (8, 7, 67472),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (8, 7, 60), "shape": (8, 7, 60),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (8, 67472, 60), "shape": (8, 67472, 60),
                "param_type": "output"}],
    "case_name": "test_shape_with_8_7_67642_8_7_60",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

ut_case.add_case(["Ascend910A","Ascend710"], {
    "params": [{"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (18, 7, 1111), "shape": (18, 7, 1111),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (18, 7, 1111), "shape": (18, 7, 1111),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (18, 1111, 1111), "shape": (18, 1111, 1111),
                "param_type": "output"}],
    "case_name": "test_shape_with_18_7_1111_18_7_1111",
    "expect": "failed",
    "format_expect": ["ND"],
    "support_expect": True})

ut_case.add_case(["Ascend910A","Ascend710"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (8, 7, 6000), "shape": (8, 7, 6000),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (8, 7, 3000), "shape": (8, 7, 3000),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (8, 6000, 3000), "shape": (8, 6000, 3000),
                "param_type": "output"}],
    "case_name": "test_shape_with_8_7_6000_8_7_3000",
    "expect": "failed",
    "format_expect": ["ND"],
    "support_expect": True})

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
