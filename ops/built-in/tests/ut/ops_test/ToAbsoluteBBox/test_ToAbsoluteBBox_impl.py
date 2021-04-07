#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ToAbsoluteBBox", "impl.to_absolute_bbox", "to_absolute_bbox")

def gen_absolute_bbox_case(boxes, shape_hw, dtype, reversed_box,
                            case_name_val, expect):
    return {"params": [{"shape": boxes, "dtype": dtype, "ori_shape": boxes,
                        "ori_format": "ND", "format": "ND"},
                       {"shape": shape_hw, "dtype": "int32", "ori_shape": shape_hw,
                        "ori_format": "ND", "format": "ND"},
                       {"shape": boxes, "dtype": dtype, "ori_shape": boxes,
                        "ori_format": "ND", "format": "ND"},
                       reversed_box],
            "case_name": case_name_val,
            "expect": expect,
            "support_expect": True}

ut_case.add_case(["Ascend310"],
                 gen_absolute_bbox_case((25, 4, 16), (4, ), "float16",
                 True, "case_1", "success"))

ut_case.add_case(["Ascend310"],
                 gen_absolute_bbox_case((1, 4, 1600000), (4, ), "float16",
                 True, "case_2", "success"))

ut_case.add_case(["Ascend310"],
                 gen_absolute_bbox_case((25, 4, 15), (4, ), "float16",
                 True, "case_3", "success"))

ut_case.add_case(["Ascend310"],
                 gen_absolute_bbox_case((1, 4, 1600001), (4, ), "float16",
                 True, "case_4", "success"))

ut_case.add_case(["Ascend310"],
                 gen_absolute_bbox_case((25, 4, 8), (4, ), "float32",
                 True, "case_5", "success"))

ut_case.add_case(["Ascend310"],
                 gen_absolute_bbox_case((1, 4, 1600000), (4, ), "float32",
                 True, "case_6", "success"))

ut_case.add_case(["Ascend310"],
                 gen_absolute_bbox_case((25, 4, 15), (4, ), "float32",
                 True, "case_7", "success"))

ut_case.add_case(["Ascend310"],
                 gen_absolute_bbox_case((1, 4, 1600001), (4, ), "float32",
                 True, "case_8", "success"))

ut_case.add_case(["Ascend310"],
                 gen_absolute_bbox_case((25, 4, 4), (4, ), "float16",
                 False, "case_9", "success"))

ut_case.add_case(["Ascend310"],
                 gen_absolute_bbox_case((1, 1600000, 4), (4, ), "float16",
                 False, "case_10", "success"))

ut_case.add_case(["Ascend310"],
                 gen_absolute_bbox_case((25, 3, 4), (4, ), "float16",
                 False, "case_11", "success"))

ut_case.add_case(["Ascend310"],
                 gen_absolute_bbox_case((1, 1600001, 4), (4, ), "float16",
                 False, "case_12", "success"))

ut_case.add_case(["Ascend310"],
                 gen_absolute_bbox_case((25, 2, 4), (4, ), "float32",
                 False, "case_13", "success"))

ut_case.add_case(["Ascend310"],
                 gen_absolute_bbox_case((1, 1600000, 4), (4, ), "float32",
                 False, "case_14", "success"))

ut_case.add_case(["Ascend310"],
                 gen_absolute_bbox_case((25, 3, 4), (4, ), "float32",
                 False, "case_15", "success"))

ut_case.add_case(["Ascend310"],
                 gen_absolute_bbox_case((1, 1600001, 4), (4, ), "float32",
                 False, "case_16", "success"))

ut_case.add_case(["Ascend310"],
                 gen_absolute_bbox_case((25, 4, 15), (4, ), "float32",
                 True, "case_17", "success"))

# ut_case.add_case(["Ascend310"],
#                  gen_absolute_bbox_case((25, 100, 3), (4, ), "float32",
#                  False, "err_1", "RuntimeError"))

# ut_case.add_case(["Ascend310"],
#                  gen_absolute_bbox_case((25, 3, 100), (4, ), "float32",
#                  True, "err_2", "RuntimeError"))

# ut_case.add_case(["Ascend310"],
#                  gen_absolute_bbox_case((25, 100, 3, 4), (4, ), "float32",
#                  False, "err_3", "RuntimeError"))

# ut_case.add_case(["Ascend310"],
#                  gen_absolute_bbox_case((25, 3, 100, 5), (4, ), "float32",
#                  True, "err_4", "RuntimeError"))

# ut_case.add_case(["Ascend310"],
#                  gen_absolute_bbox_case((25, 4, 100), (2,), "float32",
#                  True, "err_5", "RuntimeError"))

# ut_case.add_case(["Ascend310"],
#                  gen_absolute_bbox_case((25, 100, 4), (2,), "float32",
#                                         False, "err_6", "RuntimeError"))

# ut_case.add_case(["Ascend310"],
#                  gen_absolute_bbox_case((25, 4, 100), (2, 2), "float32",
#                                         True, "err_7", "RuntimeError"))

# ut_case.add_case(["Ascend310"],
#                  gen_absolute_bbox_case((25, 100, 4), (2, 2), "float32",
#                                         False, "err_8", "RuntimeError"))

# ut_case.add_case(["Ascend310"],
#                  gen_absolute_bbox_case((25, 100, 4), (4,), "int32",
#                                         False, "err_9", "RuntimeError"))

if __name__ == '__main__':
    ut_case.run("Ascend310")
    exit(0)
