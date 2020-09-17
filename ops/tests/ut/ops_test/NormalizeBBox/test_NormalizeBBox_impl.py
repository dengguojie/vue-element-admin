#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("NormalizeBBox", "impl.normalize_bbox", "normalize_bbox")

def gen_normalize_bbox_case(boxes, shape_hw, dtype, reversed_box,
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
                 gen_normalize_bbox_case((25, 4, 16), (25, 3), "float16",
                 True, "case_1", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 4, 1600000), (1, 3), "float16",
                 True, "case_2", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((25, 4, 15), (25, 3), "float16",
                 True, "case_3", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 4, 1600001), (1, 3), "float16",
                 True, "case_4", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((25, 4, 16), (25, 3), "float32",
                 True, "case_5", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 4, 1600000), (1, 3), "float32",
                 True, "case_6", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((25, 4, 15), (25, 3), "float32",
                 True, "case_7", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 4, 1600001), (1, 3), "float32",
                 True, "case_8", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((25, 4, 4), (25, 3), "float16",
                 False, "case_9", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 1600000, 4), (1, 3), "float16",
                 False, "case_10", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((25, 5, 4), (25, 3), "float16",
                 False, "case_11", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 1600001, 4), (1, 3), "float16",
                 False, "case_12", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((25, 4, 4), (25, 3), "float32",
                 False, "case_13", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 1600000, 4), (1, 3), "float32",
                 False, "case_14", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((25, 5, 4), (25, 3), "float32",
                 False, "case_15", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 1600001, 4), (1, 3), "float32",
                 False, "case_16", "success"))

ut_case.add_case(["Ascend910"],
                 gen_normalize_bbox_case((25, 4, 15), (25, 3), "float32",
                 True, "case_17", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 16001, 5, 4), (1, 3), "float32",
                 False, "err_1", "RuntimeError"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 16001, 4), (1, 2, 3), "float16",
                 False, "err_2", "RuntimeError"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 16001, 5), (1, 3), "float32",
                 False, "err_3", "RuntimeError"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 6, 5), (1, 3), "float32",
                 True, "err_4", "RuntimeError"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 4, 5), (1, 2), "float32",
                 True, "err_5", "RuntimeError"))

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
