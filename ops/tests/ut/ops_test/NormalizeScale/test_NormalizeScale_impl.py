#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("NormalizeScale", None, None)

def normalize_scale_cce(shape_x1, dtype1, shape_x2, dtype2, shape_x3, dtype3, across_spatial=True, channel_shared=True, eps=1e-10, data_format="NCHW", case_name="normalize_scale"):

    return {"params": [{"shape": shape_x1, "dtype": dtype1, "format": data_format,"ori_shape":shape_x1, "ori_format":data_format},
                       {"shape": shape_x2, "dtype": dtype2, "format": "ND","ori_shape":shape_x2, "ori_format":"ND"},
                       {"shape": shape_x3, "dtype": dtype3, "format": data_format,"ori_shape":shape_x3, "ori_format":data_format},
                       {"shape": shape_x1, "dtype": dtype1, "format": data_format,"ori_shape":shape_x1, "ori_format":data_format},
                       across_spatial, channel_shared, eps],
            "case_name": case_name,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}


case1 = normalize_scale_cce((2, 3, 2, 3), "float16", (1,), "float16", (2, 1, 1, 1), "float32", True, True, 1e-10, "NCHW",
                            "normalize_scale_1")
case2 = normalize_scale_cce((2, 2, 3, 3), "float16", (1,), "float16", (2, 1, 1, 1), "float32", True, True, 1e-10, "NHWC",
                            "normalize_scale_2")
case3 = normalize_scale_cce((2, 3, 2, 3), "float16", (3,), "float16", (2, 1, 1, 1), "float32", True, False, 1e-10, "NCHW",
                            "normalize_scale_3")
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)