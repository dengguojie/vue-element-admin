#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
ut of resizebilinearv2grad
"""

from op_test_frame.ut import OpUT

ut_case = OpUT("ResizeBilinearV2Grad", "impl.dynamic.resize_bilinear_v2_grad", "resize_bilinear_v2_grad")

case1 = {"params": [
    {"shape": (32, 1, 256, 256, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (32, 16, 256, 256),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (32, 1, 128, 128, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (32, 16, 128, 128),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (32, 1, 128, 128, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (32, 16, 128, 128),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    False, True],
         "case_name": "dynamic_resize_bilinear_v2_grad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [
    {"shape": (-1, -1, -1, -1, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (32, 2, 1, 1, 16),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (-1, -1, -1, -1, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (32, 2, 1, 1, 16),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (-1, -1, -1, -1, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (32, 2, 1, 1, 16),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    False, False],
         "case_name": "dynamic_resize_bilinear_v2_grad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [
    {"shape": (-1, -1, -1, -1, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (35, 2, 1, 1, 16),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (-1, -1, -1, -1, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (35, 2, 1, 1, 16),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (-1, -1, -1, -1, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (35, 2, 1, 1, 16),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    True, False],
         "case_name": "dynamic_resize_bilinear_v2_grad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case("all", case1)
ut_case.add_case("all", case2)
ut_case.add_case("all", case3)

if __name__ == '__main__':
    import te

    with te.op.dynamic():
        ut_case.run("Ascend910")
