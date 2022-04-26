#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT

ut_case = OpUT("Fill", "impl.dynamic.fill", "fill")

case1 = {"params": [{"shape": (-1,), "ori_shape": (2,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC",
                     'dtype': "int32"},
                    {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), "format": "NHWC", "ori_format": "NHWC",
                     'dtype': "int32"},
                    {"shape": (-1,), "ori_shape": (2,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC",
                     'dtype': "int32"}],
         "case_name": "fill_dynamic_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (-1,), "ori_shape": (3,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC",
                     'dtype': "int32"},
                    {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), "format": "NHWC", "ori_format": "NHWC",
                     'dtype': "float16"},
                    {"shape": (-1,), "ori_shape": (3,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC",
                     'dtype': "float16"}],
         "case_name": "fill_dynamic_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (-1,), "ori_shape": (3,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC",
                     'dtype': "int32"},
                    {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), "format": "NHWC", "ori_format": "NHWC",
                     'dtype': "float32"},
                    {"shape": (-1,), "ori_shape": (3,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC",
                     'dtype': "float32"}],
         "case_name": "fill_dynamic_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (-1,), "ori_shape": (3,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC",
                     'dtype': "int32"},
                    {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), "format": "NHWC", "ori_format": "NHWC",
                     'dtype': "float32"},
                    {"shape": (-1,), "ori_shape": (3,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC",
                     'dtype': "float32"}],
         "case_name": "fill_dynamic_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [{"shape": (2,), "ori_shape": (2,), "range": ((2, 2)), "format": "ND", "ori_format": "ND",
                     'dtype': "int32", "const_value": [4, 4]},
                    {"shape": (-1,), "ori_shape": (-1,), "range": ((1, 1),), "format": "ND", "ori_format": "ND",
                     'dtype': "float32"},
                    {"shape": (4, 4), "ori_shape": (4, 4), "range": ((4, 4), (4, 4)), "format": "ND", "ori_format": "ND",
                     'dtype': "float32"}],
         "case_name": "fill_dynamic_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)

if __name__ == '__main__':
    ut_case.run("Ascend910")
