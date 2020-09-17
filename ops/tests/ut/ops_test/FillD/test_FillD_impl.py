#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("FillD", None, None)

case1 = {"params": [{"shape": (1,), "ori_shape": (1,), "format": "NHWC", "ori_format": "NHWC", 'dtype': "int32"},
                    {"shape": (1,), "ori_shape": (1,), "format": "NHWC", "ori_format": "NHWC", 'dtype': "int32"},
                    (1,)],
         "case_name": "fill_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,), "ori_shape": (1,), "format": "NHWC", "ori_format": "NHWC", 'dtype': "float16"},
                    {"shape": (1,), "ori_shape": (1,), "format": "NHWC", "ori_format": "NHWC", 'dtype': "float16"},
                    (1,)],
         "case_name": "fill_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,), "ori_shape": (1,), "format": "NHWC", "ori_format": "NHWC", 'dtype': "float32"},
                    {"shape": (1,), "ori_shape": (1,), "format": "NHWC", "ori_format": "NHWC", 'dtype': "float32"},
                    (1,)],
         "case_name": "fill_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (1,), "ori_shape": (1,), "format": "NHWC", "ori_format": "NHWC", 'dtype': "int8"},
                    {"shape": (1,), "ori_shape": (1,), "format": "NHWC", "ori_format": "NHWC", 'dtype': "int8"},
                    (1,)],
         "case_name": "fill_d_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)

if __name__ == '__main__':
    ut_case.run("Ascend910")
