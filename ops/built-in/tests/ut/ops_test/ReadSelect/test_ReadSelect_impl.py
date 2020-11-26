#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("ReadSelect", None, None)

case1 = {"params": [{"shape": (1,16,13,56,16), "dtype": "float16", "valid_shape":(1,16,11,56,16), "slice_offset":[0,0,1,1,0],"src_in_flag":"DDR", "format":"NC1HWC0", "ori_shape":(1,16,13,56,16), "ori_format":"NC1HWC0"},
                    {"shape": (1,16,13,56,16), "dtype": "float16", "valid_shape":(1,16,11,56,16), "slice_offset":[0,0,1,1,0], "src_in_flag":"DDR", "format":"NC1HWC0", "ori_shape":(1,16,13,56,16), "ori_format":"NC1HWC0"}],
         "case_name": "read_select_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2, 4, 10, 5, 16), "dtype": "float16", "valid_shape":(), "slice_offset":[0,0,1,1,0],"src_in_flag":"DDR", "format":"NC1HWC0", "ori_shape":(2, 4, 10, 5, 16), "ori_format":"NC1HWC0"},
                    {"shape": (2, 4, 10, 5, 16), "dtype": "float16", "valid_shape":(), "slice_offset":[0,0,1,1,0], "src_in_flag":"DDR", "format":"NC1HWC0", "ori_shape":(2, 4, 10, 5, 16), "ori_format":"NC1HWC0"}],
         "case_name": "read_select_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,16,13,56,16), "dtype": "float16", "valid_shape":(1,16,11,56,16), "slice_offset":[0,0,1,1,0],"src_in_flag":"L1", "format":"NC1HWC0", "ori_shape":(1,16,13,56,16), "ori_format":"NC1HWC0"},
                    {"shape": (1,16,13,56,16), "dtype": "float16", "valid_shape":(1,16,11,56,16), "slice_offset":[0,0,1,1,0], "src_in_flag":"L1", "format":"NC1HWC0", "ori_shape":(1,16,13,56,16), "ori_format":"NC1HWC0"}],
         "case_name": "read_select_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2, 4, 10, 5, 16), "dtype": "float16", "valid_shape":(), "slice_offset":[0,0,1,1,0],"src_in_flag":"L1", "format":"NC1HWC0", "ori_shape":(2, 4, 10, 5, 16), "ori_format":"NC1HWC0"},
                    {"shape": (2, 4, 10, 5, 16), "dtype": "float16", "valid_shape":(), "slice_offset":[0,0,1,1,0], "src_in_flag":"L1", "format":"NC1HWC0", "ori_shape":(2, 4, 10, 5, 16), "ori_format":"NC1HWC0"}],
         "case_name": "read_select_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)


if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
