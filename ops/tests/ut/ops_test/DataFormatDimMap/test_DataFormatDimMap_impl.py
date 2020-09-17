#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("DataFormatDimMap", None, None)

case1 = {"params": [{"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"}],
         "case_name": "data_format_dim_map_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (1,2), "dtype": "int32", "format": "NCHW", "ori_shape": (1,2),"ori_format": "NCHW"},
                    {"shape": (1,2), "dtype": "int32", "format": "NCHW", "ori_shape": (1,2),"ori_format": "NCHW"}],
         "case_name": "data_format_dim_map_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

if __name__ == '__main__':
    ut_case.run()
    exit(0)