#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("SwapCi", None, None)

case1 = {"params": [{"shape": (1, 16, 3, 4), "dtype": "float16", "format": "NCHW", 'ori_shape':(1, 4, 3, 4), 'ori_format':"NCHW"},
                    {"shape": (1, 4, 3, 4, 16), "dtype": "float16", "format": "NC1HWC0", 'ori_shape':(1, 4, 3, 4, 16), 'ori_format':"NC1HWC0"},
                    4, 2],
         "case_name": "SwapCi_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

err1 = {"params": [{"shape": (1, 16, 3, 4,16), "dtype": "float16", "format": "NCHW", 'ori_shape':(1, 4, 3, 4), 'ori_format':"NCHW"},
                    {"shape": (1, 4, 3, 4, 16), "dtype": "float16", "format": "NC1HWC0", 'ori_shape':(1, 4, 3, 4, 16), 'ori_format':"NC1HWC0"},
                    4, 2],
         "case_name": "err_1",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
err2 = {"params": [{"shape": (1, 16, 3, 4), "dtype": "float16", "format": "NCHW", 'ori_shape':(1, 4, 3, 4), 'ori_format':"NCHW"},
                    {"shape": (1, 4, 3, 4), "dtype": "float16", "format": "NC1HWC0", 'ori_shape':(1, 4, 3, 4, 16), 'ori_format':"NC1HWC0"},
                    4, 2],
         "case_name": "err_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
err3 = {"params": [{"shape": (1, 16, 3, 4), "dtype": "float16", "format": "NCHW", 'ori_shape':(1, 4, 3, 4), 'ori_format':"NCHW"},
                    {"shape": (1, 16, 3, 4, 16), "dtype": "float16", "format": "NC1HWC0", 'ori_shape':(1, 4, 3, 4, 16), 'ori_format':"NC1HWC0"},
                    4, 2],
         "case_name": "err_3",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}


ut_case.add_case("Ascend910", case1)
ut_case.add_case("Ascend910", err1)
ut_case.add_case("Ascend910", err2)
ut_case.add_case("Ascend910", err3)

if __name__ == '__main__':
    ut_case.run("Ascend910")