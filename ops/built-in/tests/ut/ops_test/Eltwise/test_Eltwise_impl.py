#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("Eltwise", None, None)

case1 = {"params": [[{"shape": (16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16),"ori_format": "NCHW"},
                     {"shape": (16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16),"ori_format": "NCHW"}],
                    {"shape": (16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16),"ori_format": "NCHW"},
                    1],
         "case_name": "eltwise_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [[{"shape": (16, 16), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 16),"ori_format": "NCHW"},
                     {"shape": (16, 16), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 16),"ori_format": "NCHW"}],
                    {"shape": (16, 16), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 16),"ori_format": "NCHW"},
                    1],
         "case_name": "eltwise_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

if __name__ == '__main__':
    ut_case.run("Ascend910")
