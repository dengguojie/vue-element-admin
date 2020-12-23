#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("FullyConnectionCompress", "impl.compress_fully_connection", "compress_fully_connection")
"""
def compress_fully_connection(x, w, compress_index, b, offset_w, y,
                              num_output, transpose, axis, offset_x,
                              kernel_name="compress_fully_connection"):
"""
compress_index = {"shape": (1, ), "dtype": "int8", "format": "ND", "ori_shape": (1, ), "ori_format": "ND"}

case1 = {"params": [{"shape": (1, 16, 1, 1, 32), "dtype": "int8", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 16, 1, 1, 32)},
                    {"shape": (16, 1, 16, 32), "dtype": "int8", "format": "FRACTAL_Z", "ori_format": "FRACTAL_Z", "ori_shape": (16, 1, 16, 32)},
                    compress_index, None, None,
                    {"shape": (1, 1, 1, 1, 16), "dtype": "int32", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16)},
                    128, False, 1, 0],
         "case_name": "compress_fully_connection_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (32, 1, 1, 1, 32), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (32, 1, 1, 1, 32), "ori_format": "NC1HWC0"},
                    {"shape": (1, 8, 16, 32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (1, 8, 16, 32), "ori_format": "FRACTAL_Z"},
                    compress_index, 
                    {"shape": (1, 8, 1, 1, 16), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (1, 8, 1, 1, 16), "ori_format": "NC1HWC0"},
                    None, {"shape": (1, 1, 1, 1, 16), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"},
                    128, False, 1, 0],
         "case_name": "compress_fully_connection_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (1, 16, 1, 1, 32), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (1, 16, 1, 1, 32), "ori_format": "NC1HWC0"},
                    {"shape": (16, 1, 16, 32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (16, 1, 16, 32), "ori_format": "FRACTAL_Z"},
                    compress_index, None, None,
                    {"shape": (1, 1, 1, 1, 16), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"},
                    128, True, 1, 0],
         "case_name": "compress_fully_connection_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)

if __name__ == "__main__":
    ut_case.run("Ascend910")
    # ut_case.run()
    exit(0)