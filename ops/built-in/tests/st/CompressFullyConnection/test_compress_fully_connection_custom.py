#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys

from op_test_frame.ut import OpUT
from impl.compress_fully_connection import compress_fully_connection
from impl.compress_fully_connection import op_select_format


ut_case = OpUT("FullyConnectionCompress", "impl.compress_fully_connection", "compress_fully_connection")


"""
def compress_fully_connection(x, w, compress_index, b, offset_w, y,
                              num_output, transpose, axis, offset_x,
                              kernel_name="compress_fully_connection"):
"""
compress_index = {"shape": (1, ), "dtype": "int8", "format": "ND", "ori_shape": (1, ), "ori_format": "ND"}

# ND -> ND
def test_fc_compress_impl(test_arg):
    compress_index = {"shape": (1, ), "dtype": "int8", "format": "ND", "ori_shape": (1, ), "ori_format": "ND"}
    x = {'shape': (8, 1, 2, 2, 32), 'dtype': 'int8', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(8, 1, 2, 2, 32)}
    w = {'shape': (4, 2, 16, 32), 'dtype': 'int8', 'format': 'FRACTAL_Z', "ori_format":"FRACTAL_Z", "ori_shape":(4, 2, 16, 32)}
    b = {'shape': (1, 2, 1, 1, 16), 'dtype': 'int32', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(1, 2, 1, 1, 16)}
    y = {'shape': (8, 2, 1, 1, 16), 'dtype': 'int32', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(8, 2, 1, 1, 16)}
    compress_fully_connection(x, w, compress_index, b, None, y, 32, False, 1, 0, kernel_name="compress_fully_connection")


def test_op_select_format(test_arg):
    op_select_format({'shape': (8, 1, 2, 2, 32), 'dtype': 'int8', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(8, 1, 2, 2, 32)},
                    {'shape': (4, 2, 16, 32), 'dtype': 'int8', 'format': 'FRACTAL_Z', "ori_format":"FRACTAL_Z", "ori_shape":(4, 2, 16, 32)},
                    {"shape": (1, ), "dtype": "int8", "format": "ND", "ori_shape": (1, ), "ori_format": "ND"},
                    {'shape': (1, 2, 1, 1, 16), 'dtype': 'int32', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(1, 2, 1, 1, 16)},
                    None,
                    {'shape': (8, 2, 1, 1, 16), 'dtype': 'int32', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(8, 2, 1, 1, 16)},
                    32, False, 1, 0, kernel_name="test_compress_fully_connection_op_select_format")
    op_select_format({'shape': (8, 1, 2, 2, 32), 'dtype': 'int8', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(8, 1, 2, 2, 32)},
                    {'shape': (4, 2, 16, 32), 'dtype': 'int8', 'format': 'FRACTAL_Z', "ori_format":"FRACTAL_Z", "ori_shape":(4, 2, 16, 32)},
                    {"shape": (1, ), "dtype": "int8", "format": "ND", "ori_shape": (1, ), "ori_format": "ND"},
                    {'shape': (1, 2, 1, 1, 16), 'dtype': 'int32', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(1, 2, 1, 1, 16)},
                    None,
                    {'shape': (8, 2, 1, 1, 16), 'dtype': 'int32', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(8, 2, 1, 1, 16)},
                    32, True, 1, 0, kernel_name="test_compress_fully_connection_op_select_format")

ut_case.add_cust_test_func(test_func=test_op_select_format)
ut_case.add_cust_test_func(test_func=test_fc_compress_impl)


if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend910A"])
    sys.exit(0)
