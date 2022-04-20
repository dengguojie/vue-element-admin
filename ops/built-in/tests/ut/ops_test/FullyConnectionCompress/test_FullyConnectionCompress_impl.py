#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.compress_fully_connection import get_op_support_info

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

case4 = {"params": [{"shape": (1, 1, 1, 1, 33), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 33), "ori_format": "NC1HWC0"},
                    {"shape": (1, 1, 1, 33), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (1, 1, 1, 33), "ori_format": "FRACTAL_Z"},
                    compress_index, None, None,
                    {"shape": (1, 1, 1, 1, 33), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 33), "ori_format": "NC1HWC0"},
                    128, True, 1, 0],
         "case_name": "compress_fully_connection_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [{"shape": (1, 1, 1, 1, 33), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 33), "ori_format": "NC1HWC0"},
                    {"shape": (1, 1, 1, 33), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (1, 1, 1, 33), "ori_format": "FRACTAL_Z"},
                    compress_index, None, None,
                    {"shape": (1, 1, 1, 1, 33), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 33), "ori_format": "NC1HWC0"},
                    128, True, 1, 0],
         "case_name": "compress_fully_connection_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case6 = {"params": [{"shape": (1, 16, 1, 1, 32), "dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 16, 1, 1, 32)},
                    {"shape": (16, 1, 16, 32), "dtype": "float16", "format": "FRACTAL_Z", "ori_format": "FRACTAL_Z", "ori_shape": (16, 1, 16, 32)},
                    compress_index, None, None,
                    {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16)},
                    128, False, 1, 0],
         "case_name": "compress_fully_connection_6",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [{"shape": (1, 16, 1, 1, 32), "dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 16, 1, 1, 32)},
                    {"shape": (16, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_format": "FRACTAL_Z", "ori_shape": (16, 1, 16, 32)},
                    compress_index, None, None,
                    {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16)},
                    128, False, 1, 0],
         "case_name": "compress_fully_connection_7",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case8 = {"params": [{"shape": (1, 16, 1, 1, 32), "dtype": "int8", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 16, 1, 1, 32)},
                    {"shape": (16, 1, 16, 16), "dtype": "int8", "format": "FRACTAL_Z", "ori_format": "FRACTAL_Z", "ori_shape": (16, 1, 16, 32)},
                    compress_index, None, None,
                    {"shape": (1, 1, 1, 1, 16), "dtype": "int32", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16)},
                    128, False, 1, 0],
         "case_name": "compress_fully_connection_8",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case9 = {"params": [{"shape": (1, 16, 1, 1, 32), "dtype": "int8", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 16, 1, 1, 32)},
                    {"shape": (16, 1, 16, 32), "dtype": "int8", "format": "FRACTAL_Z", "ori_format": "FRACTAL_Z", "ori_shape": (16, 1, 16, 32)},
                    compress_index, None, None,
                    {"shape": (1, 1, 1, 1, 16), "dtype": "int32", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16)},
                    128, False, 3, 0],
         "case_name": "compress_fully_connection_9",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case10 = {"params": [{"shape": (32, 1, 1, 1, 32), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (32, 1, 1, 1, 32), "ori_format": "NC1HWC0"},
                    {"shape": (1, 8, 16, 32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (1, 8, 16, 32), "ori_format": "FRACTAL_Z"},
                    compress_index,
                    {"shape": (1, 8, 1, 1, 8), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (1, 8, 1, 1, 16), "ori_format": "NC1HWC0"},
                    None, {"shape": (1, 1, 1, 1, 16), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"},
                    128, False, 1, 0],
         "case_name": "compress_fully_connection_10",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case11 = {"params": [{"shape": (32, 1, 1, 1, 32), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (32, 1, 1, 1, 32), "ori_format": "NC1HWC0"},
                    {"shape": (1, 8, 16, 32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (1, 8, 16, 32), "ori_format": "FRACTAL_Z"},
                    compress_index, None,
                    {"shape": (1, 8, 16, 32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (1, 8, 16, 32), "ori_format": "FRACTAL_Z"},
                    {"shape": (1, 1, 1, 1, 16), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"},
                    128, False, 1, 0],
         "case_name": "compress_fully_connection_11",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case8)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case9)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case10)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case11)

# ND -> ND
def test_split_fc_compress(test_arg):
    compress_index = {"shape": (1, ), "dtype": "int8", "format": "ND", "ori_shape": (1, ), "ori_format": "ND"}
    x = {'shape': (8, 1, 2, 2, 32), 'dtype': 'int8', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(8, 1, 2, 2, 32)}
    w = {'shape': (4, 2, 16, 32), 'dtype': 'int8', 'format': 'FRACTAL_Z', "ori_format":"FRACTAL_Z", "ori_shape":(4, 2, 16, 32)}
    b = {'shape': (1, 2, 1, 1, 16), 'dtype': 'int32', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(1, 2, 1, 1, 16)}
    y = {'shape': (8, 2, 1, 1, 16), 'dtype': 'int32', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(8, 2, 1, 1, 16)}
    get_op_support_info(x, w, compress_index, b, None, y, 32, False, 1, 0, kernel_name="compress_fully_connection")

# NZ -> NZ with batch
def test_split_fc_compress_1(test_arg):
    compress_index = {"shape": (1, ), "dtype": "int8", "format": "ND", "ori_shape": (1, ), "ori_format": "ND"}
    x = {'shape': (2, 1, 2, 16, 32), 'dtype': 'int8', 'format': 'FRACTAL_NZ', "ori_format":"FRACTAL_NZ", "ori_shape": (2, 1, 2, 16, 32)}
    w = {'shape': (1, 2, 16, 32), 'dtype': 'int8', 'format': 'FRACTAL_Z', "ori_format":"FRACTAL_Z", "ori_shape": (1, 2, 16, 32)}
    b = {'shape': (1, 2, 1, 1, 16), 'dtype': 'int32', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(1, 2, 1, 1, 16)}
    y = {'shape': (2, 2, 2, 16, 16), 'dtype': 'int32', 'format': 'FRACTAL_NZ', "ori_format":"FRACTAL_NZ", "ori_shape":(2, 2, 2, 16, 16)}
    get_op_support_info(x, w, compress_index, b, None, y, 16, False, 2, 0, kernel_name="compress_fully_connection")

# NZ -> NZ no batch
def test_split_fc_compress_2(test_arg):
    compress_index = {"shape": (1, ), "dtype": "int8", "format": "ND", "ori_shape": (1, ), "ori_format": "ND"}
    x = {'shape': (1, 2, 16, 32), 'dtype': 'int8', 'format': 'FRACTAL_NZ', "ori_format":"FRACTAL_NZ", "ori_shape":(1, 2, 16, 32)}
    w = {'shape': (1, 2, 16, 32), 'dtype': 'int8', 'format': 'FRACTAL_Z', "ori_format":"FRACTAL_Z", "ori_shape":(1, 2, 16, 32)}
    y = {'shape': (2, 2, 16, 16), 'dtype': 'int32', 'format': 'FRACTAL_NZ', "ori_format":"FRACTAL_NZ", "ori_shape":(2, 2, 16, 16)}
    get_op_support_info(x, w, compress_index, None, None, y, 32, False, 1, 0, kernel_name="compress_fully_connection")

# ND -> NZ
def test_split_fc_compress_3(test_arg):
    compress_index = {"shape": (1, ), "dtype": "int8", "format": "ND", "ori_shape": (1, ), "ori_format": "ND"}
    x = {'shape': (8, 1, 2, 2, 32), 'dtype': 'int8', 'format': 'FRACTAL_NZ', "ori_format":"NC1HWC0", "ori_shape":(8, 1, 2, 2, 32)}
    w = {'shape': (4, 1, 16, 32), 'dtype': 'int8', 'format': 'FRACTAL_Z', "ori_format":"FRACTAL_Z", "ori_shape":(4, 1, 16, 32)}
    y = {'shape': (1, 1, 16, 16), 'dtype': 'int32', 'format': 'FRACTAL_NZ', "ori_format":"FRACTAL_NZ", "ori_shape":(1, 1, 16, 16)}
    get_op_support_info(x, w, compress_index, None, None, y, 16, False, 1, 0, kernel_name="compress_fully_connection")


def test_op_select_format(test_arg):
    from impl.compress_fully_connection import op_select_format
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
ut_case.add_cust_test_func(test_func=test_split_fc_compress)
ut_case.add_cust_test_func(test_func=test_split_fc_compress_1)
ut_case.add_cust_test_func(test_func=test_split_fc_compress_2)
ut_case.add_cust_test_func(test_func=test_split_fc_compress_3)

if __name__ == "__main__":
    ut_case.run("Ascend910")
    # ut_case.run()
    exit(0)
