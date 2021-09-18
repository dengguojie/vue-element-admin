#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.fully_connection import get_op_support_info
from impl.fully_connection import op_select_format

from op_test_frame.ut import OpUT
ut_case = OpUT("FullyConnection", None, None)

case1 = {"params": [{'shape': (1, 2, 4, 2, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                     "ori_format":"NC1HWC0", "ori_shape":(1, 2, 4, 2, 16)},
                    {'shape': (16, 1, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z',
                     "ori_format":"FRACTAL_Z", "ori_shape":(16, 1, 16, 16)},
                    None, None,
                    {'shape': (1, 1, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                     "ori_format":"NC1HWC0", "ori_shape":(1, 1, 1, 1, 16)},
                    16, False, 1, 0],
         "case_name": "fully_connection_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{'shape': (8, 1, 2, 2, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                     "ori_format":"NC1HWC0", "ori_shape":(8, 1, 2, 2, 16)},
                    {'shape': (4, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z',
                     "ori_format":"FRACTAL_Z", "ori_shape":(4, 2, 16, 16)},
                    {'shape': (1, 2, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                     "ori_format":"NC1HWC0", "ori_shape":(1, 2, 1, 1, 16)},
                    None,
                    {'shape': (1, 1, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                     "ori_format":"NC1HWC0", "ori_shape":(1, 1, 1, 1, 16)},
                    32, False, 1, 0],
         "case_name": "fully_connection_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{'shape': (256, 19, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_NZ',
                     "ori_format":"NCHW", "ori_shape":(304, 4096)},
                    {'shape': (256, 256, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z',
                     "ori_format":"NCHW", "ori_shape":(4096, 4096, 1, 1)},
                    {'shape': (1, 256, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                     "ori_format":"NCHW", "ori_shape":(4096, )},
                    None,
                    {'shape': (256, 19, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_NZ',
                     "ori_format":"NCHW", "ori_shape":(304, 4096)},
                    4096, False, 1, 0],
         "case_name": "fully_connection_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{'shape': (8, 1, 2, 2, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                     "ori_format":"NC1HWC0", "ori_shape":(8, 1, 2, 2, 16)},
                    {'shape': (1, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z',
                     "ori_format":"FRACTAL_Z", "ori_shape":(1, 2, 16, 16)},
                    {'shape': (1, 2, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                     "ori_format":"NC1HWC0", "ori_shape":(1, 2, 1, 1, 16)},
                    None,
                    {'shape': (1, 1, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                     "ori_format":"NC1HWC0", "ori_shape":(1, 1, 1, 1, 16)},
                    32, False, 1, 0],
         "case_name": "fully_connection_failed_0",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

gevm_L1_attach_equal = {"params": [{'shape': (1, 1, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                        "ori_format":"NHWC", "ori_shape":(1, 1, 1, 16)},
                        {'shape': (1, 1, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z',
                        "ori_format":"HWCN", "ori_shape":(1, 1, 16, 8)},
                        {'shape': (1, 1, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                        "ori_format":"NHWC", "ori_shape":(8, )},
                        None,
                        {'shape': (1, 1, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                        "ori_format":"NHWC", "ori_shape":(1, 1, 1, 8)},
                        32, False, 1, 0],
                        "case_name": "gevm_L1_attach_equal",
                        "expect": "success",
                        "format_expect": [],
                        "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], gevm_L1_attach_equal)
# ND -> ND
def test_split_fc(test_arg):
    x = {'shape': (8, 1, 2, 2, 16), 'dtype': 'float16', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(8, 1, 2, 2, 16)}
    w = {'shape': (4, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z', "ori_format":"FRACTAL_Z", "ori_shape":(4, 2, 16, 16)}
    b =  {'shape': (1, 2, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(1, 2, 1, 1, 16)}
    y = {'shape': (8, 2, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(8, 2, 1, 1, 16)}
    get_op_support_info(x, w, b, None, y, 32, False, 1)
    op_select_format(x, w, b, None, y, 32, False, 1, 0)

# NZ -> NZ with batch
def test_split_fc_1(test_arg):
    x = {'shape': (2, 1, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_NZ', "ori_format":"FRACTAL_NZ", "ori_shape":(2, 1, 2, 16, 16)}
    w = {'shape': (1, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z', "ori_format":"FRACTAL_Z", "ori_shape":(1, 2, 16, 16)}
    b =  {'shape': (1, 2, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(1, 2, 1, 1, 16)}
    y = {'shape': (2, 2, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_NZ', "ori_format":"FRACTAL_NZ", "ori_shape":(2, 2, 2, 16, 16)}
    get_op_support_info(x, w, b, None, y, 32, False, 2)
    op_select_format(x, w, b, None, y, 32, False, 2, 0)

# NZ -> NZ no batch
def test_split_fc_2(test_arg):
    x = {'shape': (1, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_NZ', "ori_format":"FRACTAL_NZ", "ori_shape":(1, 2, 16, 16)}
    w = {'shape': (1, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z', "ori_format":"FRACTAL_Z", "ori_shape":(1, 2, 16, 16)}
    y = {'shape': (2, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_NZ', "ori_format":"FRACTAL_NZ", "ori_shape":(2, 2, 16, 16)}
    get_op_support_info(x, w, None, None, y, 32, False, 1)
    op_select_format(x, w, None, None, y, 32, False, 1, 0)

# ND -> NZ
def test_split_fc_3(test_arg):
    x = {'shape': (8, 1, 2, 2, 16), 'dtype': 'float16', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(8, 1, 2, 2, 16)}
    w = {'shape': (4, 1, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z', "ori_format":"FRACTAL_Z", "ori_shape":(1, 2, 16, 16)}
    y = {'shape': (1, 1, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_NZ', "ori_format":"FRACTAL_NZ", "ori_shape":(1, 1, 16, 16)}
    get_op_support_info(x, w, None, None, y, 16, False, 1)
    op_select_format(x, w, None, None, y, 16, False, 1, 0)

ut_case.add_cust_test_func(test_func=test_split_fc)
ut_case.add_cust_test_func(test_func=test_split_fc_1)
ut_case.add_cust_test_func(test_func=test_split_fc_2)
ut_case.add_cust_test_func(test_func=test_split_fc_3)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
