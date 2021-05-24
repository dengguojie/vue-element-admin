#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.fully_connection import get_op_support_info

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

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)

def test_split_fc(test_arg):
    x = {'shape': (8, 1, 2, 2, 16), 'dtype': 'float16', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(8, 1, 2, 2, 16)}
    w = {'shape': (1, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z', "ori_format":"FRACTAL_Z", "ori_shape":(1, 2, 16, 16)}
    b =  {'shape': (1, 2, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(1, 2, 1, 1, 16)}
    y = {'shape': (1, 1, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(1, 1, 1, 1, 16)}
    get_op_support_info(x, w, b, None, y, 16, False, 1)

def test_split_fc_1(test_arg):
    x = {'shape': (2, 1, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_NZ', "ori_format":"FRACTAL_NZ", "ori_shape":(2, 1, 2, 16, 16)}
    w = {'shape': (1, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z', "ori_format":"FRACTAL_Z", "ori_shape":(1, 2, 16, 16)}
    b =  {'shape': (1, 2, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(1, 2, 1, 1, 16)}
    y = {'shape': (2, 1, 1, 1, 16), 'dtype': 'float16', 'format': 'FRACTAL_NZ', "ori_format":"FRACTAL_NZ", "ori_shape":(2, 1, 1, 1, 16)}
    get_op_support_info(x, w, b, None, y, 16, False, 2)
ut_case.add_cust_test_func(test_func=test_split_fc)
ut_case.add_cust_test_func(test_func=test_split_fc_1)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
