#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("FullyConnection", None, None)

case1 = {"params": [{'shape': (1, 2, 4, 2, 16), 'dtype': 'float16', 'format': 'NC1HWC0',"ori_format":"NC1HWC0","ori_shape":(1, 2, 4, 2, 16)},
                    {'shape': (16, 1, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z',"ori_format":"FRACTAL_Z","ori_shape":(16, 1, 16, 16)},
                    None, None,
                    {'shape': (1, 1, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',"ori_format":"NC1HWC0","ori_shape":(1, 1, 1, 1, 16)},
                    16, False, 1, 0],
         "case_name": "fully_connection_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{'shape': (8, 1, 2, 2, 16), 'dtype': 'float16', 'format': 'NC1HWC0',"ori_format":"NC1HWC0","ori_shape":(8, 1, 2, 2, 16)},
                    {'shape': (4, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z',"ori_format":"FRACTAL_Z","ori_shape":(4, 2, 16, 16)},
                    {'shape': (1, 2, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',"ori_format":"NC1HWC0","ori_shape":(1, 2, 1, 1, 16)},
                    None,
                    {'shape': (1, 1, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',"ori_format":"NC1HWC0","ori_shape":(1, 1, 1, 1, 16)},
                    32, False, 1, 0],
         "case_name": "fully_connection_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    # ut_case.run()
    exit(0)