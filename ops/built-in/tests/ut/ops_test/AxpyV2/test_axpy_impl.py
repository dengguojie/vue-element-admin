#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("AxpyV2", None, None)

ut_case.add_case("all", {
    "params": [{'shape': (13, 15, 17, 16), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 16), 'ori_format': 'ND'},
               {'shape': (13, 15, 17, 16), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 16), 'ori_format': 'ND'},
               2.0,
               {'shape': (13, 15, 17, 16), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17 ,16), 'ori_format': 'ND'}
               ],
    "expect": RuntimeError
})


def test_op_select_format(test_args):
    """
    test_op_select_format
    """
    from impl.axpy_v2 import op_select_format
    op_select_format({"shape": (1, 8, 2, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 8, 2, 16),
                      "ori_format": "ND"},
                     {"shape": (1, 8, 2, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 8, 2, 16),
                      "ori_format": "ND"},
                     {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (1, 8, 2, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 8, 2, 16),
                      "ori_format": "ND"},
                     "test_add_op_select_format_1")
    op_select_format({"shape": (1, 8, 2, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 8, 2, 16),
                      "ori_format": "ND"},
                     {"shape": (1, 8, 2, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 8, 2, 1),
                      "ori_format": "ND"},
                     {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (1, 8, 2, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 8, 2, 16),
                      "ori_format": "ND"},
                     "test_add_op_select_format_2")
#    op_select_format({"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
#                      "ori_format": "HWCN", "sub_format" : 1},
#                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
#                      "ori_format": "HWCN", "sub_format" : 1},
#                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
#                      "ori_format": "HWCN", "sub_format" : 1},
#                     "test_add_op_select_format_3")

ut_case.add_cust_test_func(test_func=test_op_select_format)

if __name__ == '__main__':
    ut_case.run("Ascend910A", simulator_mode="pv", simulator_lib_path="/usr/local/Ascend/toolkit/tools/simulator")
