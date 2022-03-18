#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("AxpyV2", "impl.axpy_v2", "op_select_format")

ut_case.add_case("all", {
    "params": [{'shape': (13, 15, 17, 19), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND'},
               {'shape': (13, 15, 17, 19), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND'},
               {'shape': (1,), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND'},
               {'shape': (13, 15, 17, 19), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND'}],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (1,), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND'},
               {'shape': (13, 15, 17, 19), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND'},
               {'shape': (1,), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND'},
               {'shape': (13, 15, 17, 19), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND'}],
    "expect": "success"
})


ut_case.add_case("all", {
    "params": [
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
         'ori_format': 'ND'},
        {'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
         'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
         'ori_format': 'ND'},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
         'ori_format': 'ND'},
        {'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
         'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
         'ori_format': 'ND'}],
    "expect": "success"
})

if __name__ == '__main__':
    ut_case.run("Ascend910")
