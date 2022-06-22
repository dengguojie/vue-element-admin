#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("GroupNorm", "impl.dynamic.group_norm", "op_select_format")


ut_case.add_case("Ascend910A", {
    "params": [{'shape': (13, 16, 17, 19), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (13, 16, 17, 19), 'ori_format': 'ND'},
               {'shape': (16,), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (16,), 'ori_format': 'ND'},
               {'shape': (16,), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (16,), 'ori_format': 'ND'},
               {'shape': (13, 16, 17, 19), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (13, 16, 17, 19), 'ori_format': 'ND'},
               {'shape': (13,), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (13,), 'ori_format': 'ND'},
               {'shape': (13,), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (13,), 'ori_format': 'ND'}, 1, "NCHW", 1e-5, False
               ],
    "expect": "success"
})

if __name__ == '__main__':
    ut_case.run("Ascend910A")
