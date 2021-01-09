#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
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

if __name__ == '__main__':
    ut_case.run("Ascend910", simulator_mode="pv", simulator_lib_path="/usr/local/Ascend/toolkit/tools/simulator")
