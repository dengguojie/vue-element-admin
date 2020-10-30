#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("MulNoNan", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1,), (1,))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 1), (1, 1))
#ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (16, 32), (16, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (16, 2, 32), (16, 2, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (16, 2, 4, 32), (16, 2, 4, 32))
#ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (512, 1024), (512, 1024))
#ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (2, 1024), (2, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (4096, 1024), (4096, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (32, 128, 1024), (32, 128, 1024))
#ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (100, 100), (100, 100))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 512, 1), (1,))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 16, 512, 512), (1, 1, 512, 512))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (9973, 1), (9973, 1))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1024, 1024, 256), (1024, 1024, 256))
#ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (11, 33), (11, 33))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (10, 12), (10, 11), expect=RuntimeError)
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (10, 13), (10, 11, 12), expect=RuntimeError)

ut_case.add_broadcast_case_simple(["Ascend310"], ["float16", "int32"], (1,), (1,))
ut_case.add_broadcast_case_simple(["Ascend310"], ["float16", "int32"], (1, 1), (1, 1))
#ut_case.add_broadcast_case_simple(["Ascend310"], ["float16", "int32"], (16, 32), (16, 32))
ut_case.add_broadcast_case_simple(["Ascend310"], ["float16", "int32"], (16, 2, 32), (16, 2, 32))
ut_case.add_broadcast_case_simple(["Ascend310"], ["float16", "int32"], (16, 2, 4, 32), (16, 2, 4, 32))
#ut_case.add_broadcast_case_simple(["Ascend310"], ["float16", "int32"], (512, 1024), (512, 1024))
#ut_case.add_broadcast_case_simple(["Ascend310"], ["float16", "int32"], (2, 1024), (2, 1024))
ut_case.add_broadcast_case_simple(["Ascend310"], ["float16", "int32"], (4096, 1024), (4096, 1024))
ut_case.add_broadcast_case_simple(["Ascend310"], ["float16", "int32"], (32, 128, 1024), (32, 128, 1024))
#ut_case.add_broadcast_case_simple(["Ascend310"], ["float16", "int32"], (100, 100), (100, 100))
ut_case.add_broadcast_case_simple(["Ascend310"], ["float16", "int32"], (1, 512, 1), (1,))
ut_case.add_broadcast_case_simple(["Ascend310"], ["float16", "int32"], (1, 16, 512, 512), (1, 1, 512, 512))
ut_case.add_broadcast_case_simple(["Ascend310"], ["float16", "int32"], (9973, 1), (9973, 1))
ut_case.add_broadcast_case_simple(["Ascend310"], ["float16", "int32"], (1024, 1024, 256), (1024, 1024, 256))
#ut_case.add_broadcast_case_simple(["Ascend310"], ["float16", "int32"], (11, 33), (11, 33))
ut_case.add_broadcast_case_simple(["Ascend310"], ["float16", "int32"], (10, 12), (10, 11), expect=RuntimeError)
ut_case.add_broadcast_case_simple(["Ascend310"], ["float16", "int32"], (10, 13), (10, 11, 12), expect=RuntimeError)

def calc_expect_func(input_x, input_y, output):
    shape_x=input_x.get("shape")
    shape_y=input_y.get("shape")
    output_arr=input_x.get("value")*input_y.get("value")
    return output_arr

ut_case.add_precision_case("all", {
    "params": [{'shape': (16, 32), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (16, 32), 'ori_format': 'ND',  "param_type": "input"},
               {'shape': (16, 32), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (16, 32), 'ori_format': 'ND',  "param_type": "input"},
               {'shape': (16, 32), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (16, 32), 'ori_format': 'ND',  "param_type": "output"},
               ],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{'shape': (4096, 1024), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (4096, 1024), 'ori_format': 'ND',  "param_type": "input"},
               {'shape': ((4096, 1024)), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (4096, 1024), 'ori_format': 'ND',  "param_type": "input"},
               {'shape': (4096, 1024), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (4096, 1024), 'ori_format': 'ND',  "param_type": "output"},
               ],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

if __name__ == '__main__':
    ut_case.run(["Ascend910"], simulator_mode="pv",
                simulator_lib_path="/home/maying/.mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")

