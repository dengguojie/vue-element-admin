"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

euclidean_norm_d ut case
"""
from op_test_frame.ut import ReduceOpUT
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = ReduceOpUT("EuclideanNormD", None, None)

ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1,), (0,), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1,), 0, False)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1,), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 1), (1,), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 1), (1,), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1023*25, ), (-1, ), False)

ut_case.add_reduce_case_simple(["Ascend310"], ["float16", "int32"], (1,), (0,), True)
ut_case.add_reduce_case_simple(["Ascend310"], ["float16", "int32"], (1,), 0, False)
ut_case.add_reduce_case_simple(["Ascend310"], ["float16", "int32"], (1, 1), (1,), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "int32"], (5,6,3,4), (0,2), False)

def calc_expect_func(x, output, axis, keepdims):
    shape_x=x.get("shape")
    x_value=x.get("value")
    res_mul=np.multiply(x_value, x_value)
    res_sum=np.sum(res_mul, axis=axis, keepdims=keepdims)
    output=np.sqrt(res_sum)
    return output

ut_case.add_precision_case("all", {
    "params": [{'shape': (1,), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND',  "param_type": "input"},
               {'shape': (1,), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND',  "param_type": "output"},
               0, True,
               ],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{'shape': (1, 1), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1, 1), 'ori_format': 'ND',  "param_type": "input"},
               {'shape': (1, 1), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1, 1), 'ori_format': 'ND',  "param_type": "output"},
               (1,), True,
               ],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{'shape': (1, 1), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1, 1), 'ori_format': 'ND',  "param_type": "input"},
               {'shape': (1,), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND',  "param_type": "output"},
               (1,), False,
               ],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{'shape': (2,), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (2,), 'ori_format': 'ND',  "param_type": "input"},
               {'shape': (1,), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND',  "param_type": "output"},
               0, True,
               ],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{'shape': (2,), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (2,), 'ori_format': 'ND',  "param_type": "input"},
               {'shape': (1,), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND',  "param_type": "output"},
               (-1,), True,
               ],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{'shape': (1023*25, ), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (1023*25, ), 'ori_format': 'ND',  "param_type": "input"},
               {'shape': (1, ), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (1, ), 'ori_format': 'ND',  "param_type": "output"},
               (-1,), True,
               ],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{'shape': (5,6,3,4), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (5,6,3,4), 'ori_format': 'ND',  "param_type": "input"},
               {'shape': (6,4), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (6,4), 'ori_format': 'ND',  "param_type": "output"},
               (0,2), False,
               ],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)

