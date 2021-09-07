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

BasicLstmCellCStateGrad ut case
"""
from op_test_frame.ut import OpUT
import sys
import time
import unittest

ut_case = OpUT("DynamicGRUCellGrad", "impl.dynamic.dynamic_gru_cell_grad", "dynamic_gru_cell_grad")

case1 = {
    "params": [
        {"shape": (32, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 512),
         "ori_format": "ND",
         "desc": "dh_pre_t"},
        {"shape": (1, 32, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16, 512),
         "ori_format": "ND",
         "desc": "h"},
        {"shape": (1, 32, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16, 512),
         "ori_format": "ND",
         "desc": "dy"},
        {"shape": (1, 32, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16, 512),
         "ori_format": "ND",
         "desc": "dh"},
        {"shape": (1, 32, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16, 512),
         "ori_format": "ND",
         "desc": "update"},
        {"shape": (1, 32, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16, 512),
         "ori_format": "ND",
         "desc": "reset"},
        {"shape": (1, 32, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16, 512),
         "ori_format": "ND",
         "desc": "new"},
        {"shape": (1, 32, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16, 512),
         "ori_format": "ND",
         "desc": "hidden_new"},
        {"shape": (32, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 512),
         "ori_format": "ND",
         "desc": "init_h"},
        {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,), "ori_format": "ND",
         "desc": "t_state"},
        {"shape": (32, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 512),
         "ori_format": "ND",
         "desc": "dh_prev"},
        {"shape": (1, 96, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16, 1536),
         "ori_format": "ND",
         "desc": "dgate_h"},
        {"shape": (1, 32, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16, 512),
         "ori_format": "ND",
         "desc": "dnt_x"},
        'zrh'
    ],
    "case_name": "dynamic_gru_cell_grad_1",
    "expect": "success",
}
case2 = {
    "params": [
        {"shape": (1, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 16),
         "ori_format": "ND",
         "desc": "dh_pre_t"},
        {"shape": (1, 1, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16, 16),
         "ori_format": "ND",
         "desc": "h"},
        {"shape": (1, 1, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16, 16),
         "ori_format": "ND",
         "desc": "dy"},
        {"shape": (1, 1, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16, 16),
         "ori_format": "ND",
         "desc": "dh"},
        {"shape": (1, 1, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16, 16),
         "ori_format": "ND",
         "desc": "update"},
        {"shape": (1, 1, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16, 16),
         "ori_format": "ND",
         "desc": "reset"},
        {"shape": (1, 1, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16, 16),
         "ori_format": "ND",
         "desc": "new"},
        {"shape": (1, 1, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16, 16),
         "ori_format": "ND",
         "desc": "hidden_new"},
        {"shape": (1, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 16),
         "ori_format": "ND",
         "desc": "init_h"},
        {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,), "ori_format": "ND",
         "desc": "t_state"},
        {"shape": (1, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 16),
         "ori_format": "ND",
         "desc": "dh_prev"},
        {"shape": (1, 3, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 3, 1, 16, 16),
         "ori_format": "FRACTAL_NZ",
         "desc": "dgate_h"},
        {"shape": (1, 1, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1, 1, 16, 16),
         "ori_format": "FRACTAL_NZ",
         "desc": "dnt_x"},
        'zrh'
    ],
    "case_name": "dynamic_gru_cell_grad_2",
    "expect": "success",
}

ut_case.add_case(['Ascend910A'], case1)
ut_case.add_case(['Ascend910A'], case2)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
    print('run test_dynamic_gru_cell_grad end')
    exit(0)
