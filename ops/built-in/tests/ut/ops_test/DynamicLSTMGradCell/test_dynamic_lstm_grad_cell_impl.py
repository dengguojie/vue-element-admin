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

ut_case = OpUT("DynamicLSTMGradCell", "impl.dynamic.dynamic_lstm_grad_cell", "dynamic_lstm_grad_cell")

case1 = {
    "params": [
        {"shape": (1, 1, 4, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "init_c"},
        {"shape": (2, 1, 4, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "c"},
        {"shape": (2, 1, 4, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "dy"},
        {"shape": (1, 1, 4, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "dht"},
        {"shape": (1, 1, 4, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "dct"},
        {"shape": (2, 1, 4, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "it"},
        {"shape": (2, 1, 4, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "jt"},
        {"shape": (2, 1, 4, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "ft"},
        {"shape": (2, 1, 4, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "ot"},
        {"shape": (2, 1, 4, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "tanhct"},
        {"shape": (2, 1, 4, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "mask"},
        {"shape": (1,), "dtype": "int64", "format": "ND", "ori_shape": (1,), "ori_format": "ND",
         "desc": "t_state"},
        {"shape": (1, 4, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "dgate"},
        {"shape": (1, 1, 4, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "dct_1"},
        1.0, 'None', 'Forward', 'ijfo'
    ],
    "case_name": "dynamic_lstm_grad_cell_0",
    "expect": "success",
}
case2 = {
    "params": [
        {"shape": (1, 5, 10, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "init_c"},
        {"shape": (2, 5, 10, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "c"},
        {"shape": (2, 5, 10, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "dy"},
        {"shape": (1, 5, 10, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "dht"},
        {"shape": (1, 5, 10, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "dct"},
        {"shape": (2, 5, 10, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "it"},
        {"shape": (2, 5, 10, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "jt"},
        {"shape": (2, 5, 10, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "ft"},
        {"shape": (2, 5, 10, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "ot"},
        {"shape": (2, 5, 10, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "tanhct"},
        {"shape": (2, 5, 10, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "mask"},
        {"shape": (1,), "dtype": "int64", "format": "ND", "ori_shape": (1,), "ori_format": "ND",
         "desc": "t_state"},
        {"shape": (1, 20, 10, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "dgate"},
        {"shape": (1, 5, 10, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),
         "ori_format": "ND",
         "desc": "dct_1"},
        1.0, 'None', 'Forward', 'ijfo'
    ],
    "case_name": "dynamic_lstm_grad_cell_1",
    "expect": "success",
}

ut_case.add_case(['Ascend910A'], case1)
ut_case.add_case(['Ascend910A'], case2)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
    print('run test_dynamic_lstm_grad_cell end')
    exit(0)
