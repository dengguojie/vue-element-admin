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
import sys
import time
import unittest
ut_case = OpUT("BasicLSTMCellCStateGradV2", "impl.basic_lstm_cell_c_state_grad_v2", "basic_lstm_cell_c_state_grad_v2")

case1 = {"params": [{"shape": (1,1,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1,1,4,16,16),"ori_format": "FRACTAL_NZ"},
                    {"shape": (1,1,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1,1,4,16,16),"ori_format": "FRACTAL_NZ"},
                    {"shape": (1,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1,1,4,16,16),"ori_format": "FRACTAL_NZ"},
                    {"shape": (1,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1,1,4,16,16),"ori_format": "FRACTAL_NZ"},
                    {"shape": (1,1,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1,1,4,16,16),"ori_format": "FRACTAL_NZ"},
                    {"shape": (1,1,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1,1,4,16,16),"ori_format": "FRACTAL_NZ"},
                    {"shape": (1,1,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1,1,4,16,16),"ori_format": "FRACTAL_NZ"},
                    {"shape": (1,1,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1,1,4,16,16),"ori_format": "FRACTAL_NZ"},
                    {"shape": (1,1,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1,1,4,16,16),"ori_format": "FRACTAL_NZ"},
                    {"shape": (1,4,4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1,1,4,16,16),"ori_format": "FRACTAL_NZ"},
                    {"shape": (1,1,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1,1,4,16,16),"ori_format": "FRACTAL_NZ"},
                    1.0,"None",
                    ],
         "case_name": "BasicLSTMCellCStateGradV2_1",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910A"], case1)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
    exit(0)

