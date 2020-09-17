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

BasicLSTMCellWeightGrad ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("BasicLSTMCellWeightGrad", "impl.basic_lstm_cell_weight_grad", "basic_lstm_cell_weight_grad")

case1 = {"params": [{"shape": (2,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 32),"ori_format": "ND"}, #x
                    {"shape": (4,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #h
                    {"shape": (16,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 256),"ori_format": "ND"}, #dgate
                    {"shape": (6,16,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (96, 256),"ori_format": "ND"}, #dw
                    {"shape": (256,), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (256,),"ori_format": "ND"}, #db
                    ],
         "case_name": "BasicLSTMCellWeightGrad_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (200,64,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1024, 3200),"ori_format": "ND"}, #x
                    {"shape": (4,64,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1024, 64),"ori_format": "ND"}, #h
                    {"shape": (16,64,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1024, 256),"ori_format": "ND"}, #dgate
                    {"shape": (204,16,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (3264, 256),"ori_format": "ND"}, #dw
                    {"shape": (256,), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (256,),"ori_format": "ND"}, #db
                    ],
         "case_name": "BasicLSTMCellWeightGrad_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (7,2,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 112),"ori_format": "ND"}, #x
                    {"shape": (3,2,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 48),"ori_format": "ND"}, #h
                    {"shape": (12,2,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 192),"ori_format": "ND"}, #dgate
                    {"shape": (10,12,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (160, 192),"ori_format": "ND"}, #dw
                    {"shape": (192,), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (192,),"ori_format": "ND"}, #db
                    ],
         "case_name": "BasicLSTMCellWeightGrad_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (400,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 6400),"ori_format": "ND"}, #x
                    {"shape": (1,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 16),"ori_format": "ND"}, #h
                    {"shape": (4,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 64),"ori_format": "ND"}, #dgate
                    {"shape": (401,4,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (6416, 64),"ori_format": "ND"}, #dw
                    {"shape": (64,), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64,),"ori_format": "ND"}, #db
                    ],
         "case_name": "BasicLSTMCellWeightGrad_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (400,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 6400),"ori_format": "ND"}, #x
                    {"shape": (1,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 16),"ori_format": "ND"}, #h
                    {"shape": (4,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 64),"ori_format": "ND"}, #dgate
                    {"shape": (401,4,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (6416, 64),"ori_format": "ND"}, #dw
                    {"shape": (64,), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64,),"ori_format": "ND"}, #db
                    ],
         "case_name": "BasicLSTMCellWeightGrad_5",
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape": (7,32,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (512, 112),"ori_format": "ND"}, #x
                    {"shape": (3,32,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (512, 48),"ori_format": "ND"}, #h
                    {"shape": (12,32,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (512, 192),"ori_format": "ND"}, #dgate
                    {"shape": (10,12,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (160, 192),"ori_format": "ND"}, #dw
                    {"shape": (192,), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (192,),"ori_format": "ND"}, #db
                    ],
         "case_name": "BasicLSTMCellWeightGrad_6",
         "expect": "success",
         "support_expect": True}

case7 = {"params": [{"shape": (2,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 32),"ori_format": "ND"}, #x
                    {"shape": (400,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 6400),"ori_format": "ND"}, #h
                    {"shape": (1600,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 25600),"ori_format": "ND"}, #dgate
                    {"shape": (402,1600,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (6432, 25600),"ori_format": "ND"}, #dw
                    {"shape": (25600,), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (25600,),"ori_format": "ND"}, #db
                    ],
         "case_name": "BasicLSTMCellWeightGrad_7",
         "expect": "success",
         "support_expect": True}

case8 = {"params": [{"shape": (7,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 112),"ori_format": "ND"}, #x
                    {"shape": (3,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 48),"ori_format": "ND"}, #h
                    {"shape": (12,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 192),"ori_format": "ND"}, #dgate
                    {"shape": (10,12,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (160, 192),"ori_format": "ND"}, #dw
                    {"shape": (192,), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (192,),"ori_format": "ND"}, #db
                    ],
         "case_name": "BasicLSTMCellWeightGrad_8",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910"], case1)
ut_case.add_case(["Ascend910"], case2)
ut_case.add_case(["Ascend910"], case3)
ut_case.add_case(["Ascend910"], case4)
ut_case.add_case(["Ascend910"], case5)
ut_case.add_case(["Ascend910"], case6)
ut_case.add_case(["Ascend910"], case7)
ut_case.add_case(["Ascend910"], case8)

if __name__ == '__main__':
    ut_case.run("Ascend910")
