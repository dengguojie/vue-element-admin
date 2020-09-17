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

BasicLSTMCellInputGrad ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("BasicLSTMCellInputGrad", "impl.basic_lstm_cell_input_grad", "basic_lstm_cell_input_grad")


case1 = {"params": [{"shape": (16,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 256),"ori_format": "ND"}, #dgate
                    {"shape": (6,16,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (96, 256),"ori_format": "ND"}, #w
                    {"shape": (64,), "dtype": "uint8", "format": "ND", "ori_shape": (64,),"ori_format": "ND"}, #dropout_mask
                    {"shape": (2,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,32),"ori_format": "ND"}, #dxt
                    {"shape": (4,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,64),"ori_format": "ND"}, #dht
                    ],
         "case_name": "BasicLSTMCellInputGrad_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (16,100,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1600, 256),"ori_format": "ND"}, #dgate
                    {"shape": (6,16,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (96, 256),"ori_format": "ND"}, #w
                    {"shape": (6400,), "dtype": "uint8", "format": "ND", "ori_shape": (6400,),"ori_format": "ND"}, #dropout_mask
                    {"shape": (2,100,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1600,32),"ori_format": "ND"}, #dxt
                    {"shape": (4,100,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1600,64),"ori_format": "ND"}, #dht
                    ],
         "case_name": "BasicLSTMCellInputGrad_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (16,16,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (256, 256),"ori_format": "ND"}, #dgate
                    {"shape": (6,16,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (96, 256),"ori_format": "ND"}, #w
                    {"shape": (1024,), "dtype": "uint8", "format": "ND", "ori_shape": (1024,),"ori_format": "ND"}, #dropout_mask
                    {"shape": (2,16,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (256,32),"ori_format": "ND"}, #dxt
                    {"shape": (4,16,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (256,64),"ori_format": "ND"}, #dht
                    ],
         "case_name": "BasicLSTMCellInputGrad_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (16,128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 256),"ori_format": "ND"}, #dgate
                    {"shape": (6,16,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (96, 256),"ori_format": "ND"}, #w
                    {"shape": (8192,), "dtype": "uint8", "format": "ND", "ori_shape": (8192,),"ori_format": "ND"}, #dropout_mask
                    {"shape": (2,128,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2048,32),"ori_format": "ND"}, #dxt
                    {"shape": (4,128,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2048,64),"ori_format": "ND"}, #dht
                    ],
         "case_name": "BasicLSTMCellInputGrad_4",
         "expect": "success",
         "support_expect": True}


case5 = {"params": [{"shape": (16,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 256),"ori_format": "ND"}, #dgate
                    {"shape": (12,16,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (192, 256),"ori_format": "ND"}, #w
                    {"shape": (256,), "dtype": "uint8", "format": "ND", "ori_shape": (256,),"ori_format": "ND"}, #dropout_mask
                    {"shape": (8,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,128),"ori_format": "ND"}, #dxt
                    {"shape": (4,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,64),"ori_format": "ND"}, #dht
                    ],
         "case_name": "BasicLSTMCellInputGrad_5",
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape": (16,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 256),"ori_format": "ND"}, #dgate
                    {"shape": (260,16,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (4160, 256),"ori_format": "ND"}, #w
                    {"shape": (8192,), "dtype": "uint8", "format": "ND", "ori_shape": (8192,),"ori_format": "ND"}, #dropout_mask
                    {"shape": (256,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,4096),"ori_format": "ND"}, #dxt
                    {"shape": (4,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,64),"ori_format": "ND"}, #dht
                    ],
         "case_name": "BasicLSTMCellInputGrad_6",
         "expect": "success",
         "support_expect": True}


### 1600 2048 2048
case7 = {"params": [{"shape": (512,100,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1600, 8192),"ori_format": "ND"}, #dgate
                    {"shape": (256,512,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (4096, 8192),"ori_format": "ND"}, #w
                    {"shape": (409600,), "dtype": "uint8", "format": "ND", "ori_shape": (409600,),"ori_format": "ND"}, #dropout_mask
                    {"shape": (128,100,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1600,2048),"ori_format": "ND"}, #dxt
                    {"shape": (128,100,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1600,2048),"ori_format": "ND"}, #dht
                    ],
         "case_name": "BasicLSTMCellInputGrad_7",
         "expect": "success",
         "support_expect": True}

### 64 128 2048
case8 = {"params": [{"shape": (512,4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 8192),"ori_format": "ND"}, #dgate
                    {"shape": (136,512,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (2176, 8192),"ori_format": "ND"}, #w
                    {"shape": (1024,), "dtype": "uint8", "format": "ND", "ori_shape": (1024,),"ori_format": "ND"}, #dropout_mask
                    {"shape": (8,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64,128),"ori_format": "ND"}, #dxt
                    {"shape": (128,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64,2048),"ori_format": "ND"}, #dht
                    ],
         "case_name": "BasicLSTMCellInputGrad_8",
         "expect": "success",
         "support_expect": True}


case9 = {"params": [{"shape": (512,4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 8192),"ori_format": "ND"}, #dgate
                    {"shape": (136,512,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (2176, 8192),"ori_format": "ND"}, #w
                    {"shape": (1024,), "dtype": "uint8", "format": "ND", "ori_shape": (1024,),"ori_format": "ND"}, #dropout_mask
                    {"shape": (8,4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64,128),"ori_format": "ND"}, #dxt
                    {"shape": (128,4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64,2048),"ori_format": "ND"}, #dht
                    ],
         "case_name": "BasicLSTMCellInputGrad_9",
         "expect": "success",
         "support_expect": True}


case10 = {"params": [{"shape": (1024,128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 16384),"ori_format": "ND"}, #dgate
                    {"shape": (512,1024,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (8192, 16384),"ori_format": "ND"}, #w
                    None, #dropout_mask
                    {"shape": (256,128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048,4096),"ori_format": "ND"}, #dxt
                    {"shape": (256,128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048,4096),"ori_format": "ND"}, #dht
                    ],
         "case_name": "BasicLSTMCellInputGrad_10",
         "expect": "success",
         "support_expect": True}


ut_case.add_case(["Ascend910"], case1)
ut_case.add_case(["Ascend910"], case2)
ut_case.add_case(["Ascend910"], case3)
# TODO fix me, run failed
# ut_case.add_case(["Ascend910"], case4)
ut_case.add_case(["Ascend910"], case5)
ut_case.add_case(["Ascend910"], case6)
ut_case.add_case(["Ascend910"], case7)
ut_case.add_case(["Ascend910"], case8)
ut_case.add_case(["Ascend910"], case9)
ut_case.add_case(["Ascend910"], case10)

if __name__ == '__main__':
    ut_case.run("Ascend910")
