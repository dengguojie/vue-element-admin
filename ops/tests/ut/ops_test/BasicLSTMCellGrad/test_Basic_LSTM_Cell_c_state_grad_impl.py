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

BasicLSTMCellCStateGrad ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("BasicLSTMCellCStateGrad", "impl.basic_lstm_cell_c_state_grad", "basic_lstm_cell_c_state_grad")

case1 = {"params": [{"shape": (4,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #c
                    {"shape": (4,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #dht
                    {"shape": (4,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #dct
                    {"shape": (4,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #it
                    {"shape": (4,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #jt
                    {"shape": (4,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #ft
                    {"shape": (4,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #ot
                    {"shape": (4,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #tanhct
                    {"shape": (4,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #dgate
                    {"shape": (4,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #dct_1
                    ],
         "case_name": "BasicLSTMCellCStateGrad_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (128,128,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2048, 2048),"ori_format": "ND"}, #c
                    {"shape": (128,128,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2048, 2048),"ori_format": "ND"}, #dht
                    {"shape": (128,128,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2048, 2048),"ori_format": "ND"}, #dct
                    {"shape": (128,128,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2048, 2048),"ori_format": "ND"}, #it
                    {"shape": (128,128,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2048, 2048),"ori_format": "ND"}, #jt
                    {"shape": (128,128,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2048, 2048),"ori_format": "ND"}, #ft
                    {"shape": (128,128,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2048, 2048),"ori_format": "ND"}, #ot
                    {"shape": (128,128,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2048, 2048),"ori_format": "ND"}, #tanhct
                    {"shape": (128,128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 2048),"ori_format": "ND"}, #dgate
                    {"shape": (128,128,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2048, 2048),"ori_format": "ND"}, #dct_1
                    ],
         "case_name": "BasicLSTMCellCStateGrad_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (1,256,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (4096, 16),"ori_format": "ND"}, #c
                    {"shape": (1,256,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (4096, 16),"ori_format": "ND"}, #dht
                    {"shape": (1,256,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (4096, 16),"ori_format": "ND"}, #dct
                    {"shape": (1,256,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (4096, 16),"ori_format": "ND"}, #it
                    {"shape": (1,256,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (4096, 16),"ori_format": "ND"}, #jt
                    {"shape": (1,256,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (4096, 16),"ori_format": "ND"}, #ft
                    {"shape": (1,256,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (4096, 16),"ori_format": "ND"}, #ot
                    {"shape": (1,256,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (4096, 16),"ori_format": "ND"}, #tanhct
                    {"shape": (1,256,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4096, 16),"ori_format": "ND"}, #dgate
                    {"shape": (1,256,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (4096, 16),"ori_format": "ND"}, #dct_1
                    ],
         "case_name": "BasicLSTMCellCStateGrad_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (4,1000,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16000, 64),"ori_format": "ND"}, #c
                    {"shape": (4,1000,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16000, 64),"ori_format": "ND"}, #dht
                    {"shape": (4,1000,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16000, 64),"ori_format": "ND"}, #dct
                    {"shape": (4,1000,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16000, 64),"ori_format": "ND"}, #it
                    {"shape": (4,1000,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16000, 64),"ori_format": "ND"}, #jt
                    {"shape": (4,1000,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16000, 64),"ori_format": "ND"}, #ft
                    {"shape": (4,1000,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16000, 64),"ori_format": "ND"}, #ot
                    {"shape": (4,1000,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16000, 64),"ori_format": "ND"}, #tanhct
                    {"shape": (4,1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 64),"ori_format": "ND"}, #dgate
                    {"shape": (4,1000,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16000, 64),"ori_format": "ND"}, #dct_1
                    ],
         "case_name": "BasicLSTMCellCStateGrad_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (4000,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64, 64000),"ori_format": "ND"}, #c
                    {"shape": (4000,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64, 64000),"ori_format": "ND"}, #dht
                    {"shape": (4000,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64, 64000),"ori_format": "ND"}, #dct
                    {"shape": (4000,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64, 64000),"ori_format": "ND"}, #it
                    {"shape": (4000,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64, 64000),"ori_format": "ND"}, #jt
                    {"shape": (4000,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64, 64000),"ori_format": "ND"}, #ft
                    {"shape": (4000,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64, 64000),"ori_format": "ND"}, #ot
                    {"shape": (4000,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64, 64000),"ori_format": "ND"}, #tanhct
                    {"shape": (4000,4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 64000),"ori_format": "ND"}, #dgate
                    {"shape": (4000,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64, 64000),"ori_format": "ND"}, #dct_1
                    ],
         "case_name": "BasicLSTMCellCStateGrad_5",
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape": (4,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #c
                    {"shape": (4,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #dht
                    {"shape": (4,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #dct
                    {"shape": (4,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #it
                    {"shape": (4,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #jt
                    {"shape": (4,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #ft
                    {"shape": (4,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #ot
                    {"shape": (4,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #tanhct
                    {"shape": (4,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #dgate
                    {"shape": (4,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 64),"ori_format": "ND"}, #dct_1
                    ],
         "case_name": "BasicLSTMCellCStateGrad_6",
         "expect": "success",
         "support_expect": True}

case7 = {"params": [{"shape": (128,128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 2048),"ori_format": "ND"}, #c
                    {"shape": (128,128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 2048),"ori_format": "ND"}, #dht
                    {"shape": (128,128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 2048),"ori_format": "ND"}, #dct
                    {"shape": (128,128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 2048),"ori_format": "ND"}, #it
                    {"shape": (128,128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 2048),"ori_format": "ND"}, #jt
                    {"shape": (128,128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 2048),"ori_format": "ND"}, #ft
                    {"shape": (128,128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 2048),"ori_format": "ND"}, #ot
                    {"shape": (128,128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 2048),"ori_format": "ND"}, #tanhct
                    {"shape": (128,128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 2048),"ori_format": "ND"}, #dgate
                    {"shape": (128,128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 2048),"ori_format": "ND"}, #dct_1
                    ],
         "case_name": "BasicLSTMCellCStateGrad_7",
         "expect": "success",
         "support_expect": True}

case8 = {"params": [{"shape": (1,256,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4096, 16),"ori_format": "ND"}, #c
                    {"shape": (1,256,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4096, 16),"ori_format": "ND"}, #dht
                    {"shape": (1,256,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4096, 16),"ori_format": "ND"}, #dct
                    {"shape": (1,256,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4096, 16),"ori_format": "ND"}, #it
                    {"shape": (1,256,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4096, 16),"ori_format": "ND"}, #jt
                    {"shape": (1,256,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4096, 16),"ori_format": "ND"}, #ft
                    {"shape": (1,256,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4096, 16),"ori_format": "ND"}, #ot
                    {"shape": (1,256,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4096, 16),"ori_format": "ND"}, #tanhct
                    {"shape": (1,256,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4096, 16),"ori_format": "ND"}, #dgate
                    {"shape": (1,256,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4096, 16),"ori_format": "ND"}, #dct_1
                    ],
         "case_name": "BasicLSTMCellCStateGrad_8",
         "expect": "success",
         "support_expect": True}

case9 = {"params": [{"shape": (4,1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 64),"ori_format": "ND"}, #c
                    {"shape": (4,1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 64),"ori_format": "ND"}, #dht
                    {"shape": (4,1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 64),"ori_format": "ND"}, #dct
                    {"shape": (4,1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 64),"ori_format": "ND"}, #it
                    {"shape": (4,1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 64),"ori_format": "ND"}, #jt
                    {"shape": (4,1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 64),"ori_format": "ND"}, #ft
                    {"shape": (4,1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 64),"ori_format": "ND"}, #ot
                    {"shape": (4,1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 64),"ori_format": "ND"}, #tanhct
                    {"shape": (4,1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 64),"ori_format": "ND"}, #dgate
                    {"shape": (4,1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 64),"ori_format": "ND"}, #dct_1
                    ],
         "case_name": "BasicLSTMCellCStateGrad_9",
         "expect": "success",
         "support_expect": True}

case10 = {"params": [{"shape": (4000,4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 64000),"ori_format": "ND"}, #c
                    {"shape": (4000,4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 64000),"ori_format": "ND"}, #dht
                    {"shape": (4000,4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 64000),"ori_format": "ND"}, #dct
                    {"shape": (4000,4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 64000),"ori_format": "ND"}, #it
                    {"shape": (4000,4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 64000),"ori_format": "ND"}, #jt
                    {"shape": (4000,4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 64000),"ori_format": "ND"}, #ft
                    {"shape": (4000,4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 64000),"ori_format": "ND"}, #ot
                    {"shape": (4000,4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 64000),"ori_format": "ND"}, #tanhct
                    {"shape": (4000,4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 64000),"ori_format": "ND"}, #dgate
                    {"shape": (4000,4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 64000),"ori_format": "ND"}, #dct_1
                    ],
         "case_name": "BasicLSTMCellCStateGrad_10",
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
ut_case.add_case(["Ascend910"], case9)
ut_case.add_case(["Ascend910"], case10)

if __name__ == '__main__':
    ut_case.run()
    exit(0)
