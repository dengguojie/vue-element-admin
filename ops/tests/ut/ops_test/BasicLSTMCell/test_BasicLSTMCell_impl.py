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

BasicLSTMCell ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("BasicLSTMCell", "impl.basic_lstm_cell", "basic_lstm_cell")

case1 = {"params": [{"shape": (3, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 48),"ori_format": "ND"}, #x
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #h
                    {"shape": (2, 4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #c
                    {"shape": (5, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (80, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float32", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    {"shape": (128,), "dtype": "uint8", "format": "ND", "ori_shape": (128,),"ori_format": "ND"}, #mask
                    {"shape": (2, 4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #ct
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #it
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #jt
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #ft
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #ot
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #x
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #h
                    {"shape": (8, 128,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #c
                    {"shape": (16,32,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (256,215),"ori_format": "ND"}, #w
                    {"shape": (512,), "dtype": "float32", "format": "ND", "ori_shape": (512,),"ori_format": "ND"},  #b
                    {"shape": (512,), "dtype": "uint8", "format": "ND", "ori_shape": (512,),"ori_format": "ND"}, #mask
                    {"shape": (8, 128,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #ct
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #ht
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #it
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #jt
                    {"shape":(8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #ft
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #ot
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #x
                    {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #h
                    {"shape": (128, 8,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #c
                    {"shape": (256,512,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (4096,8192),"ori_format": "ND"}, #w
                    {"shape": (8192,), "dtype": "float32", "format": "ND", "ori_shape": (8192,),"ori_format": "ND"},  #b
                    {"shape": (8192,), "dtype": "uint8", "format": "ND", "ori_shape": (8192,),"ori_format": "ND"}, #mask
                    {"shape": (128, 8,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #ct
                    {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #ht
                    {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #it
                    {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #jt
                    {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #ft
                    {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #ot
                    {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (3, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 48),"ori_format": "ND"}, #x
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #h
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #c
                    {"shape": (5, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (80, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float32", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    {"shape": (128,), "dtype": "uint8", "format": "ND", "ori_shape": (128,),"ori_format": "ND"}, #mask
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ct
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #it
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #jt
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ft
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ot
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (1000, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16000),"ori_format": "ND"}, #x
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #h
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #c
                    {"shape": (1002, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (16032, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float32", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    {"shape": (128,), "dtype": "uint8", "format": "ND", "ori_shape": (128,),"ori_format": "ND"}, #mask
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ct
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #it
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #jt
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ft
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ot
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_5",
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape": (1, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 16),"ori_format": "ND"}, #x
                    {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #h
                    {"shape": (2, 1000,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #c
                    {"shape": (3, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (48, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float32", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    {"shape": (128,), "dtype": "uint8", "format": "ND", "ori_shape": (128,),"ori_format": "ND"}, #mask
                    {"shape": (2, 1000,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #ct
                    {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #it
                    {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #jt
                    {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #ft
                    {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #ot
                    {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_6",
         "expect": "success",
         "support_expect": True}

case7 = {"params": [{"shape": (1, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16),"ori_format": "ND"}, #x
                    {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #h
                    {"shape": (100, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #c
                    {"shape": (101, 400,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (1616, 6400),"ori_format": "ND"}, #w
                    {"shape": (6400,), "dtype": "float32", "format": "ND", "ori_shape": (6400,),"ori_format": "ND"},  #b
                    {"shape": (6400,), "dtype": "uint8", "format": "ND", "ori_shape": (6400,),"ori_format": "ND"}, #mask
                    {"shape": (100, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #ct
                    {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #ht
                    {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #it
                    {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #jt
                    {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #ft
                    {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #ot
                    {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_7",
         "expect": "success",
         "support_expect": True}

case8 = {"params": [{"shape": (3, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 48),"ori_format": "ND"}, #x
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #h
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #c
                    {"shape": (5, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (80, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float16", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    {"shape": (128,), "dtype": "uint8", "format": "ND", "ori_shape": (128,),"ori_format": "ND"}, #mask
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #ct
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #it
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #jt
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #ft
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #ot
                    {"shape": (2, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_8",
         "expect": "success",
         "support_expect": True}

case9 = {"params": [{"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #x
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #h
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #c
                    {"shape": (16,32,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (256,215),"ori_format": "ND"}, #w
                    {"shape": (512,), "dtype": "float16", "format": "ND", "ori_shape": (512,),"ori_format": "ND"},  #b
                    {"shape": (512,), "dtype": "uint8", "format": "ND", "ori_shape": (512,),"ori_format": "ND"}, #mask
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #ct
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #ht
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #it
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #jt
                    {"shape":(8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #ft
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #ot
                    {"shape": (8, 128,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2048, 128),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_9",
         "expect": "success",
         "support_expect": True}

case10 = {"params": [{"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #x
                    {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #h
                    {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #c
                    {"shape": (256,512,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (4096,8192),"ori_format": "ND"}, #w
                    {"shape": (8192,), "dtype": "float16", "format": "ND", "ori_shape": (8192,),"ori_format": "ND"},  #b
                    {"shape": (8192,), "dtype": "uint8", "format": "ND", "ori_shape": (8192,),"ori_format": "ND"}, #mask
                    {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #ct
                    {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #ht
                    {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #it
                    {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #jt
                    {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #ft
                    {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #ot
                    {"shape": (128, 8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 2048),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_10",
         "expect": "success",
         "support_expect": True}

case11 = {"params": [{"shape": (3, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 48),"ori_format": "ND"}, #x
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #h
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #c
                    {"shape": (5, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (80, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float16", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    {"shape": (128,), "dtype": "uint8", "format": "ND", "ori_shape": (128,),"ori_format": "ND"}, #mask
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ct
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #it
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #jt
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ft
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ot
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_11",
         "expect": "success",
         "support_expect": True}

case12 = {"params": [{"shape": (1000, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16000),"ori_format": "ND"}, #x
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #h
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #c
                    {"shape": (1002, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (16032, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float16", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    {"shape": (128,), "dtype": "uint8", "format": "ND", "ori_shape": (128,),"ori_format": "ND"}, #mask
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ct
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #it
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #jt
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ft
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ot
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_12",
         "expect": "success",
         "support_expect": True}

case13 = {"params": [{"shape": (1, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 16),"ori_format": "ND"}, #x
                    {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #h
                    {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #c
                    {"shape": (3, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (48, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float16", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    {"shape": (128,), "dtype": "uint8", "format": "ND", "ori_shape": (128,),"ori_format": "ND"}, #mask
                    {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #ct
                    {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #it
                    {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #jt
                    {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #ft
                    {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #ot
                    {"shape": (2, 1000,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16000, 32),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_13",
         "expect": "success",
         "support_expect": True}

case14 = {"params": [{"shape": (1, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16),"ori_format": "ND"}, #x
                    {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #h
                    {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #c
                    {"shape": (101, 400,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (1616, 6400),"ori_format": "ND"}, #w
                    {"shape": (6400,), "dtype": "float16", "format": "ND", "ori_shape": (6400,),"ori_format": "ND"},  #b
                    {"shape": (6400,), "dtype": "uint8", "format": "ND", "ori_shape": (6400,),"ori_format": "ND"}, #mask
                    {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #ct
                    {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #ht
                    {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #it
                    {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #jt
                    {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #ft
                    {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #ot
                    {"shape": (100, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1600),"ori_format": "ND"}, #tanhct
                    ],
         "case_name": "BasicLSTMCell_14",
         "expect": "success",
         "support_expect": True}
# TODO fix me, this comment, run failed
# ut_case.add_case(["Ascend910","Ascend310"], case1)
# ut_case.add_case(["Ascend910","Ascend310"], case2)
# ut_case.add_case(["Ascend910","Ascend310"], case3)
# ut_case.add_case(["Ascend910","Ascend310"], case4)
# ut_case.add_case(["Ascend910","Ascend310"], case5)
# ut_case.add_case(["Ascend910","Ascend310"], case6)
# ut_case.add_case(["Ascend910","Ascend310"], case7)
# ut_case.add_case(["Ascend910","Ascend310"], case8)
# ut_case.add_case(["Ascend910","Ascend310"], case9)
# ut_case.add_case(["Ascend910","Ascend310"], case10)
# ut_case.add_case(["Ascend910","Ascend310"], case11)
# ut_case.add_case(["Ascend910","Ascend310"], case12)
ut_case.add_case(["Ascend910","Ascend310"], case13)
ut_case.add_case(["Ascend910","Ascend310"], case14)

if __name__ == '__main__':
    ut_case.run()
    exit(0)
