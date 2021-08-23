from op_test_frame.ut import OpUT

ut_case = OpUT("BasicLSTMCellV2", "impl.basic_lstm_cell_v2", "basic_lstm_cell_v2")


case1 = {"params": [{"shape": (1, 3, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 48),"ori_format": "ND"}, #x
                    {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    None,
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #h
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #c
                    {"shape": (5, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (80, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float16", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    None, #mask
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"} #ct
                    ],
         "case_name": "BasicLSTMCell_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (1, 1000, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 48),"ori_format": "ND"}, #x
                    {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    None,
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #h
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #c
                    {"shape": (1002, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (80, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float16", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    None, #mask
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"} #ct
                    ],
         "case_name": "BasicLSTMCell_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (1, 1000, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 48),"ori_format": "ND"}, #x
                    {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    None,
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #h
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #c
                    {"shape": (1002, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (80, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float16", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    None, #mask
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ct
                    4
                    ],
         "case_name": "BasicLSTMCell_2",
         "expect": RuntimeError,
         "support_expect": True}

case4 = {"params": [{"shape": (1, 1000, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 48),"ori_format": "ND"}, #x
                    {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    None,
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #h
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #c
                    {"shape": (1002, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (80, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float16", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    None, #mask
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"} #ct
                    ],
         "case_name": "BasicLSTMCell_2",
         "expect": RuntimeError,
         "support_expect": True}

case5 = {"params": [{"shape": (1, 1000, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 48),"ori_format": "ND"}, #x
                    {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    None,
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #h
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #c
                    {"shape": (1002, 9,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (80, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float16", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    None, #mask
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"} #ct
                    ],
         "case_name": "BasicLSTMCell_2",
         "expect": RuntimeError,
         "support_expect": True}

case6 = {"params": [{"shape": (1, 1000, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 48),"ori_format": "ND"}, #x
                    {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    None,
                    None, #h
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #c
                    {"shape": (1002, 9,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (80, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float16", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    None, #mask
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ct
                    0,
                    True
                    ],
         "case_name": "BasicLSTMCell_2",
         "expect": RuntimeError,
         "support_expect": True}

case7 = {"params": [{"shape": (1, 1000, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 48),"ori_format": "ND"}, #x
                    {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    None,
                    {"shape": (2, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),
                     "ori_format": "ND"}, #h
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #c
                    {"shape": (1002, 9,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (80, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float16", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    None, #mask
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ct
                    0,
                    True
                    ],
         "case_name": "BasicLSTMCell_2",
         "expect": RuntimeError,
         "support_expect": True}

case8 = {"params": [{"shape": (1, 1000, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 48),"ori_format": "ND"}, #x
                    {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    None,
                    {"shape": (2, 1, 16, 16), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (1, 32),
                     "ori_format": "ND"}, #h
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #c
                    {"shape": (1002, 9,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (80, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float16", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    None, #mask
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ct
                    0,
                    True
                    ],
         "case_name": "BasicLSTMCell_2",
         "expect": RuntimeError,
         "support_expect": True}

'''
(x=None, cont=None, w_xc_x_static=None,
                       h=None, c=None, w_xh=None, bias=None,
                       w_xh_deqscale=None,
                       h_t=None, c_t=None, num_output=0,
                       expose_hidden=False, xh_scale=0.0,
                       sqrt_mode=False, xh_offset=0, w_xh_offset=0,
                       kernel_name="BasicLSTMCellV2",
                       impl_mode="high_performance"):
'''

ut_case.add_case("Ascend910A", case1)
ut_case.add_case("Ascend910A", case2)
ut_case.add_case("Ascend910A", case3)
ut_case.add_case("Ascend910A", case4)
ut_case.add_case("Ascend910A", case5)
ut_case.add_case("Ascend910A", case6)
ut_case.add_case("Ascend910A", case7)
ut_case.add_case("Ascend910A", case8)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
