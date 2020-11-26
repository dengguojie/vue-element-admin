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




ut_case.add_case("Ascend910", case1)
ut_case.add_case("Ascend910", case2)


if __name__ == '__main__':
    ut_case.run("Ascend910")