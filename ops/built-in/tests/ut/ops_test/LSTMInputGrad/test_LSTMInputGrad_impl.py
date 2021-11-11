from op_test_frame.ut import OpUT

ut_case = OpUT("LstmInputGrad", None, None)


case1 = {"params": [{"shape": (128,96,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (1536,2048),"ori_format": "ND"},
                    {"shape": (32,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 512),"ori_format": "ND"},
                    {"shape": (1,32,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 512),"ori_format": "ND"},
                    {"shape": (1,32,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 512),"ori_format": "ND"},
                    {"shape": (1,32,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 512),"ori_format": "ND"},
                    {"shape": (1,32,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 512),"ori_format": "ND"},
                    {"shape": (1,32,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 512),"ori_format": "ND"},
                    {"shape": (1,32,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 512),"ori_format": "ND"},
                    {"shape": (1,32,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 512),"ori_format": "ND"},
                    {"shape": (1,32,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 512),"ori_format": "ND"},
                    {"shape": (1,32,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 512),"ori_format": "ND"},
                    {"shape": (1,64,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 16, 512),"ori_format": "ND"},
                    {"shape": (1,32,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 512),"ori_format": "ND"}, #h
                    {"shape": (1,32,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 512),"ori_format": "ND"}, #c
                    {"shape": (1,128,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 512),"ori_format": "ND"}
                    ],
         "case_name": "lstm_inputgrad_1",
         "expect": IndexError,
         "support_expect": True}


ut_case.add_case("Ascend910A", case1)
# ut_case.add_case("Ascend910", case2)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
